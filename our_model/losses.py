from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class FeedbackFlowLossWeights:
    stability: float = 0.0
    ignore_index: int = -100


def _valid_mask_from_labels(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return labels.ne(int(ignore_index))


def _masked_average(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits prefix {tuple(logits.shape[:2])} does not match labels {tuple(labels.shape)}.")
    if mask is None:
        mask = _valid_mask_from_labels(labels, ignore_index)
    valid = mask.to(labels.device).bool() & _valid_mask_from_labels(labels, ignore_index)
    if not valid.any():
        return logits.sum() * 0.0
    safe_labels = labels.masked_fill(~valid, int(ignore_index))
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        safe_labels.reshape(-1),
        ignore_index=int(ignore_index),
        reduction="none",
    )
    return _masked_average(per_token.view_as(labels), valid)


def masked_hidden_mse(
    raw_hidden: torch.Tensor,
    steered_hidden: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if raw_hidden.shape != steered_hidden.shape:
        raise ValueError("raw_hidden and steered_hidden must have identical shape.")
    per_token = (steered_hidden.float() - raw_hidden.detach().float()).pow(2).mean(dim=-1)
    return _masked_average(per_token, mask)


def feedback_flow_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    state=None,
    target_mask: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    weights: FeedbackFlowLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generation objective plus a hidden-state stability regularizer."""

    weights = FeedbackFlowLossWeights() if weights is None else weights
    loss = logits.new_zeros(())
    metrics: dict[str, float] = {}

    gen_mask = target_mask
    if gen_mask is None:
        gen_mask = _valid_mask_from_labels(labels, weights.ignore_index)
    gen = masked_token_cross_entropy(logits, labels, gen_mask, weights.ignore_index)
    loss = loss + gen
    metrics["generation_loss"] = float(gen.detach().cpu())

    if weights.stability:
        if state is None or state.last_raw_hidden is None or state.last_steered_hidden is None:
            raise ValueError("stability loss requires a FeedbackFlowState with cached hidden states.")
        stable_mask = response_mask
        stable = masked_hidden_mse(state.last_raw_hidden, state.last_steered_hidden, stable_mask)
        loss = loss + float(weights.stability) * stable
        metrics["stability_loss"] = float(stable.detach().cpu())

    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics
