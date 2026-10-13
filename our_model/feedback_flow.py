from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import nn
from torch.nn import functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model.backbone import forward_model  # noqa: E402
from model.steering import model_metadata, transformer_layers  # noqa: E402


FEEDBACK_FLOW_ARTIFACT_VERSION = 3
FEEDBACK_FLOW_ARTIFACT_KIND = "llada_previous_prediction_feedback_flow"
SUPPORTED_FEEDBACK_FLOW_ARTIFACT_VERSIONS = {1, 2, FEEDBACK_FLOW_ARTIFACT_VERSION}


@torch.no_grad()
def normalized_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return detached H = -sum(p * log(p)) / log(V) for each token."""
    if logits.dim() != 3 or logits.size(-1) < 1:
        raise ValueError("logits must have shape [batch, seq, vocab] with a nonempty vocabulary.")
    if logits.size(-1) == 1:
        return torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    probabilities = F.softmax(logits.float(), dim=-1)
    entropy = torch.special.entr(probabilities).sum(dim=-1) / math.log(logits.size(-1))
    return entropy.clamp(0.0, 1.0)


@dataclass
class FeedbackFlowConfig:
    hidden_size: int
    num_heads: int = 8
    steps: int = 3
    ff_multiplier: float = 4.0
    dropout: float = 0.0
    timestep_gate_hidden: int = 128


class TimestepResidualGate(nn.Module):
    """Compute gamma[s,i] = sigmoid(f(s) + softplus(g(s)) * (H[s+1,i] - 0.5))."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("timestep_gate_hidden must be positive.")
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.entropy_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.entropy_net[-1].weight)
        nn.init.zeros_(self.entropy_net[-1].bias)

    def forward(
        self,
        timestep: torch.Tensor | int | float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        previous_entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=device, dtype=torch.float32)
        else:
            timestep = timestep.to(device=device, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.expand(int(batch_size))
        elif timestep.dim() == 2 and timestep.size(-1) == 1:
            timestep = timestep.squeeze(-1)
        elif timestep.dim() != 1:
            raise ValueError(f"timestep must be scalar or [batch], got shape {tuple(timestep.shape)}.")
        if timestep.numel() == 1 and int(batch_size) != 1:
            timestep = timestep.expand(int(batch_size))
        if timestep.numel() != int(batch_size):
            raise ValueError(f"timestep batch {timestep.numel()} does not match hidden batch {batch_size}.")
        gate_dtype = self.net[0].weight.dtype
        gate_input = timestep.view(int(batch_size), 1).to(dtype=gate_dtype)
        f_s = self.net(gate_input)
        if previous_entropy is None:
            # The first denoising step has no previous prediction: use H = 0.5.
            gate_logits = f_s
        else:
            if previous_entropy.dim() != 2 or previous_entropy.size(0) != int(batch_size):
                raise ValueError("previous_entropy must have shape [batch, seq] matching the hidden batch.")
            entropy = previous_entropy.detach().to(device=device, dtype=gate_dtype)
            g_s = self.entropy_net(gate_input)
            gate_logits = f_s + F.softplus(g_s) * (entropy - 0.5)
        return torch.sigmoid(gate_logits).to(dtype=dtype).unsqueeze(-1)


class FeedbackFlowStep(nn.Module):
    """One cross-attn/self-attn/FFN refinement step."""

    def __init__(self, config: FeedbackFlowConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if hidden_size % int(config.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        ff_size = max(hidden_size, int(hidden_size * float(config.ff_multiplier)))
        self.cross_q_norm = nn.LayerNorm(hidden_size)
        self.cross_kv_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(float(config.dropout)),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        feedback: torch.Tensor,
        hidden_key_padding_mask: Optional[torch.Tensor] = None,
        feedback_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cross, _ = self.cross_attn(
            query=self.cross_q_norm(hidden),
            key=self.cross_kv_norm(feedback),
            value=self.cross_kv_norm(feedback),
            key_padding_mask=feedback_key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + cross

        self_out, _ = self.self_attn(
            query=self.self_norm(hidden),
            key=self.self_norm(hidden),
            value=self.self_norm(hidden),
            key_padding_mask=hidden_key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self_out
        return hidden + self.ff(self.ff_norm(hidden))


class FeedbackFlowBlock(nn.Module):
    """A shared-weight FlowBlock applied recurrently for several refinement steps."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        steps: int = 3,
        ff_multiplier: float = 4.0,
        dropout: float = 0.0,
        timestep_gate_hidden: int = 128,
    ):
        super().__init__()
        if int(steps) <= 0:
            raise ValueError("steps must be positive.")
        self.config = FeedbackFlowConfig(
            hidden_size=int(hidden_size),
            num_heads=int(num_heads),
            steps=int(steps),
            ff_multiplier=float(ff_multiplier),
            dropout=float(dropout),
            timestep_gate_hidden=int(timestep_gate_hidden),
        )
        self.step = FeedbackFlowStep(self.config)
        self.timestep_gate = TimestepResidualGate(self.config.timestep_gate_hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        feedback: torch.Tensor,
        hidden_attention_mask: Optional[torch.Tensor] = None,
        feedback_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feedback.size(1) == 0:
            return hidden

        hidden_key_padding_mask = _attention_to_padding_mask(hidden_attention_mask, hidden)
        feedback_key_padding_mask = _attention_to_padding_mask(feedback_attention_mask, feedback)
        for _ in range(self.config.steps):
            hidden = self.step(
                hidden,
                feedback,
                hidden_key_padding_mask=hidden_key_padding_mask,
                feedback_key_padding_mask=feedback_key_padding_mask,
            )
        return hidden


@dataclass
class FeedbackFlowState:
    layer_idx: int
    feedback_layer_idx: int
    flow_block: nn.Module
    scale: float = 1.0
    handles: list = field(default_factory=list)
    feedback_hidden: Optional[torch.Tensor] = None
    feedback_attention_mask: Optional[torch.Tensor] = None
    response_mask: Optional[torch.Tensor] = None
    timestep: Optional[torch.Tensor] = None
    previous_entropy: Optional[torch.Tensor] = None
    conditional_batch_size: Optional[int] = None
    enabled: bool = True
    last_raw_hidden: Optional[torch.Tensor] = None
    last_steered_hidden: Optional[torch.Tensor] = None
    last_gate: Optional[torch.Tensor] = None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_cache(self) -> None:
        self.feedback_hidden = None
        self.feedback_attention_mask = None
        self.response_mask = None
        self.timestep = None
        self.previous_entropy = None
        self.conditional_batch_size = None
        self.last_raw_hidden = None
        self.last_steered_hidden = None
        self.last_gate = None


class LayerHiddenCollector:
    """Collect sequence hidden states from one LLaDA transformer layer."""

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layers = transformer_layers(model)
        if not 0 <= int(layer_idx) < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} is out of range for {len(self.layers)} layers.")
        self.layer_idx = int(layer_idx)
        self.handle = None
        self.hidden: Optional[torch.Tensor] = None

    def __enter__(self):
        self.handle = self.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.hidden = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.hidden = hidden.detach()
        return output

    @torch.no_grad()
    def collect(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **forward_kwargs) -> torch.Tensor:
        self.hidden = None
        if forward_kwargs:
            if attention_mask is None:
                self.model(input_ids, **forward_kwargs)
            else:
                self.model(input_ids, attention_mask=attention_mask, **forward_kwargs)
        else:
            forward_model(self.model, input_ids, attention_mask)
        if self.hidden is None:
            raise RuntimeError(f"Failed to collect hidden states from layer {self.layer_idx}.")
        return self.hidden


class FeedbackFlowController:
    """Prepare previous-prediction feedback and keep the hook state in sync."""

    def __init__(self, model):
        state = getattr(model, "_feedback_flow_state", None)
        if state is None:
            raise RuntimeError("Feedback flow is not attached. Call attach_feedback_flow first.")
        self.model = model
        self.state: FeedbackFlowState = state

    @torch.no_grad()
    def encode_feedback(
        self,
        feedback_ids: torch.Tensor,
        feedback_attention_mask: Optional[torch.Tensor] = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        feedback_ids = feedback_ids.to(_model_device(self.model))
        if feedback_attention_mask is not None:
            feedback_attention_mask = feedback_attention_mask.to(feedback_ids.device)
        with feedback_flow_disabled(self.model):
            with LayerHiddenCollector(self.model, self.state.feedback_layer_idx) as collector:
                return collector.collect(feedback_ids, feedback_attention_mask, **forward_kwargs).detach()

    @torch.no_grad()
    def prepare_feedback(
        self,
        feedback_ids: torch.Tensor,
        feedback_attention_mask: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        timestep: torch.Tensor | int | float | None = None,
        conditional_batch_size: Optional[int] = None,
        previous_entropy: Optional[torch.Tensor] = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        feedback_hidden = self.encode_feedback(feedback_ids, feedback_attention_mask, **forward_kwargs)
        self.state.feedback_hidden = feedback_hidden
        self.state.feedback_attention_mask = None if feedback_attention_mask is None else feedback_attention_mask.detach().bool()
        self.state.response_mask = None if response_mask is None else response_mask.detach().bool()
        self.state.timestep = _detach_optional_tensor(timestep, _model_device(self.model))
        self.state.previous_entropy = _detach_optional_tensor(previous_entropy, _model_device(self.model))
        self.state.conditional_batch_size = conditional_batch_size
        self.state.last_raw_hidden = None
        self.state.last_steered_hidden = None
        self.state.last_gate = None
        return feedback_hidden

    @contextmanager
    def feedback_from_ids(
        self,
        feedback_ids: torch.Tensor,
        feedback_attention_mask: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        timestep: torch.Tensor | int | float | None = None,
        conditional_batch_size: Optional[int] = None,
        previous_entropy: Optional[torch.Tensor] = None,
        **forward_kwargs,
    ) -> Iterator[FeedbackFlowState]:
        previous = _snapshot_state(self.state)
        self.prepare_feedback(
            feedback_ids,
            feedback_attention_mask,
            response_mask,
            timestep,
            conditional_batch_size,
            previous_entropy=previous_entropy,
            **forward_kwargs,
        )
        try:
            yield self.state
        finally:
            _restore_state(self.state, previous)

    def clear_feedback(self) -> None:
        self.state.clear_cache()

    def remove(self) -> None:
        clear_feedback_flow(self.model)


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _model_dtype(model) -> torch.dtype:
    parameter = next(model.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


def _detach_optional_tensor(value, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device).detach()
    return torch.as_tensor(value, device=device).detach()


def _attention_to_padding_mask(attention_mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}.")
    if attention_mask.shape != reference.shape[:2]:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match hidden shape {tuple(reference.shape[:2])}."
        )
    return ~attention_mask.to(device=reference.device, dtype=torch.bool)


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.size(0) == batch_size:
        return tensor
    if tensor.size(0) == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(f"{name} batch {tensor.size(0)} does not match expected batch {batch_size}.")


def _expand_response_mask(mask: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2:
        raise ValueError(f"response_mask must have shape [batch, seq], got {tuple(mask.shape)}.")
    if mask.size(1) != seq_len:
        raise ValueError(f"response_mask length {mask.size(1)} does not match hidden length {seq_len}.")
    return _expand_batch(mask.to(device=device, dtype=torch.bool), batch_size, "response_mask")


def _conditional_view(state: FeedbackFlowState, hidden: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    conditional_batch_size = state.conditional_batch_size
    if conditional_batch_size is None:
        return hidden, None
    conditional_batch_size = int(conditional_batch_size)
    if hidden.size(0) == conditional_batch_size:
        return hidden, None
    if hidden.size(0) == 2 * conditional_batch_size:
        return hidden[:conditional_batch_size], hidden[conditional_batch_size:]
    raise ValueError(
        f"Hidden batch {hidden.size(0)} is incompatible with conditional batch {conditional_batch_size}."
    )


def _merge_conditional(conditional: torch.Tensor, unconditional: Optional[torch.Tensor]) -> torch.Tensor:
    return conditional if unconditional is None else torch.cat([conditional, unconditional], dim=0)


def _snapshot_state(state: FeedbackFlowState) -> dict:
    return {
        "feedback_hidden": state.feedback_hidden,
        "feedback_attention_mask": state.feedback_attention_mask,
        "response_mask": state.response_mask,
        "timestep": state.timestep,
        "previous_entropy": state.previous_entropy,
        "conditional_batch_size": state.conditional_batch_size,
        "last_raw_hidden": state.last_raw_hidden,
        "last_steered_hidden": state.last_steered_hidden,
        "last_gate": state.last_gate,
        "enabled": state.enabled,
    }


def _restore_state(state: FeedbackFlowState, snapshot: dict) -> None:
    for key, value in snapshot.items():
        setattr(state, key, value)


def clear_feedback_flow(model) -> None:
    state = getattr(model, "_feedback_flow_state", None)
    if state is not None:
        state.remove()
        delattr(model, "_feedback_flow_state")



def _torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_feedback_flow_artifact(
    path: str | Path,
    state_or_flow_block,
    metadata: Optional[dict] = None,
    train_history: Optional[list[dict]] = None,
) -> None:
    flow_block = getattr(state_or_flow_block, "flow_block", state_or_flow_block)
    if not hasattr(flow_block, "config"):
        raise ValueError("Only FeedbackFlowBlock-like modules with a config can be saved.")
    payload = {
        "artifact_version": FEEDBACK_FLOW_ARTIFACT_VERSION,
        "kind": FEEDBACK_FLOW_ARTIFACT_KIND,
        "flow_config": dict(flow_block.config.__dict__),
        "flow_state_dict": {
            key: value.detach().cpu()
            for key, value in flow_block.state_dict().items()
        },
        "metadata": {} if metadata is None else dict(metadata),
        "train_history": [] if train_history is None else list(train_history),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_feedback_flow_artifact(path: str | Path, map_location="cpu") -> dict:
    payload = _torch_load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported Feedback Flow artifact type: {type(payload).__name__}.")
    version = int(payload.get("artifact_version", -1))
    if version not in SUPPORTED_FEEDBACK_FLOW_ARTIFACT_VERSIONS:
        raise ValueError(f"Unsupported Feedback Flow artifact version: {version}.")
    if payload.get("kind") != FEEDBACK_FLOW_ARTIFACT_KIND:
        raise ValueError(f"Unsupported Feedback Flow artifact kind: {payload.get('kind')!r}.")
    return payload


def load_feedback_flow_block(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> tuple[FeedbackFlowBlock, dict]:
    payload = load_feedback_flow_artifact(path, map_location=device)
    flow_block = FeedbackFlowBlock(**payload["flow_config"])
    legacy = int(payload["artifact_version"]) < FEEDBACK_FLOW_ARTIFACT_VERSION
    flow_block.load_state_dict(payload["flow_state_dict"], strict=not legacy)
    if legacy:
        # Old artifacts have no learned entropy branch; preserve their gate to float precision.
        nn.init.zeros_(flow_block.timestep_gate.entropy_net[-1].weight)
        nn.init.constant_(flow_block.timestep_gate.entropy_net[-1].bias, -20.0)
    flow_block.to(device=device)
    if dtype is not None:
        flow_block.to(dtype=dtype)
    return flow_block.eval(), payload


def attach_feedback_flow_from_artifact(
    model,
    path: str | Path,
    scale: Optional[float] = None,
    layer_idx: Optional[int] = None,
    feedback_layer_idx: Optional[int] = None,
) -> FeedbackFlowState:
    payload_device = _model_device(model)
    flow_block, payload = load_feedback_flow_block(path, device=payload_device, dtype=_model_dtype(model))
    metadata = dict(payload.get("metadata", {}))
    target_layer = int(metadata.get("layer_idx", len(transformer_layers(model)) // 2) if layer_idx is None else layer_idx)
    feedback_layer = int(metadata.get("feedback_layer_idx", target_layer) if feedback_layer_idx is None else feedback_layer_idx)
    flow_scale = float(metadata.get("scale", 1.0) if scale is None else scale)
    return attach_feedback_flow(
        model,
        flow_block=flow_block,
        layer_idx=target_layer,
        feedback_layer_idx=feedback_layer,
        scale=flow_scale,
    )


def attach_feedback_flow(
    model,
    flow_block: Optional[nn.Module] = None,
    layer_idx: Optional[int] = None,
    feedback_layer_idx: Optional[int] = None,
    num_heads: int = 8,
    steps: int = 3,
    ff_multiplier: float = 4.0,
    dropout: float = 0.0,
    timestep_gate_hidden: int = 128,
    scale: float = 1.0,
) -> FeedbackFlowState:
    clear_feedback_flow(model)
    layers = transformer_layers(model)
    if not layers:
        raise ValueError("Cannot attach feedback flow to a model with no transformer layers.")

    target_layer = len(layers) // 2 if layer_idx is None else int(layer_idx)
    feedback_layer = target_layer if feedback_layer_idx is None else int(feedback_layer_idx)
    if not 0 <= target_layer < len(layers):
        raise IndexError(f"layer_idx {target_layer} is out of range for {len(layers)} layers.")
    if not 0 <= feedback_layer < len(layers):
        raise IndexError(f"feedback_layer_idx {feedback_layer} is out of range for {len(layers)} layers.")

    hidden_size = int(model_metadata(model)["hidden_size"])
    if flow_block is None:
        flow_block = FeedbackFlowBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            steps=steps,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            timestep_gate_hidden=timestep_gate_hidden,
        )
    flow_block.to(device=_model_device(model), dtype=_model_dtype(model))

    state = FeedbackFlowState(
        layer_idx=target_layer,
        feedback_layer_idx=feedback_layer,
        flow_block=flow_block,
        scale=float(scale),
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        state.last_raw_hidden = hidden.detach()
        if (
            not state.enabled
            or state.feedback_hidden is None
            or (state.timestep is None and state.scale == 0.0)
        ):
            state.last_steered_hidden = hidden
            state.last_gate = None
            return output

        conditional_hidden, unconditional_hidden = _conditional_view(state, hidden)
        batch_size, seq_len = conditional_hidden.shape[:2]
        feedback = _expand_batch(
            state.feedback_hidden.to(device=hidden.device, dtype=hidden.dtype),
            batch_size,
            "feedback_hidden",
        )
        feedback_mask = None
        if state.feedback_attention_mask is not None:
            feedback_mask = _expand_batch(
                state.feedback_attention_mask.to(device=hidden.device, dtype=torch.bool),
                batch_size,
                "feedback_attention_mask",
            )
        response_mask = _expand_response_mask(state.response_mask, batch_size, seq_len, hidden.device)

        flowed = state.flow_block(
            conditional_hidden,
            feedback,
            hidden_attention_mask=None,
            feedback_attention_mask=feedback_mask,
        )
        delta = (flowed - conditional_hidden) * response_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        if state.timestep is None or not hasattr(state.flow_block, "timestep_gate"):
            gate = hidden.new_full((batch_size, 1, 1), float(state.scale))
        elif state.previous_entropy is not None:
            entropy = state.previous_entropy
            if entropy.dim() != 2 or entropy.size(1) != seq_len:
                raise ValueError("previous_entropy must have shape [batch, seq] matching the hidden sequence.")
            entropy = _expand_batch(entropy, batch_size, "previous_entropy")
            gate = state.flow_block.timestep_gate(
                state.timestep, batch_size, hidden.device, hidden.dtype, previous_entropy=entropy,
            )
        else:
            gate = state.flow_block.timestep_gate(state.timestep, batch_size, hidden.device, hidden.dtype)
        if state.timestep is not None:
            gate = gate * float(state.scale)
        steered_conditional = conditional_hidden + gate * delta
        steered = _merge_conditional(steered_conditional, unconditional_hidden)
        state.last_steered_hidden = steered
        state.last_gate = gate.detach()
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    state.handles.append(layers[target_layer].register_forward_hook(hook))
    model._feedback_flow_state = state
    return state


@contextmanager
def feedback_flow_disabled(model) -> Iterator[None]:
    state = getattr(model, "_feedback_flow_state", None)
    if state is None:
        yield
        return
    previous = state.enabled
    state.enabled = False
    try:
        yield
    finally:
        state.enabled = previous


@contextmanager
def feedback_flow_context(
    model,
    feedback_hidden: Optional[torch.Tensor] = None,
    feedback_attention_mask: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    timestep: torch.Tensor | int | float | None = None,
    conditional_batch_size: Optional[int] = None,
    previous_entropy: Optional[torch.Tensor] = None,
) -> Iterator[FeedbackFlowState | None]:
    state = getattr(model, "_feedback_flow_state", None)
    if state is None:
        yield None
        return

    previous = _snapshot_state(state)
    state.feedback_hidden = feedback_hidden
    state.feedback_attention_mask = None if feedback_attention_mask is None else feedback_attention_mask.detach().bool()
    state.response_mask = None if response_mask is None else response_mask.detach().bool()
    state.timestep = _detach_optional_tensor(timestep, _model_device(model))
    state.previous_entropy = _detach_optional_tensor(previous_entropy, _model_device(model))
    state.conditional_batch_size = conditional_batch_size
    state.last_raw_hidden = None
    state.last_steered_hidden = None
    state.last_gate = None
    try:
        yield state
    finally:
        _restore_state(state, previous)


def forward_with_feedback(
    model,
    controller: FeedbackFlowController,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    feedback_ids: torch.Tensor,
    feedback_attention_mask: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    timestep: torch.Tensor | int | float | None = None,
    previous_entropy: Optional[torch.Tensor] = None,
    **forward_kwargs,
):
    with controller.feedback_from_ids(
        feedback_ids,
        feedback_attention_mask=feedback_attention_mask,
        response_mask=response_mask,
        timestep=timestep,
        conditional_batch_size=input_ids.size(0),
        previous_entropy=previous_entropy,
    ):
        if attention_mask is None:
            return model(input_ids, **forward_kwargs)
        return model(input_ids, attention_mask=attention_mask, **forward_kwargs)
