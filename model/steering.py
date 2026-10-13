from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch

from model.backbone import backbone_name, prediction_mask


ARTIFACT_VERSION = 1


@dataclass
class SteeringState:
    layer_idx: int
    handles: list = field(default_factory=list)
    vector: Optional[torch.Tensor] = None
    scale: float = 1.0
    response_mask: Optional[torch.Tensor] = None
    conditional_batch_size: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def core_model(model):
    return model.model if hasattr(model, "model") else model


def transformer_layers(model) -> list:
    core = core_model(model)
    layers = getattr(core, "layers", None)
    if layers is not None:
        return list(layers)
    transformer = getattr(core, "transformer", None)
    if transformer is None:
        raise AttributeError("Model does not expose model.layers or a transformer container.")

    blocks = getattr(transformer, "blocks", None)
    if blocks is not None:
        return list(blocks)

    block_groups = getattr(transformer, "block_groups", None)
    if block_groups is not None:
        return [block for group in block_groups for block in group]

    raise AttributeError("Model does not expose transformer.blocks or transformer.block_groups.")


def model_metadata(model) -> dict:
    config = model.config
    layers = transformer_layers(model)
    return {
        "model_id": str(getattr(config, "_name_or_path", "") or ""),
        "model_type": str(getattr(config, "model_type", "") or ""),
        "backbone": backbone_name(model),
        "hidden_size": int(getattr(config, "d_model", getattr(config, "hidden_size", 0))),
        "num_layers": int(getattr(config, "n_layers", getattr(config, "num_hidden_layers", len(layers)))),
    }


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _artifact_value(mapping: dict, layer_idx: int):
    if layer_idx in mapping:
        return mapping[layer_idx]
    key = str(layer_idx)
    if key in mapping:
        return mapping[key]
    raise KeyError(f"Steering artifact does not contain layer {layer_idx}.")


def _model_name(value: str) -> str:
    return Path(value).name if value else ""


def _validate_artifact(model, metadata: dict, vector: torch.Tensor, layer_idx: int) -> None:
    current = model_metadata(model)
    hidden_size = metadata.get("hidden_size")
    num_layers = metadata.get("num_layers")
    model_id = str(metadata.get("model_id", "") or "")
    model_type = str(metadata.get("model_type", "") or "")
    artifact_backbone = metadata.get("backbone")
    if artifact_backbone and artifact_backbone != current["backbone"]:
        raise ValueError(
            f"Artifact backbone {artifact_backbone!r} does not match loaded backbone {current['backbone']!r}."
        )

    if hidden_size is not None and int(hidden_size) != current["hidden_size"]:
        raise ValueError(
            f"Artifact hidden size {hidden_size} does not match model hidden size {current['hidden_size']}."
        )
    if num_layers is not None and int(num_layers) != current["num_layers"]:
        raise ValueError(
            f"Artifact layer count {num_layers} does not match model layer count {current['num_layers']}."
        )
    if model_type and current["model_type"] and model_type != current["model_type"]:
        raise ValueError(
            f"Artifact model type {model_type!r} does not match loaded model type {current['model_type']!r}."
        )
    if model_id and current["model_id"] and _model_name(model_id) != _model_name(current["model_id"]):
        raise ValueError(
            f"Artifact model {model_id!r} does not match loaded model {current['model_id']!r}."
        )
    if vector.numel() != current["hidden_size"]:
        raise ValueError(
            f"Steering vector length {vector.numel()} does not match hidden size {current['hidden_size']}."
        )
    if not 0 <= layer_idx < current["num_layers"]:
        raise IndexError(f"Layer {layer_idx} is out of range for {current['num_layers']} layers.")


def _normalize_direction(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("A loaded steering vector must have a finite, non-zero norm.")
    return vector / norm


def load_steering_direction(model, path: str | Path, layer_idx: Optional[int]) -> tuple[torch.Tensor, int, dict]:
    payload = _torch_load(path)
    metadata = {}

    if torch.is_tensor(payload):
        layers = transformer_layers(model)
        selected_layer = len(layers) // 2 if layer_idx is None else layer_idx
        vector = payload
    elif isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}))
        version = payload.get("artifact_version", metadata.get("artifact_version"))
        if version is not None and int(version) != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported steering artifact version: {version}.")

        selected_layer = layer_idx
        if selected_layer is None:
            selected_layer = payload.get("selected_layer", metadata.get("selected_layer"))
        if selected_layer is None:
            selected_layer = len(transformer_layers(model)) // 2
        selected_layer = int(selected_layer)

        vectors = payload.get("normalized_vectors")
        if vectors is None:
            vectors = payload.get("raw_vectors")
        if vectors is not None:
            vector = _artifact_value(vectors, selected_layer)
        else:
            vector = None
            for key in ("steering_vector", "vector", "direction", "delta", "tensor"):
                if key in payload:
                    vector = payload[key]
                    break
            if vector is None:
                raise ValueError(f"Cannot find steering vectors in artifact {path}.")
    else:
        raise TypeError(f"Unsupported steering artifact type: {type(payload).__name__}.")

    vector = vector if torch.is_tensor(vector) else torch.as_tensor(vector)
    direction = _normalize_direction(vector)
    _validate_artifact(model, metadata, direction, selected_layer)
    return direction, selected_layer, metadata


def _masked_delta(state: SteeringState, hidden: torch.Tensor) -> Optional[torch.Tensor]:
    if state.vector is None or state.scale == 0.0 or state.response_mask is None:
        return None

    mask = state.response_mask
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2:
        raise ValueError(f"response_mask must have rank 1 or 2, got rank {mask.dim()}.")
    if mask.size(1) != hidden.size(1):
        raise ValueError(
            f"response_mask length {mask.size(1)} does not match hidden sequence length {hidden.size(1)}."
        )

    conditional_batch_size = state.conditional_batch_size
    if conditional_batch_size is not None:
        if mask.size(0) != conditional_batch_size:
            raise ValueError(
                f"response_mask batch {mask.size(0)} does not match conditional batch {conditional_batch_size}."
            )
        if hidden.size(0) == conditional_batch_size:
            expanded_mask = mask
        elif hidden.size(0) == 2 * conditional_batch_size:
            expanded_mask = torch.cat([mask, torch.zeros_like(mask)], dim=0)
        else:
            raise ValueError(
                f"Hidden batch {hidden.size(0)} is incompatible with conditional batch {conditional_batch_size}."
            )
    elif mask.size(0) == hidden.size(0):
        expanded_mask = mask
    elif mask.size(0) == 1:
        expanded_mask = mask.expand(hidden.size(0), -1)
    else:
        raise ValueError(
            f"response_mask batch {mask.size(0)} does not match hidden batch {hidden.size(0)}."
        )

    direction = state.vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
    mask_delta = expanded_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return mask_delta * direction * state.scale


def clear_steering(model) -> None:
    state = getattr(model, "_steering_state", None)
    if state is not None:
        state.remove()
        delattr(model, "_steering_state")


def attach_midlayer_steering(
    model,
    layer_idx: Optional[int] = None,
    vector: Optional[torch.Tensor] = None,
    vector_path: Optional[str] = None,
    scale: float = 1.0,
    control: str = "none",
    seed: int = 42,
) -> SteeringState:
    clear_steering(model)
    layers = transformer_layers(model)

    metadata = {}
    if vector is not None and vector_path:
        raise ValueError("Provide either vector or vector_path, not both.")
    if vector_path:
        vector, layer_idx, metadata = load_steering_direction(model, vector_path, layer_idx)
    else:
        layer_idx = len(layers) // 2 if layer_idx is None else layer_idx
        if vector is not None:
            vector = _normalize_direction(vector)
            _validate_artifact(model, {}, vector, layer_idx)

    if not 0 <= int(layer_idx) < len(layers):
        raise IndexError(f"layer_idx {layer_idx} is out of range for {len(layers)} layers.")
    layer_idx = int(layer_idx)

    if control not in {"none", "negative", "random"}:
        raise ValueError(f"Unsupported steering control: {control}.")
    if vector is not None and control == "negative":
        vector = -vector
    elif vector is not None and control == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vector = _normalize_direction(torch.randn(vector.shape, generator=generator))

    state = SteeringState(
        layer_idx=layer_idx,
        vector=vector,
        scale=float(scale),
        metadata=metadata,
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = _masked_delta(state, hidden)
        if delta is None:
            return output
        steered = hidden + delta
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    state.handles.append(layers[layer_idx].register_forward_hook(hook))
    model._steering_state = state
    return state


@contextmanager
def steering_mask(
    model,
    response_mask: torch.Tensor,
    conditional_batch_size: Optional[int] = None,
) -> Iterator[None]:
    state = getattr(model, "_steering_state", None)
    if state is None:
        yield
        return

    previous_mask = state.response_mask
    previous_conditional_batch_size = state.conditional_batch_size
    state.response_mask = prediction_mask(model, response_mask.detach().to(dtype=torch.bool))
    state.conditional_batch_size = conditional_batch_size
    try:
        yield
    finally:
        state.response_mask = previous_mask
        state.conditional_batch_size = previous_conditional_batch_size


def configure_steering(model, args=None):
    if args is not None and getattr(args, "disable_steering", False):
        clear_steering(model)
        return None

    vector_path = None
    if args is not None:
        vector_path = getattr(args, "steering_path", None)
        if vector_path is None:
            vector_path = getattr(args, "steering_vector_path", None)

    return attach_midlayer_steering(
        model,
        layer_idx=None if args is None else getattr(args, "steering_layer", None),
        vector_path=vector_path,
        scale=1.0 if args is None else getattr(args, "steering_scale", 1.0),
        control="none" if args is None else getattr(args, "steering_control", "none"),
        seed=42 if args is None else getattr(args, "seed", 42),
    )
