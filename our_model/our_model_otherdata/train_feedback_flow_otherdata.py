from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model.backbone import load_backbone, prediction_mask, resolve_backbone_args, token_logits  # noqa: E402
from model.steering import model_metadata, transformer_layers  # noqa: E402
from model.steering_otherdata.common import (  # noqa: E402
    DATASET_CHOICES,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    answer_text,
    build_split_indices,
    default_data_path,
    format_chat_prompt,
    load_frame,
    save_split_manifest,
    split_indices_from_manifest,
    split_summary,
    task_prompt,
)
from our_model.feedback_flow import (  # noqa: E402
    FeedbackFlowController,
    attach_feedback_flow,
    feedback_flow_disabled,
    normalized_token_entropy,
    save_feedback_flow_artifact,
)
from our_model.generate_feedback import previous_prediction_feedback_ids  # noqa: E402
from our_model.losses import FeedbackFlowLossWeights, feedback_flow_loss  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_DENOISING_STEPS = 128


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_mask: torch.Tensor
    response_mask: torch.Tensor
    timestep: int
    index: int


def set_seed(seed: int) -> None:
    # Configure cuBLAS before model loading initializes CUDA. A seed alone does
    # not make Flash SDPA backward or other CUDA kernels deterministic.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def load_base_model(args):
    model, tokenizer = load_backbone(
        args.model_dir, backbone=args.backbone, device=args.device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def choose_layer(requested: Optional[int], num_layers: int, name: str) -> int:
    layer = num_layers // 2 if requested is None else int(requested)
    if not 0 <= layer < num_layers:
        raise IndexError(f"{name} {layer} is out of range for {num_layers} layers.")
    return layer


def non_special_mask(tokenizer, ids: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(ids, dtype=torch.bool)
    for token_id in tuple(getattr(tokenizer, "all_special_ids", ()) or ()):
        mask &= ids.ne(int(token_id))
    return mask


def sample_target_mask(candidate: torch.Tensor, mask_prob: float, generator: torch.Generator) -> torch.Tensor:
    if not candidate.any():
        raise ValueError("answer has no trainable tokens")
    if mask_prob >= 1.0:
        return candidate.clone()
    probs = torch.rand(candidate.shape, generator=generator)
    selected = candidate & probs.lt(float(mask_prob))
    if selected.any():
        return selected
    choices = candidate.nonzero(as_tuple=False).flatten()
    picked = choices[torch.randint(len(choices), (1,), generator=generator).item()]
    selected[picked] = True
    return selected


def sample_timestep(args, generator: torch.Generator) -> int:
    steps = max(int(args.steps), 1)
    return int(torch.randint(0, steps, (1,), generator=generator).item())


def correct_answer(row: dict) -> str:
    choices = list(row["choices"])
    label = int(row["label"])
    if not 0 <= label < len(choices):
        raise ValueError(f"label {label} is out of range for {len(choices)} choices")
    return str(choices[label]).strip()


def encode_example(tokenizer, dataset: str, row: dict, index: int, args, generator: torch.Generator, mask_id: int) -> EncodedExample:
    prompt = task_prompt(dataset, row)
    prompt_ids = tokenizer(
        format_chat_prompt(tokenizer, prompt),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    answer_ids = tokenizer(
        answer_text(correct_answer(row)),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    if answer_ids.numel() == 0:
        raise ValueError("empty answer")

    full_ids = torch.cat([prompt_ids, answer_ids], dim=0).long()
    if args.max_seq_length is not None and full_ids.numel() > args.max_seq_length:
        raise ValueError(f"sequence length {full_ids.numel()} exceeds --max-seq-length")

    start = prompt_ids.numel()
    answer_candidate = non_special_mask(tokenizer, answer_ids)
    selected_answer = sample_target_mask(answer_candidate, args.mask_prob, generator)

    target_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    target_mask[start:] = selected_answer
    response_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    response_mask[start:] = answer_candidate

    input_ids = full_ids.clone()
    input_ids[target_mask] = int(mask_id)
    labels = torch.full_like(full_ids, -100)
    labels[target_mask] = full_ids[target_mask]
    attention_mask = torch.ones_like(full_ids, dtype=torch.long)
    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        target_mask=target_mask,
        response_mask=response_mask,
        timestep=sample_timestep(args, generator),
        index=int(index),
    )


def collect_examples(tokenizer, dataset: str, frame, indices: list[int], args, mask_id: int, desc: str) -> tuple[list[EncodedExample], list[dict]]:
    examples: list[EncodedExample] = []
    skipped: list[dict] = []
    generator = torch.Generator().manual_seed(int(args.seed))
    progress = tqdm(indices, desc=desc, unit="sample")
    for index in progress:
        try:
            row = frame.loc[index].to_dict()
            examples.append(encode_example(tokenizer, dataset, row, int(index), args, generator, mask_id))
        except Exception as exc:  # noqa: BLE001 - report bad rows without stopping the whole build.
            skipped.append({"index": int(index), "reason": str(exc)})
        progress.set_postfix(used=len(examples), skipped=len(skipped))
    if not examples:
        raise RuntimeError(f"No valid examples collected for {desc}.")
    return examples, skipped


def pad_1d(values: torch.Tensor, length: int, value: int) -> torch.Tensor:
    output = values.new_full((length,), value)
    output[: values.numel()] = values
    return output


def collate_examples(batch: list[EncodedExample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item.input_ids.numel() for item in batch)
    return {
        "input_ids": torch.stack([pad_1d(item.input_ids, max_len, pad_id) for item in batch]),
        "attention_mask": torch.stack([pad_1d(item.attention_mask, max_len, 0) for item in batch]),
        "labels": torch.stack([pad_1d(item.labels, max_len, -100) for item in batch]),
        "target_mask": torch.stack([pad_1d(item.target_mask, max_len, 0) for item in batch]).bool(),
        "response_mask": torch.stack([pad_1d(item.response_mask, max_len, 0) for item in batch]).bool(),
        "timestep": torch.tensor([item.timestep for item in batch], dtype=torch.float32),
        "index": torch.tensor([item.index for item in batch], dtype=torch.long),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def previous_prediction(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with feedback_flow_disabled(model):
        logits = token_logits(model, input_ids, attention_mask)
    return logits.argmax(dim=-1), normalized_token_entropy(logits)


def prepare_feedback(controller: FeedbackFlowController, model, batch: dict[str, torch.Tensor], mask_id: int) -> torch.Tensor:
    pred, entropy = previous_prediction(model, batch["input_ids"], batch["attention_mask"])
    feedback_ids = previous_prediction_feedback_ids(batch["input_ids"], pred, batch["response_mask"], mask_id)
    controller.prepare_feedback(
        feedback_ids,
        feedback_attention_mask=batch["attention_mask"],
        response_mask=prediction_mask(model, batch["response_mask"]),
        timestep=batch["timestep"],
        conditional_batch_size=batch["input_ids"].size(0),
        previous_entropy=prediction_mask(model, entropy),
    )
    return feedback_ids


def run_epoch(model, controller, loader, optimizer, args, train: bool) -> dict:
    state = controller.state
    state.flow_block.train(mode=train)
    totals: dict[str, float] = {}
    total_items = 0
    progress = tqdm(loader, desc="Training" if train else "Validating", unit="batch")
    for batch in progress:
        batch = move_batch(batch, model.device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        prepare_feedback(controller, model, batch, int(model.config.mask_token_id))
        with torch.set_grad_enabled(train):
            logits = token_logits(model, batch["input_ids"], batch["attention_mask"])
            loss, metrics = feedback_flow_loss(
                logits,
                batch["labels"],
                state=state,
                target_mask=batch["target_mask"],
                response_mask=prediction_mask(model, batch["response_mask"]),
                weights=FeedbackFlowLossWeights(stability=args.stability_weight),
            )
            if state.last_gate is not None:
                metrics["timestep_gate"] = float(state.last_gate.float().mean().detach().cpu())
        if train:
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(state.flow_block.parameters(), args.grad_clip)
            optimizer.step()
        count = batch["input_ids"].size(0)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * count
        total_items += count
        progress.set_postfix({key: f"{value / total_items:.4f}" for key, value in totals.items()})
    return {key: value / max(total_items, 1) for key, value in totals.items()}


def resolve_splits(frame, args) -> dict[str, list[int]]:
    if args.split_manifest_path and Path(args.split_manifest_path).exists():
        return split_indices_from_manifest(args.split_manifest_path, frame)
    return build_split_indices(
        frame,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def limited_indices(indices: list[int], max_samples: int | None, name: str) -> list[int]:
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"--max-{name}-samples must be positive when provided.")
        return list(indices)[:max_samples]
    return list(indices)


def metric_value(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("inf")
    return float(value)


def make_loader(examples: list[EncodedExample], args, pad_id: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_examples(batch, pad_id),
    )


def default_output_path(dataset: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{dataset}_feedback_flow.pt"


def output_dir_for_backbone(backbone: str) -> Path:
    return DEFAULT_OUTPUT_DIR / "dream" if backbone == "dream" else DEFAULT_OUTPUT_DIR


def default_test_output_path(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".test.json")


def build_metadata(model, tokenizer, args, splits: dict, train_examples, val_examples, skipped_train, skipped_val) -> dict:
    metadata = model_metadata(model)
    metadata.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=args.dataset,
        tokenizer_id=str(getattr(tokenizer, "name_or_path", args.model_dir)),
        dataset_path=str(Path(args.data_path).resolve()),
        layer_idx=int(model._feedback_flow_state.layer_idx),
        feedback_layer_idx=int(model._feedback_flow_state.feedback_layer_idx),
        scale=float(args.flow_scale),
        flow_steps=int(args.flow_steps),
        timestep_gate_hidden=int(args.timestep_gate_hidden),
        denoising_steps=int(args.steps),
        num_heads=int(args.num_heads),
        mask_prob=float(args.mask_prob),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=1.0 - float(args.train_ratio) - float(args.val_ratio),
        split_seed=int(args.seed),
        num_samples=int(args.num_samples),
        split_counts=split_summary(splits),
        split_indices={
            split: [int(index) for index in splits[split]]
            for split in ("train", "val", "test")
        },
        used_train_indices=[int(item.index) for item in train_examples],
        used_val_indices=[int(item.index) for item in val_examples],
        skipped_train=skipped_train,
        skipped_val=skipped_val,
        prompt_template=(
            "evaluate_COPA dataset formatter followed by tokenizer chat template "
            "with add_generation_prompt=True"
        ),
        answer_source="correct_choice",
        answer_tokenization="leading space plus stripped choice text",
        feedback_source="raw previous argmax prediction at masked answer positions",
        selection_metric="val_loss",
        best_epoch=getattr(args, "best_epoch", None),
        best_val_loss=getattr(args, "best_val_loss", None),
        train_args=vars(args),
    )
    return metadata


def run_heldout_test(args, output_path: Path, manifest_path: Path) -> dict:
    from our_model.our_model_otherdata import eval_feedback_flow_otherdata as test_eval

    test_output = Path(args.test_output_path) if args.test_output_path else default_test_output_path(output_path)
    test_args = argparse.Namespace(
        dataset=args.dataset,
        backbone=args.backbone,
        model_dir=args.model_dir,
        data_path=args.data_path,
        flow_path=str(output_path),
        split_manifest_path=str(manifest_path),
        output_path=str(test_output),
        device=args.device,
        num_samples=args.num_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split=args.test_split,
        max_samples=args.test_max_samples,
        normalize_by_length=args.test_normalize_by_length,
        flow_scale=args.test_flow_scale,
        mc_timestep=args.test_mc_timestep,
    )
    return test_eval.evaluate(test_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Train previous-prediction Feedback Flow on MC otherdata tasks.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--backbone", choices=["llada", "dream", "auto"], default="llada")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--data-path")
    parser.add_argument("--output-path")
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_DENOISING_STEPS)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--feedback-layer-idx", type=int)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=3)
    parser.add_argument("--timestep-gate-hidden", type=int, default=128)
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--ff-multiplier", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stability-weight", type=float, default=0.01)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--test-output-path")
    parser.add_argument("--test-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--test-max-samples", type=int)
    parser.add_argument("--test-normalize-by-length", action="store_true")
    parser.add_argument("--test-flow-scale", type=float)
    parser.add_argument("--test-mc-timestep", type=float, default=0.0)
    return resolve_backbone_args(parser.parse_args())


def validate_args(args) -> None:
    if not 0.0 < args.mask_prob <= 1.0:
        raise ValueError("--mask-prob must be in (0, 1].")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")
    if not 0.0 < args.val_ratio < 1.0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--val-ratio must be positive and train+val must be below 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.data_path = args.data_path or default_data_path(args.dataset)
    args.output_path = args.output_path or str(
        output_dir_for_backbone(args.backbone) / f"{args.dataset}_feedback_flow.pt"
    )

    frame = load_frame(args.dataset, args.data_path, args.num_samples)
    splits = resolve_splits(frame, args)
    train_ids = limited_indices(splits["train"], args.max_train_samples, "train")
    val_ids = limited_indices(splits["val"], args.max_val_samples, "val")
    if not train_ids:
        raise ValueError("The training split is empty.")

    model, tokenizer = load_base_model(args)
    layers = transformer_layers(model)
    layer_idx = choose_layer(args.layer_idx, len(layers), "layer_idx")
    feedback_layer_idx = choose_layer(args.feedback_layer_idx, len(layers), "feedback_layer_idx")
    state = attach_feedback_flow(
        model,
        layer_idx=layer_idx,
        feedback_layer_idx=feedback_layer_idx,
        num_heads=args.num_heads,
        steps=args.flow_steps,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        timestep_gate_hidden=args.timestep_gate_hidden,
        scale=args.flow_scale,
    )
    controller = FeedbackFlowController(model)
    for parameter in state.flow_block.parameters():
        parameter.requires_grad_(True)

    mask_id = int(model.config.mask_token_id)
    train_examples, skipped_train = collect_examples(tokenizer, args.dataset, frame, train_ids, args, mask_id, "Building train examples")
    val_examples, skipped_val = (
        collect_examples(tokenizer, args.dataset, frame, val_ids, args, mask_id, "Building val examples")
        if val_ids
        else ([], [])
    )
    train_loader = make_loader(train_examples, args, mask_id, shuffle=True)
    val_loader = make_loader(val_examples, args, mask_id, shuffle=False) if val_examples else None
    optimizer = torch.optim.AdamW(state.flow_block.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_epoch = None
    best_val_loss = float("inf")
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        record = {"epoch": epoch, "train": run_epoch(model, controller, train_loader, optimizer, args, train=True)}
        if val_loader is not None:
            with torch.no_grad():
                record["val"] = run_epoch(model, controller, val_loader, optimizer, args, train=False)
            val_loss = metric_value(record["val"], "loss")
            record["is_best"] = val_loss < best_val_loss
            if record["is_best"]:
                best_epoch = epoch
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in state.flow_block.state_dict().items()
                })
        else:
            best_epoch = epoch
            best_val_loss = metric_value(record["train"], "loss")
            best_state_dict = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in state.flow_block.state_dict().items()
            })
            record["is_best"] = True
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    if best_state_dict is not None:
        state.flow_block.load_state_dict(best_state_dict)
    args.best_epoch = best_epoch
    args.best_val_loss = None if best_val_loss == float("inf") else best_val_loss

    output_path = Path(args.output_path)
    manifest_path = Path(args.split_manifest_path) if args.split_manifest_path else output_path.with_suffix(".splits.json")
    metadata = build_metadata(model, tokenizer, args, splits, train_examples, val_examples, skipped_train, skipped_val)
    save_feedback_flow_artifact(output_path, state, metadata=metadata, train_history=history)
    save_split_manifest(
        manifest_path,
        dataset=args.dataset,
        data_path=args.data_path,
        num_samples=args.num_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_indices=splits,
        used_train_indices=metadata["used_train_indices"],
    )

    summary = {
        "artifact": str(output_path.resolve()),
        "split_manifest": str(manifest_path.resolve()),
        "dataset": args.dataset,
        "backbone": args.backbone,
        "split_counts": split_summary(splits),
        "best_epoch": best_epoch,
        "best_val_loss": args.best_val_loss,
        "final": history[-1],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.skip_test:
        del model, tokenizer, controller, state, train_loader, val_loader, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"status": "starting_test", "dataset": args.dataset, "split": args.test_split}, ensure_ascii=False))
        test_report = run_heldout_test(args, output_path, manifest_path)
        print(json.dumps({
            "status": "test_complete",
            "output_path": str(Path(args.test_output_path).resolve() if args.test_output_path else default_test_output_path(output_path).resolve()),
            "summary": test_report["summary"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
