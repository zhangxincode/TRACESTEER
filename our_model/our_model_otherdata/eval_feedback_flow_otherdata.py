from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model.backbone import (  # noqa: E402
    load_backbone,
    prediction_mask,
    resolve_backbone_args,
    token_logits,
)
from model.steering import model_metadata  # noqa: E402
from model.steering_otherdata.common import (  # noqa: E402
    DATASET_CHOICES,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    answer_text,
    default_data_path,
    format_chat_prompt,
    load_frame,
    load_torch_payload,
    resolve_split_indices,
    select_split,
    split_summary,
    task_prompt,
)
from our_model.feedback_flow import (  # noqa: E402
    FeedbackFlowController,
    attach_feedback_flow_from_artifact,
    feedback_flow_disabled,
    normalized_token_entropy,
)
from our_model.generate_feedback import previous_prediction_feedback_ids  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "logs"


def set_seed(seed: int) -> None:
    # Match the training runtime and configure cuBLAS before CUDA is initialized.
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


def load_model(model_dir: str, device_name: str, backbone: str):
    return load_backbone(model_dir, backbone=backbone, device=device_name)


@torch.no_grad()
def previous_prediction(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with feedback_flow_disabled(model):
        logits = token_logits(model, input_ids, attention_mask)
    return logits.argmax(dim=-1), normalized_token_entropy(logits)


@torch.no_grad()
def masked_choice_logprob(
    model,
    tokenizer,
    controller: FeedbackFlowController,
    prompt: str,
    answer: str,
    normalize_by_length: bool,
    timestep: float,
) -> float:
    prompt_ids = tokenizer(
        format_chat_prompt(tokenizer, prompt),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    answer_ids = tokenizer(
        answer_text(answer),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    if answer_ids.numel() == 0:
        return -math.inf

    mask_id = int(model.config.mask_token_id)
    mask_ids = torch.full_like(answer_ids, mask_id)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    start = prompt_ids.size(1)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, start:] = True

    pred, entropy = previous_prediction(model, input_ids, attention_mask)
    feedback_ids = previous_prediction_feedback_ids(input_ids, pred, response_mask, mask_id)
    controller.prepare_feedback(
        feedback_ids,
        feedback_attention_mask=attention_mask,
        response_mask=prediction_mask(model, response_mask),
        timestep=torch.full((input_ids.size(0),), float(timestep), device=model.device),
        conditional_batch_size=input_ids.size(0),
        previous_entropy=prediction_mask(model, entropy),
    )
    logits = token_logits(model, input_ids, attention_mask)
    controller.clear_feedback()

    token_logprobs = F.log_softmax(logits[:, start:start + answer_ids.size(1)], dim=-1)
    score = token_logprobs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    if normalize_by_length:
        score /= answer_ids.size(1)
    return float(score)


def score_row_mc1(model, tokenizer, controller, dataset: str, row: dict, args) -> dict:
    prompt = task_prompt(dataset, row)
    option_scores = [
        masked_choice_logprob(
            model,
            tokenizer,
            controller,
            prompt,
            choice,
            args.normalize_by_length,
            args.mc_timestep,
        )
        for choice in row["choices"]
    ]
    predicted = max(range(len(option_scores)), key=option_scores.__getitem__)
    label = int(row["label"])
    value = int(predicted == label)
    return {
        "id": str(row["id"]),
        "label": label,
        "predicted": int(predicted),
        "choices": [str(choice) for choice in row["choices"]],
        "option_scores": [float(score) for score in option_scores],
        "mc1": {
            "acc": value,
            "score": value,
        },
    }


def save_report(path: str | Path, report: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_config(args) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def artifact_num_samples(flow_path: str | Path) -> int | None:
    payload = load_torch_payload(flow_path)
    if not isinstance(payload, dict):
        return None
    value = payload.get("metadata", {}).get("num_samples")
    return None if value is None else int(value)


def validate_artifact_backbone(model, flow_path: str | Path) -> None:
    payload = load_torch_payload(flow_path)
    if not isinstance(payload, dict):
        return
    metadata = payload.get("metadata", {})
    current = model_metadata(model)
    for key in ("backbone", "model_type"):
        saved = str(metadata.get(key, "") or "").lower()
        loaded = str(current.get(key, "") or "").lower()
        if saved and loaded and saved != loaded:
            raise ValueError(
                f"Feedback Flow artifact {key} {saved!r} does not match loaded model {loaded!r}."
            )
    saved_id = str(metadata.get("model_id", "") or "")
    loaded_id = str(current.get("model_id", "") or "")
    if saved_id and loaded_id and Path(saved_id).name != Path(loaded_id).name:
        raise ValueError(
            f"Feedback Flow artifact model {saved_id!r} does not match loaded model {loaded_id!r}."
        )


def evaluate(args) -> dict:
    set_seed(args.seed)
    args.data_path = args.data_path or default_data_path(args.dataset)
    output_dir = DEFAULT_OUTPUT_DIR / "dream" if args.backbone == "dream" else DEFAULT_OUTPUT_DIR
    args.output_path = args.output_path or str(
        output_dir / f"{args.dataset}_feedback_flow.{args.split}.json"
    )
    if args.num_samples is None:
        args.num_samples = artifact_num_samples(args.flow_path) or DEFAULT_NUM_SAMPLES

    frame = load_frame(args.dataset, args.data_path, args.num_samples)
    splits = resolve_split_indices(
        frame,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_manifest_path=args.split_manifest_path,
        artifact_path=args.flow_path,
    )
    eval_frame = select_split(frame, splits, args.split, args.max_samples)

    model, tokenizer = load_model(args.model_dir, args.device, args.backbone)
    validate_artifact_backbone(model, args.flow_path)
    state = attach_feedback_flow_from_artifact(model, args.flow_path, scale=args.flow_scale)
    controller = FeedbackFlowController(model)

    correct = 0
    results = []
    total = len(eval_frame)
    progress = tqdm(eval_frame.iterrows(), total=total, desc=f"Evaluating {args.dataset}:{args.split}", unit="sample")
    for sample_no, (index, row) in enumerate(progress, start=1):
        result = score_row_mc1(model, tokenizer, controller, args.dataset, row.to_dict(), args)
        result["index"] = int(index)
        correct += int(result["mc1"]["acc"])
        results.append(result)
        progress.set_postfix(sample=f"{sample_no}/{total}", mc1=f"{correct / sample_no:.4f}")

    accuracy = correct / total if total else 0.0
    report = {
        "config": safe_config(args),
        "split_summary": split_summary(splits),
        "summary": {
            "num_samples": int(total),
            "mc1": {
                "acc": float(accuracy),
                "score": float(accuracy),
            },
        },
        "feedback_flow": {
            "layer_idx": int(state.layer_idx),
            "feedback_layer_idx": int(state.feedback_layer_idx),
            "scale": float(state.scale),
            "artifact": str(Path(args.flow_path).resolve()),
        },
        "results": results,
    }
    save_report(args.output_path, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Feedback Flow on COPA, StoryCloze, MMLU, Wino, or BoolQ.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--backbone", choices=["llada", "dream", "auto"], default="llada")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--data-path")
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--flow-scale", type=float)
    parser.add_argument("--mc-timestep", type=float, default=0.0)
    return resolve_backbone_args(parser.parse_args())


if __name__ == "__main__":
    report = evaluate(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
