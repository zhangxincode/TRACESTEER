from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate_COPA import (  # noqa: E402
    default_paths,
    format_boolq_prompt,
    format_copa_prompt,
    format_mmlu_prompt,
    format_storycloze_prompt,
    format_wino_prompt,
    load_boolq,
    load_copa,
    load_mmlu,
    load_storycloze,
    load_wino,
)
from model.backbone import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    load_backbone,
    resolve_backbone,
    token_logits,
)
from model.steering import (  # noqa: E402
    ARTIFACT_VERSION,
    clear_steering,
    configure_steering,
    model_metadata,
    steering_mask,
    transformer_layers,
)
from model.truthfulqa_steering import (  # noqa: E402
    ResidualCollector,
    encode_question_answer,
    format_prompt,
    masked_answer_logprob as truthfulqa_masked_answer_logprob,
    stratified_question_split,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_NUM_SAMPLES = 817
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SCALES = [8.0, 16.0, 32.0, 64.0, 128.0]
DATASET_CHOICES = ("copa", "storycloze", "mmlu", "wino", "boolq")


@dataclass(frozen=True)
class DatasetSpec:
    loader: Callable[[str], list[dict]]
    formatter: Callable[[dict], str]


DATASET_SPECS = {
    "copa": DatasetSpec(load_copa, format_copa_prompt),
    "storycloze": DatasetSpec(load_storycloze, format_storycloze_prompt),
    "mmlu": DatasetSpec(load_mmlu, format_mmlu_prompt),
    "wino": DatasetSpec(load_wino, format_wino_prompt),
    "boolq": DatasetSpec(load_boolq, format_boolq_prompt),
}


def default_data_path(dataset: str) -> str:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data_path, _output_path = default_paths(dataset)
    return data_path


def default_output_dir(
    output_dir: str | Path | None = None,
    backbone: str | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    _, selected = resolve_backbone(backbone=backbone)
    return DEFAULT_OUTPUT_DIR / "dream" if selected == "dream" else DEFAULT_OUTPUT_DIR


def default_artifact_path(
    dataset: str,
    output_dir: str | Path | None = None,
    *,
    backbone: str | None = None,
) -> Path:
    return default_output_dir(output_dir, backbone) / f"{dataset}_steering.pt"


def default_manifest_path(
    dataset: str,
    output_dir: str | Path | None = None,
    *,
    backbone: str | None = None,
) -> Path:
    return default_output_dir(output_dir, backbone) / f"{dataset}_steering.splits.json"


def default_sweep_path(
    dataset: str,
    output_dir: str | Path | None = None,
    *,
    backbone: str | None = None,
) -> Path:
    return default_output_dir(output_dir, backbone) / f"{dataset}_steering_sweep.json"


def default_eval_path(
    dataset: str,
    method: str,
    output_dir: str | Path | None = None,
    *,
    backbone: str | None = None,
) -> Path:
    names = {
        "baseline": f"{dataset}_test_baseline.json",
        "5shot-icl": f"{dataset}_test_5shot_baseline.json",
        "caa": f"{dataset}_test_steered.json",
    }
    if method not in names:
        raise ValueError(f"Unsupported evaluation method: {method}")
    return default_output_dir(output_dir, backbone) / names[method]


def _normalize_row(row: dict, index: int, dataset: str) -> dict:
    if "choices" not in row or "label" not in row:
        raise ValueError(f"{dataset} sample {index} must contain choices and label.")

    choices = [str(choice).strip() for choice in row["choices"]]
    label = int(row["label"])
    if not choices or any(not choice for choice in choices):
        raise ValueError(f"{dataset} sample {index} contains an empty answer choice.")
    if not 0 <= label < len(choices):
        raise ValueError(
            f"{dataset} sample {index} has label {label} for {len(choices)} choices."
        )

    normalized = dict(row)
    normalized["id"] = str(normalized.get("id", index))
    normalized["choices"] = choices
    normalized["label"] = label
    normalized["category"] = "unknown"
    return normalized


def load_frame(dataset: str, data_path: str, num_samples: int = DEFAULT_NUM_SAMPLES) -> pd.DataFrame:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {DATASET_CHOICES}.")
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    rows = DATASET_SPECS[dataset].loader(data_path)
    if len(rows) < num_samples:
        raise ValueError(
            f"{dataset} contains only {len(rows)} samples, but --num-samples={num_samples} "
            "was requested."
        )

    selected = [_normalize_row(row, index, dataset) for index, row in enumerate(rows[:num_samples])]
    frame = pd.DataFrame(selected)
    frame.index = range(len(frame))
    return frame


def build_split_indices(
    frame: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> dict[str, list[int]]:
    split_indices = stratified_question_split(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    _validate_split_indices(split_indices, frame)
    return {name: [int(index) for index in values] for name, values in split_indices.items()}


def _validate_split_indices(split_indices: dict, frame: pd.DataFrame) -> None:
    expected = set(int(index) for index in frame.index)
    actual_values = []
    for split in ("train", "val", "test"):
        if split not in split_indices:
            raise ValueError(f"Split manifest is missing {split!r}.")
        values = [int(index) for index in split_indices[split]]
        actual_values.extend(values)
        missing = sorted(set(values) - expected)
        if missing:
            raise ValueError(
                f"Split {split!r} contains indices absent from the selected dataset: {missing[:5]}."
            )

    if len(actual_values) != len(expected) or len(set(actual_values)) != len(expected):
        raise ValueError("Train, validation, and test splits must be exhaustive and disjoint.")


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_indices_from_manifest(
    path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, list[int]]:
    manifest = load_manifest(path)
    indices = manifest.get("indices")
    if indices is None:
        raise ValueError(f"Split manifest {path} does not contain an 'indices' object.")
    normalized = {
        split: [int(index) for index in indices.get(split, [])]
        for split in ("train", "val", "test")
    }
    _validate_split_indices(normalized, frame)
    return normalized


def split_indices_from_artifact(
    artifact_path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, list[int]] | None:
    artifact = load_torch_payload(artifact_path)
    if not isinstance(artifact, dict):
        return None
    metadata = artifact.get("metadata", {})
    indices = metadata.get("split_indices")
    if indices is None:
        return None
    normalized = {
        split: [int(index) for index in indices.get(split, [])]
        for split in ("train", "val", "test")
    }
    _validate_split_indices(normalized, frame)
    return normalized


def resolve_split_indices(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_manifest_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, list[int]]:
    if split_manifest_path:
        return split_indices_from_manifest(split_manifest_path, frame)
    if artifact_path:
        artifact_splits = split_indices_from_artifact(artifact_path, frame)
        if artifact_splits is not None:
            return artifact_splits
    return build_split_indices(frame, seed, train_ratio, val_ratio)


def save_split_manifest(
    path: str | Path,
    *,
    dataset: str,
    data_path: str,
    num_samples: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_indices: dict[str, list[int]],
    used_train_indices: Iterable[int] | None = None,
) -> None:
    payload = {
        "dataset": dataset,
        "data_path": str(Path(data_path).resolve()),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(1.0 - train_ratio - val_ratio),
        },
        "indices": {
            split: [int(index) for index in split_indices[split]]
            for split in ("train", "val", "test")
        },
        "used_train_indices": [
            int(index) for index in (used_train_indices or split_indices["train"])
        ],
    }
    write_json(path, payload)


def split_summary(split_indices: dict[str, list[int]]) -> dict[str, int]:
    return {
        split: len(split_indices.get(split, []))
        for split in ("train", "val", "test")
    }


def select_split(
    frame: pd.DataFrame,
    split_indices: dict[str, list[int]],
    split: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported dataset split {split!r}.")
    selected = frame.loc[split_indices[split]]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided.")
        selected = selected.head(max_samples)
    return selected


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_dir: str | None = None,
    device: str | None = None,
    steering_args: SimpleNamespace | None = None,
    backbone: str | None = None,
):
    if model_dir is None:
        model_dir = getattr(steering_args, "model_dir", None)
    if backbone is None:
        backbone = getattr(steering_args, "backbone", None)
    if device is None:
        device = getattr(steering_args, "device", "auto")
    model, tokenizer = load_backbone(model_dir, backbone=backbone, device=device)
    if steering_args is not None:
        configure_steering(model, steering_args)
    return model, tokenizer


def task_prompt(dataset: str, row: dict) -> str:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset {dataset!r}.")
    return DATASET_SPECS[dataset].formatter(row)


def format_chat_prompt(
    tokenizer,
    prompt: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    if not few_shot_examples:
        return format_prompt(tokenizer, prompt)

    messages = []
    for example in few_shot_examples:
        example_prompt = str(example["prompt"]).strip()
        example_answer = str(example["answer"]).strip()
        if not example_prompt or not example_answer:
            continue
        messages.append({"role": "user", "content": example_prompt})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def answer_text(answer: str) -> str:
    answer = str(answer).strip()
    if not answer:
        raise ValueError("Answer choices must not be empty.")
    # The dataset prompts end in "Answer:", so match evaluate_COPA's word boundary.
    return " " + answer


def _answer_ids(tokenizer, answer: str, device: torch.device) -> torch.Tensor:
    return tokenizer(
        answer_text(answer),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)


@torch.no_grad()
def masked_choice_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    normalize_by_length: bool,
    few_shot_examples: list[dict] | None = None,
) -> float:
    prompt_ids = tokenizer(
        format_chat_prompt(tokenizer, prompt, few_shot_examples),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    answer_ids = _answer_ids(tokenizer, answer, model.device)
    if answer_ids.numel() == 0:
        return -math.inf

    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise AttributeError("The loaded model config does not define mask_token_id.")
    mask_ids = torch.full_like(answer_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    start = prompt_ids.size(1)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, start:] = True

    with steering_mask(
        model,
        response_mask,
        conditional_batch_size=input_ids.size(0),
    ):
        logits = token_logits(model, input_ids, attention_mask)
    token_logprobs = F.log_softmax(
        logits[:, start:start + answer_ids.size(1)],
        dim=-1,
    )
    score = token_logprobs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    if normalize_by_length:
        score /= answer_ids.size(1)
    return float(score)


def score_row_mc1(
    model,
    tokenizer,
    dataset: str,
    row: dict,
    normalize_by_length: bool,
    few_shot_examples: list[dict] | None = None,
) -> dict:
    prompt = task_prompt(dataset, row)
    option_scores = [
        masked_choice_logprob(
            model,
            tokenizer,
            prompt,
            choice,
            normalize_by_length,
            few_shot_examples,
        )
        for choice in row["choices"]
    ]
    predicted = max(range(len(option_scores)), key=option_scores.__getitem__)
    value = int(predicted == int(row["label"]))
    return {
        "id": str(row["id"]),
        "label": int(row["label"]),
        "predicted": int(predicted),
        "mc1": {
            "acc": value,
            "score": value,
        },
    }


def evaluate_mc1(
    model,
    tokenizer,
    dataset: str,
    frame: pd.DataFrame,
    normalize_by_length: bool,
    few_shot_examples: list[dict] | None = None,
    *,
    keep_results: bool = True,
    description: str | None = None,
) -> tuple[dict, list[dict]]:
    correct = 0
    results = []
    total = len(frame)
    progress = tqdm(
        frame.iterrows(),
        total=total,
        desc=description or f"Evaluating {dataset}",
        unit="sample",
    )
    for sample_no, (_index, row) in enumerate(progress, start=1):
        result = score_row_mc1(
            model,
            tokenizer,
            dataset,
            row.to_dict(),
            normalize_by_length,
            few_shot_examples,
        )
        correct += result["mc1"]["acc"]
        if keep_results:
            results.append(result)
        progress.set_postfix(
            sample=f"{sample_no}/{total}",
            mc1=f"{correct / sample_no:.4f}",
        )

    accuracy = correct / total if total else 0.0
    return {
        "num_samples": total,
        "mc1": {
            "acc": float(accuracy),
            "score": float(accuracy),
        },
    }, results


def build_few_shot_examples(
    dataset: str,
    frame: pd.DataFrame,
    split_indices: dict[str, list[int]],
    count: int,
    seed: int,
) -> list[dict]:
    if count < 0:
        raise ValueError("--few-shot-count cannot be negative.")
    if count == 0:
        return []

    indices = list(split_indices["train"])
    if len(indices) < count:
        raise ValueError(
            f"Need {count} few-shot examples, but train split contains only {len(indices)}."
        )
    random.Random(seed).shuffle(indices)

    examples = []
    for index in indices[:count]:
        row = frame.loc[index].to_dict()
        examples.append(
            {
                "index": int(index),
                "id": str(row["id"]),
                "prompt": task_prompt(dataset, row),
                "answer": str(row["choices"][int(row["label"])]).strip(),
            }
        )
    return examples


def _activation_for_answer(
    collector: ResidualCollector,
    tokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_question_answer(
        tokenizer,
        prompt,
        answer_text(answer),
        device,
    )
    return collector.collect(
        encoded["input_ids"],
        encoded["attention_mask"],
        encoded["answer_mask"],
    )


def build_steering_artifact(
    model,
    tokenizer,
    dataset: str,
    frame: pd.DataFrame,
    split_indices: dict[str, list[int]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    data_path: str,
    model_dir: str,
    negative_selection: str = "hardest",
    max_train_samples: int | None = None,
    eps: float = 1e-12,
) -> dict:
    if negative_selection not in ("hardest", "mean"):
        raise ValueError(f"Unsupported negative selection: {negative_selection!r}.")

    train_indices = list(split_indices["train"])
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive when provided.")
        train_indices = train_indices[:max_train_samples]
    if not train_indices:
        raise ValueError("The training split is empty.")

    sum_difference = None
    used_indices = []
    progress = tqdm(train_indices, desc="Building steering vectors", unit="question")
    for index in progress:
        row = frame.loc[index].to_dict()
        choices = row["choices"]
        label = int(row["label"])
        positive = choices[label]
        negatives = [
            choice
            for choice_index, choice in enumerate(choices)
            if choice_index != label and str(choice).strip()
        ]
        if not negatives:
            raise ValueError(
                f"{dataset} sample {index} has no incorrect choice for CAA extraction."
            )

        prompt = task_prompt(dataset, row)
        with ResidualCollector(model) as collector:
            positive_activation = _activation_for_answer(
                collector,
                tokenizer,
                prompt,
                positive,
                model.device,
            )
            negative_activations = [
                _activation_for_answer(
                    collector,
                    tokenizer,
                    prompt,
                    negative,
                    model.device,
                )
                for negative in negatives
            ]

        negative_stack = torch.stack(negative_activations)
        if negative_selection == "mean":
            negative_activation = negative_stack.float().mean(dim=0)
        else:
            negative_scores = [
                truthfulqa_masked_answer_logprob(
                    model,
                    tokenizer,
                    prompt,
                    answer_text(negative),
                    model.device,
                )
                for negative in negatives
            ]
            hardest_index = max(
                range(len(negative_scores)),
                key=negative_scores.__getitem__,
            )
            negative_activation = negative_stack[hardest_index]

        difference = positive_activation - negative_activation
        sum_difference = (
            difference
            if sum_difference is None
            else sum_difference + difference
        )
        used_indices.append(int(index))

    if sum_difference is None or not used_indices:
        raise RuntimeError("No valid contrastive examples were found.")

    raw_vectors = sum_difference.float() / len(used_indices)
    norms = torch.linalg.vector_norm(raw_vectors, dim=-1)
    normalized_vectors = raw_vectors / norms.clamp_min(eps).unsqueeze(-1)
    metadata = model_metadata(model)
    metadata.update(
        {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "tokenizer_id": str(getattr(tokenizer, "name_or_path", model_dir)),
            "dataset_path": str(Path(data_path).resolve()),
            "pooling": "answer_mean",
            "pooling_positions": "answer_prediction_positions",
            "positive_source": "correct_choice",
            "negative_selection": negative_selection,
            "prompt_template": (
                "evaluate_COPA dataset formatter followed by the tokenizer chat template "
                "with add_generation_prompt=True"
            ),
            "answer_tokenization": "leading space plus stripped choice text",
            "split_seed": int(seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "split_indices": {
                split: [int(index) for index in split_indices[split]]
                for split in ("train", "val", "test")
            },
            "used_train_indices": used_indices,
            "num_train_questions": len(used_indices),
        }
    )
    return {
        "artifact_version": ARTIFACT_VERSION,
        "raw_vectors": {
            int(index): vector.cpu()
            for index, vector in enumerate(raw_vectors)
        },
        "normalized_vectors": {
            int(index): vector.cpu()
            for index, vector in enumerate(normalized_vectors)
        },
        "vector_norms": {
            int(index): float(norm)
            for index, norm in enumerate(norms)
        },
        "metadata": metadata,
    }


def load_torch_payload(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_artifact(path: str | Path, artifact: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def configure_model_steering(
    model,
    *,
    steering_path: str | Path | None,
    layer: int,
    scale: float,
    control: str,
    seed: int,
) -> None:
    if steering_path is None:
        clear_steering(model)
        return
    configure_steering(
        model,
        SimpleNamespace(
            disable_steering=False,
            steering_path=str(steering_path),
            steering_layer=int(layer),
            steering_scale=float(scale),
            steering_control=control,
            seed=int(seed),
        ),
    )


def infer_method(
    *,
    steering_path: str | Path | None,
    disable_steering: bool,
    few_shot_count: int,
) -> str:
    if steering_path is not None and not disable_steering:
        return "caa"
    if few_shot_count > 0:
        return "5shot-icl"
    return "baseline"


def best_selection_from_sweep(path: str | Path) -> dict:
    report = load_manifest(path)
    selection = report.get("best_by_mc1") or report.get("best")
    if selection is None:
        selection = report.get("steering_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Sweep report {path} does not contain a best configuration.")
    if selection.get("layer") is None or selection.get("scale") is None:
        raise ValueError(f"Sweep report {path} has an incomplete best configuration.")
    return selection
    return selection
