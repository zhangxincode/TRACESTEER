from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from model.backbone import (
    DEFAULT_MODEL_DIR,
    backbone_output_path,
    forward_model,
    load_backbone,
    prediction_mask,
    resolve_backbone_args,
    token_logits,
)
from model.steering import ARTIFACT_VERSION, model_metadata, transformer_layers
from utils.load_data import load_truthfulqa


DEFAULT_DATA_PATH = "./row_data/TruthfulQA_v1.csv"
DEFAULT_OUTPUT_PATH = "./logs/truthfulqa_steering.pt"
PROMPT_TEMPLATE = "tokenizer chat template with add_generation_prompt=True, followed by the answer text"


def deduplicate_answers(values) -> list[str]:
    seen = set()
    answers = []
    for value in values:
        answer = str(value).strip()
        key = answer.casefold()
        if answer and key not in seen:
            seen.add(key)
            answers.append(answer)
    return answers


def _apportion_counts(group_sizes: dict[str, int], target: int, ratio: float, capacities=None) -> dict[str, int]:
    if capacities is None:
        capacities = group_sizes
    counts = {
        category: min(int(size * ratio), capacities[category])
        for category, size in group_sizes.items()
    }
    remaining = target - sum(counts.values())
    order = sorted(
        group_sizes,
        key=lambda category: (-(group_sizes[category] * ratio % 1.0), category),
    )
    while remaining > 0:
        changed = False
        for category in order:
            if counts[category] < capacities[category]:
                counts[category] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise RuntimeError("Unable to apportion TruthfulQA split counts.")
    return counts


# def stratified_question_split(
#     frame: pd.DataFrame,
#     seed: int = 42,
#     train_ratio: float = 0.70,
#     val_ratio: float = 0.15,
# ) -> dict[str, list[int]]:
#     if not 0.0 < train_ratio < 1.0:
#         raise ValueError("train_ratio must be between 0 and 1.")
#     if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
#         raise ValueError("val_ratio must be positive and train_ratio + val_ratio must be below 1.")

#     groups = defaultdict(list)
#     categories = frame["category"] if "category" in frame else pd.Series("unknown", index=frame.index)
#     for index, category in categories.items():
#         groups[str(category)].append(int(index))

#     group_sizes = {category: len(indices) for category, indices in groups.items()}
#     train_target = round(len(frame) * train_ratio)
#     val_target = round(len(frame) * val_ratio)
#     train_counts = _apportion_counts(group_sizes, train_target, train_ratio)
#     remaining_capacities = {
#         category: group_sizes[category] - train_counts[category]
#         for category in group_sizes
#     }
#     val_counts = _apportion_counts(group_sizes, val_target, val_ratio, remaining_capacities)

#     split_indices = {"train": [], "val": [], "test": []}
#     for category in sorted(groups):
#         indices = sorted(groups[category])
#         random.Random(f"{seed}:{category}").shuffle(indices)
#         train_count = train_counts[category]
#         val_count = val_counts[category]
#         train_end = train_count
#         val_end = train_count + val_count
#         split_indices["train"].extend(indices[:train_end])
#         split_indices["val"].extend(indices[train_end:val_end])
#         split_indices["test"].extend(indices[val_end:])

#     for split in split_indices:
#         split_indices[split].sort()

#     all_indices = [index for values in split_indices.values() for index in values]
#     if len(all_indices) != len(frame) or len(set(all_indices)) != len(frame):
#         raise RuntimeError("TruthfulQA split is not exhaustive and disjoint.")
#     return split_indices
def stratified_question_split(
    frame: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_samples: int | None = None,
) -> dict[str, list[int]]:

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/val ratios.")

    # Use only the first max_samples rows.
    if max_samples is not None:
        frame = frame.head(max_samples)

    # Treat missing categories as unknown.
    categories = frame["category"] if "category" in frame else "unknown"

    groups = defaultdict(list)
    for idx, category in zip(frame.index, categories):
        groups[str(category)].append(int(idx))

    sizes = {k: len(v) for k, v in groups.items()}

    train_target = round(len(frame) * train_ratio)
    val_target = round(len(frame) * val_ratio)

    train_counts = _apportion_counts(sizes, train_target, train_ratio)

    remaining = {
        k: sizes[k] - train_counts[k]
        for k in sizes
    }
    val_counts = _apportion_counts(
        sizes, val_target, val_ratio, remaining
    )

    splits = {"train": [], "val": [], "test": []}

    for category, indices in groups.items():
        rng = random.Random(f"{seed}:{category}")
        rng.shuffle(indices)

        n_train = train_counts[category]
        n_val = val_counts[category]

        splits["train"] += indices[:n_train]
        splits["val"] += indices[n_train:n_train + n_val]
        splits["test"] += indices[n_train + n_val:]

    for split in splits.values():
        split.sort()

    return splits

def contrastive_direction(
    positive_question_means: torch.Tensor,
    negative_question_means: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if positive_question_means.shape != negative_question_means.shape:
        raise ValueError("Positive and negative question means must have identical shapes.")
    if positive_question_means.dim() != 3:
        raise ValueError("Expected question activations with shape [questions, layers, hidden].")

    raw_vectors = (positive_question_means.float() - negative_question_means.float()).mean(dim=0)
    norms = torch.linalg.vector_norm(raw_vectors, dim=-1)
    normalized_vectors = raw_vectors / norms.clamp_min(eps).unsqueeze(-1)
    return raw_vectors, normalized_vectors, norms


def format_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )


def encode_question_answer(tokenizer, question: str, answer: str, device: torch.device) -> dict:
    prompt_ids = tokenizer(
        format_prompt(tokenizer, question),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if answer_ids.numel() == 0:
        raise ValueError("Cannot extract an activation from an empty answer.")

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    answer_mask[:, prompt_ids.size(1):] = True
    if tokenizer.all_special_ids:
        special = torch.zeros_like(answer_ids, dtype=torch.bool)
        for token_id in tokenizer.all_special_ids:
            special |= answer_ids.eq(token_id)
        answer_mask[:, prompt_ids.size(1):] &= ~special
    if not answer_mask.any():
        raise ValueError("The answer span contains only special tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_mask": answer_mask,
        "answer_ids": answer_ids,
        "prompt_length": prompt_ids.size(1),
    }


class ResidualCollector:
    def __init__(self, model):
        self.model = model
        self.layers = transformer_layers(model)
        self.answer_mask = None
        self.activations = [None] * len(self.layers)
        self.handles = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mask = self.answer_mask
            if mask is None:
                raise RuntimeError("ResidualCollector requires an answer mask before forward.")
            pooled = hidden.masked_select(mask.unsqueeze(-1)).view(-1, hidden.size(-1)).float().mean(dim=0)
            self.activations[layer_idx] = pooled.detach().cpu()
            return output

        return hook

    @torch.no_grad()
    def collect(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        self.answer_mask = prediction_mask(self.model, answer_mask)
        self.activations = [None] * len(self.layers)
        forward_model(self.model, input_ids, attention_mask)
        if any(value is None for value in self.activations):
            missing = [index for index, value in enumerate(self.activations) if value is None]
            raise RuntimeError(f"Failed to collect residual states from layers: {missing}.")
        return torch.stack(self.activations)


@torch.no_grad()
def masked_answer_logprob(model, tokenizer, question: str, answer: str, device: torch.device) -> float:
    prompt_ids = tokenizer(
        format_prompt(tokenizer, question),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if answer_ids.numel() == 0:
        return float("-inf")

    mask_ids = torch.full_like(answer_ids, model.config.mask_token_id)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    start = prompt_ids.size(1)
    logits = token_logits(model, input_ids, attention_mask)
    token_logprobs = F.log_softmax(logits[:, start:start + answer_ids.size(1)], dim=-1)
    score = token_logprobs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    return float(score)


def mean_answer_activations(
    collector: ResidualCollector,
    tokenizer,
    question: str,
    answers: list[str],
    device: torch.device,
) -> torch.Tensor:
    activations = []
    for answer in answers:
        encoded = encode_question_answer(tokenizer, question, answer, device)
        activations.append(
            collector.collect(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["answer_mask"],
            )
        )
    return torch.stack(activations).mean(dim=0)


def build_artifact(model, tokenizer, frame: pd.DataFrame, split_indices: dict, args) -> dict:
    train_indices = split_indices["train"]
    if args.max_train_questions is not None:
        train_indices = train_indices[: args.max_train_questions]
    if not train_indices:
        raise ValueError("The training split is empty.")

    positive_source = getattr(args, "positive_source", "best_answer")
    negative_selection = getattr(args, "negative_selection", "hardest")
    sum_difference = None
    used_indices = []
    progress = tqdm(train_indices, desc="Building steering vectors", unit="question")
    for index in progress:
        row = frame.loc[index]
        if positive_source == "all_correct":
            positives = deduplicate_answers([row["best_answer"], *row["correct_answers"]])
        elif positive_source == "best_answer":
            positives = deduplicate_answers([row["best_answer"]]) or deduplicate_answers(row["correct_answers"])
        else:
            raise ValueError(f"Unsupported positive source: {positive_source}.")
        negatives = deduplicate_answers(row["incorrect_answers"])
        if not positives or not negatives:
            continue

        with ResidualCollector(model) as collector:
            positive_mean = mean_answer_activations(
                collector, tokenizer, row["question"], positives, model.device
            )
            negative_activations = []
            for answer in negatives:
                encoded = encode_question_answer(tokenizer, row["question"], answer, model.device)
                negative_activations.append(
                    collector.collect(
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        encoded["answer_mask"],
                    )
                )

        negative_stack = torch.stack(negative_activations)
        if negative_selection == "mean":
            negative_mean = negative_stack.float().mean(dim=0)
        elif negative_selection == "hardest":
            negative_scores = [
                masked_answer_logprob(model, tokenizer, row["question"], answer, model.device)
                for answer in negatives
            ]
            hardest_index = max(range(len(negative_scores)), key=negative_scores.__getitem__)
            negative_mean = negative_stack[hardest_index]
        else:
            raise ValueError(f"Unsupported negative selection: {negative_selection}.")

        difference = positive_mean - negative_mean
        sum_difference = difference if sum_difference is None else sum_difference + difference
        used_indices.append(int(index))

    if sum_difference is None or not used_indices:
        raise RuntimeError("No valid TruthfulQA contrastive examples were found.")

    raw_vectors = sum_difference.float() / len(used_indices)
    norms = torch.linalg.vector_norm(raw_vectors, dim=-1)
    normalized_vectors = raw_vectors / norms.clamp_min(args.eps).unsqueeze(-1)
    metadata = model_metadata(model)
    metadata.update(
        {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer_id": str(getattr(tokenizer, "name_or_path", args.model_dir)),
            "chat_template": str(getattr(tokenizer, "chat_template", "") or ""),
            "dataset_path": str(Path(args.data_path).resolve()),
            "pooling": "answer_mean",
            "pooling_positions": "answer_prediction_positions",
            "positive_source": positive_source,
            "negative_selection": negative_selection,
            "prompt_template": PROMPT_TEMPLATE,
            "split_seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "split_indices": split_indices,
            "used_train_indices": used_indices,
            "num_train_questions": len(used_indices),
        }
    )
    return {
        "artifact_version": ARTIFACT_VERSION,
        "raw_vectors": {index: vector.cpu() for index, vector in enumerate(raw_vectors)},
        "normalized_vectors": {index: vector.cpu() for index, vector in enumerate(normalized_vectors)},
        "vector_norms": {index: float(norm) for index, norm in enumerate(norms)},
        "metadata": metadata,
    }


def save_split_manifest(path: Path, artifact: dict) -> None:
    metadata = artifact["metadata"]
    manifest = {
        "seed": metadata["split_seed"],
        "ratios": {
            "train": metadata["train_ratio"],
            "val": metadata["val_ratio"],
            "test": metadata["test_ratio"],
        },
        "indices": metadata["split_indices"],
        "used_train_indices": metadata["used_train_indices"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build TruthfulQA contrastive activation steering vectors.")
    parser.add_argument("--backbone", choices=["llada", "dream", "auto"], default="llada")
    parser.add_argument("--model-dir", default=None, help="Checkpoint path; defaults to the selected backbone.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path")
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-questions", type=int)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--positive-source", choices=["best_answer", "all_correct"], default="best_answer")
    parser.add_argument("--negative-selection", choices=["hardest", "mean"], default="hardest")
    args = resolve_backbone_args(parser.parse_args())
    args.output_path = args.output_path or str(backbone_output_path(DEFAULT_OUTPUT_PATH, args.backbone))
    return args


def main():
    args = parse_args()
    frame = load_truthfulqa(args.data_path)
    split_indices = stratified_question_split(frame, args.seed, args.train_ratio, args.val_ratio)
    model, tokenizer = load_backbone(args.model_dir, args.backbone, args.device)

    artifact = build_artifact(model, tokenizer, frame, split_indices, args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)

    manifest_path = Path(args.split_manifest_path) if args.split_manifest_path else output_path.with_suffix(".splits.json")
    save_split_manifest(manifest_path, artifact)
    print(f"Saved steering artifact to {output_path}")
    print(f"Saved split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
