from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model import steering_DLLM_ as truthfulqa_eval  # noqa: E402
from model.backbone import (  # noqa: E402
    backbone_output_path,
    load_backbone,
    prediction_mask,
    resolve_backbone_args,
)
from utils.load_data import load_truthfulqa  # noqa: E402

try:  # noqa: E402
    from .feedback_flow import (  # noqa: E402
        FeedbackFlowController,
        attach_feedback_flow_from_artifact,
        feedback_flow_disabled,
        normalized_token_entropy,
    )
    from .generate_feedback import (  # noqa: E402
        feedback_model_logits,
        generate_with_feedback,
        previous_prediction_feedback_ids,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from feedback_flow import (  # type: ignore  # noqa: E402
        FeedbackFlowController,
        attach_feedback_flow_from_artifact,
        feedback_flow_disabled,
        normalized_token_entropy,
    )
    from generate_feedback import (  # type: ignore  # noqa: E402
        feedback_model_logits,
        generate_with_feedback,
        previous_prediction_feedback_ids,
    )


DEFAULT_FLOW_PATH = "./logs/feedback_flow.pt"
DEFAULT_OUTPUT_PATH = "./logs/feedback_flow_test.json"
MC_SINGLE_PASS_TIMESTEP = 0.0
_GPT_EVAL_CACHE_PRIMED = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir: str, backbone: str = "llada", device: str = "auto"):
    return load_backbone(model_dir, backbone, device)


def load_eval_frame(args):
    frame = load_truthfulqa(args.data_path)
    split_args = SimpleNamespace(
        truthfulqa_split=args.truthfulqa_split,
        split_manifest_path=args.split_manifest_path,
        steering_path=args.flow_path,
    )
    frame = truthfulqa_eval.select_truthfulqa_split(frame, split_args)
    return frame.head(args.max_samples) if args.max_samples is not None else frame


def format_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )


def answer_list(value) -> list[str]:
    return truthfulqa_eval.answer_list(value)


@torch.no_grad()
def previous_prediction(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with feedback_flow_disabled(model):
        logits = feedback_model_logits(model, input_ids, attention_mask)
    return logits.argmax(dim=-1), normalized_token_entropy(logits)


@torch.no_grad()
def masked_answer_logprob(model, tokenizer, question: str, answer: str, normalize_by_length: bool) -> float:
    prompt_ids = tokenizer(
        format_prompt(tokenizer, question),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    if answer_ids.numel() == 0:
        return -math.inf

    masks = torch.full_like(answer_ids, model.config.mask_token_id)
    input_ids = torch.cat([prompt_ids, masks], dim=1)
    attention_mask = torch.ones_like(input_ids)
    start = prompt_ids.size(1)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, start:] = True
    pred, entropy = previous_prediction(model, input_ids, attention_mask)
    feedback_ids = previous_prediction_feedback_ids(input_ids, pred, response_mask, int(model.config.mask_token_id))
    controller = FeedbackFlowController(model)
    controller.prepare_feedback(
        feedback_ids,
        feedback_attention_mask=attention_mask,
        response_mask=prediction_mask(model, response_mask),
        timestep=MC_SINGLE_PASS_TIMESTEP,
        conditional_batch_size=input_ids.size(0),
        previous_entropy=entropy,
    )
    logits = feedback_model_logits(model, input_ids, attention_mask)
    controller.clear_feedback()
    token_logprobs = F.log_softmax(logits[:, start:start + answer_ids.size(1)], dim=-1)
    score = token_logprobs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    return score / answer_ids.size(1) if normalize_by_length else score


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, args) -> str:
    prompt = format_prompt(tokenizer, question)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    output_ids, _trace = generate_with_feedback(
        model,
        tokenizer,
        inputs.input_ids,
        inputs.attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        dream_alg=getattr(args, "dream_alg", "origin"),
        trace_feedback=False,
    )
    answer_ids = output_ids[0, inputs.input_ids.size(1):]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def score_mc_metrics(model, tokenizer, row: dict, args) -> dict:
    selected = [name for name in args.metrics if name in truthfulqa_eval.MC_METRICS]
    if not selected:
        return {}

    question = row["question"]
    correct = answer_list(row["correct_answers"])
    incorrect = answer_list(row["incorrect_answers"])
    best_answer = row["best_answer"]
    best_incorrect = incorrect[0]

    incorrect_scores = [
        masked_answer_logprob(model, tokenizer, question, answer, args.mc_normalize_by_length)
        for answer in incorrect
    ]
    correct_scores = [
        masked_answer_logprob(model, tokenizer, question, answer, args.mc_normalize_by_length)
        for answer in correct
    ]
    best_score = masked_answer_logprob(model, tokenizer, question, best_answer, args.mc_normalize_by_length)
    best_false_score = masked_answer_logprob(model, tokenizer, question, best_incorrect, args.mc_normalize_by_length)

    scores = {}
    if "mc0" in selected:
        scores["mc0"] = truthfulqa_eval.mc0_metric([best_score, best_false_score], correct_index=0)
    if "mc1" in selected:
        value = truthfulqa_eval.mc1_metric(best_score, incorrect_scores)
        scores["mc1"] = {"score": value, "acc": value}
    if "mc2" in selected:
        scores["mc2"] = {"score": truthfulqa_eval.mc2_metric(correct_scores, incorrect_scores)}
    if "mc3" in selected:
        scores["mc3"] = {"score": truthfulqa_eval.mc3_metric(correct_scores, incorrect_scores)}
    return scores


def generation_diagnostics(prediction: str) -> dict:
    return {
        "is_refusal": truthfulqa_eval.is_refusal(prediction),
        "answer_length_words": len(prediction.split()),
    }


def summarize(results: list[dict], metric_names: list[str], args) -> dict:
    names = list(metric_names)
    if any("truthful_and_informative" in item["metrics"] for item in results):
        names.append("truthful_and_informative")
    summary = truthfulqa_eval.summarize(results, names, args.seed, args.bootstrap_samples)
    diagnostics = [item["diagnostics"] for item in results if "diagnostics" in item]
    if diagnostics:
        summary["diagnostics"] = {
            "refusal_rate": sum(item["is_refusal"] for item in diagnostics) / len(diagnostics),
            "mean_answer_length_words": sum(item["answer_length_words"] for item in diagnostics) / len(diagnostics),
        }
    return summary


@contextmanager
def cpu_only_gpt_eval(enabled: bool):
    global _GPT_EVAL_CACHE_PRIMED
    if not enabled:
        yield
        return

    original_is_available = torch.cuda.is_available
    cache_clear = getattr(truthfulqa_eval.load_eval_model, "cache_clear", None)
    try:
        torch.cuda.is_available = lambda: False
        if not _GPT_EVAL_CACHE_PRIMED and callable(cache_clear):
            cache_clear()
            _GPT_EVAL_CACHE_PRIMED = True
        yield
    finally:
        torch.cuda.is_available = original_is_available


def add_truthful_info_metric(metrics: dict) -> None:
    judge = metrics.get("gpt_judge")
    info = metrics.get("gpt_info")
    if isinstance(judge, dict) and "error" not in judge and isinstance(info, dict) and "error" not in info:
        metrics["truthful_and_informative"] = {
            "score": float(judge["score"]) * float(info["score"]),
            "acc": int(bool(judge["acc"]) and bool(info["acc"])),
        }


def save_report(path: str | Path, report: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_answers_report(args, state, results: list[dict]) -> dict:
    generations = []
    for item in results:
        if "prediction" not in item:
            continue
        generations.append({
            "index": item["index"],
            "question": item["question"],
            "prediction": item["prediction"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "diagnostics": item.get("diagnostics", {}),
        })
    return {
        "config": truthfulqa_eval.safe_config(args),
        "truthfulqa_split": args.truthfulqa_split,
        "num_generations": len(generations),
        "feedback_flow": {
            "layer_idx": int(state.layer_idx),
            "feedback_layer_idx": int(state.feedback_layer_idx),
            "scale": float(state.scale),
            "artifact": str(Path(args.flow_path).resolve()),
        },
        "generations": generations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Feedback Flow on a held-out TruthfulQA split.")
    parser.add_argument("--backbone", choices=["llada", "dream", "auto"], default="llada")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--data-path", default=truthfulqa_eval.DEFAULT_DATA_PATH)
    parser.add_argument("--flow-path", default=None)
    parser.add_argument("--split-manifest-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--answers-output-path", default=None)
    parser.add_argument("--truthfulqa-split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mc-normalize-by-length", action="store_true")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--dream-alg",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
        default="origin",
    )
    parser.add_argument("--flow-scale", type=float, default=None)
    parser.add_argument("--gpt-device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--gpt-judge-scores")
    parser.add_argument("--gpt-info-scores")
    parser.add_argument("--openai-api-url", default=truthfulqa_eval.OPENAI_API_URL)
    parser.add_argument("--openai-api-key", default=truthfulqa_eval.OPENAI_API_KEY)
    parser.add_argument("--gpt-eval-model", default=truthfulqa_eval.OPENAI_EVAL_MODEL)
    parser.add_argument("--gpt-timeout", type=int, default=100000)
    parser.add_argument("--openai-user-agent", default=truthfulqa_eval.OPENAI_USER_AGENT)
    parser.add_argument("--metrics", nargs="+", choices=["all"] + truthfulqa_eval.GPT_METRICS + truthfulqa_eval.MC_METRICS, default=["all"])
    args = resolve_backbone_args(parser.parse_args())
    if args.temperature is None:
        args.temperature = 0.3 if args.backbone == "dream" else 0.0
    args.flow_path = args.flow_path or str(backbone_output_path(DEFAULT_FLOW_PATH, args.backbone))
    args.output_path = args.output_path or str(backbone_output_path(DEFAULT_OUTPUT_PATH, args.backbone))
    return args


def evaluate(args) -> dict:
    resolve_backbone_args(args)
    set_seed(args.seed)
    args.metrics = truthfulqa_eval.normalize_metrics(args.metrics)
    frame = load_eval_frame(args)
    model, tokenizer = load_model(
        args.model_dir, args.backbone, getattr(args, "device", "auto")
    )
    state = attach_feedback_flow_from_artifact(model, args.flow_path, scale=args.flow_scale)
    need_gpt_metrics = any(name in truthfulqa_eval.GPT_METRICS for name in args.metrics)
    need_generation = need_gpt_metrics or bool(getattr(args, "answers_output_path", None))

    results = []
    progress = tqdm(frame.iterrows(), total=len(frame), desc=f"Evaluating {args.truthfulqa_split} split", unit="sample")
    for sample_no, (index, row) in enumerate(progress, start=1):
        progress.set_postfix(sample=f"{sample_no}/{len(frame)}", index=int(index))
        row = row.to_dict()
        row["index"] = int(index)
        prediction = generate_answer(model, tokenizer, row["question"], args) if need_generation else ""
        if need_gpt_metrics:
            with cpu_only_gpt_eval(args.gpt_device == "cpu"):
                metrics = truthfulqa_eval.score_gpt_metrics(prediction, row, args)
        else:
            metrics = {}
        metrics.update(score_mc_metrics(model, tokenizer, row, args))
        add_truthful_info_metric(metrics)
        result = {
            "index": int(index),
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
            "metrics": metrics,
            "feedback_flow": {
                "layer_idx": int(state.layer_idx),
                "feedback_layer_idx": int(state.feedback_layer_idx),
                "scale": float(state.scale),
                "artifact": str(Path(args.flow_path).resolve()),
            },
        }
        if need_generation:
            result["prediction"] = prediction
            result["diagnostics"] = generation_diagnostics(prediction)
        results.append(result)

    report = {
        "config": truthfulqa_eval.safe_config(args),
        "summary": summarize(results, args.metrics, args),
        "results": results,
    }
    save_report(args.output_path, report)
    answers_output_path = getattr(args, "answers_output_path", None)
    if answers_output_path:
        save_report(answers_output_path, build_answers_report(args, state, results))
    return report


if __name__ == "__main__":
    report = evaluate(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
