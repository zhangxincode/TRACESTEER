from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = "/home/zhangxin/projects/HM_DLLM/llada-Instruct"
COPA_DATA_PATH = "/home/zhangxin/projects/HM_DLLM/row_data/copa-all.xml"
COPA_OUTPUT_PATH = "/home/zhangxin/projects/HM_DLLM/code/logs/copa_mc1.json"
STORYCLOZE_DATA_PATH = "/home/zhangxin/projects/HM_DLLM/row_data/StoryCloze_test_spring2016.json"
STORYCLOZE_OUTPUT_PATH = "/home/zhangxin/projects/HM_DLLM/code/logs/storycloze_mc1.json"
MMLU_DATA_PATH = "/home/zhangxin/projects/HM_DLLM/row_data/mc_test_MMLU.csv"
MMLU_OUTPUT_PATH = "/home/zhangxin/projects/HM_DLLM/code/logs/mmlu_mc1.json"
WINO_DATA_PATH = "/home/zhangxin/projects/HM_DLLM/row_data/Wino.json"
WINO_OUTPUT_PATH = "/home/zhangxin/projects/HM_DLLM/code/logs/wino_mc1.json"
BOOLQ_DATA_PATH = "/home/zhangxin/projects/HM_DLLM/row_data/boolq_train.jsonl"
BOOLQ_OUTPUT_PATH = "/home/zhangxin/projects/HM_DLLM/code/logs/boolq_mc1.json"

DATASET_DEFAULTS = {
    "boolq": (BOOLQ_DATA_PATH, BOOLQ_OUTPUT_PATH),
    "copa": (COPA_DATA_PATH, COPA_OUTPUT_PATH),
    "storycloze": (STORYCLOZE_DATA_PATH, STORYCLOZE_OUTPUT_PATH),
    "mmlu": (MMLU_DATA_PATH, MMLU_OUTPUT_PATH),
    "wino": (WINO_DATA_PATH, WINO_OUTPUT_PATH),
}


def default_paths(dataset: str) -> tuple[str, str]:
    return DATASET_DEFAULTS[dataset]


def load_model(model_dir: str):
    """Load a local DLLM/LLaDA model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    return model, tokenizer


def load_copa(path: str) -> list[dict]:
    """Load COPA data from XML."""
    root = ET.parse(path).getroot()
    rows = []
    for item in root.findall("item"):
        rows.append(
            {
                "id": item.attrib["id"],
                "premise": item.findtext("p", "").strip(),
                "asks_for": item.attrib["asks-for"],
                "choices": [
                    item.findtext("a1", "").strip(),
                    item.findtext("a2", "").strip(),
                ],
                "label": int(item.attrib["most-plausible-alternative"]) - 1,
            }
        )
    return rows


def load_storycloze(path: str) -> list[dict]:
    """Load StoryCloze data from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for item in data:
        rows.append(
            {
                "id": item["InputStoryid"],
                "sentences": [
                    item[f"InputSentence{i}"].strip()
                    for i in range(1, 5)
                ],
                "choices": [
                    item["RandomFifthSentenceQuiz1"].strip(),
                    item["RandomFifthSentenceQuiz2"].strip(),
                ],
                "label": int(item["AnswerRightEnding"]) - 1,
            }
        )
    return rows


def load_mmlu(path: str) -> list[dict]:
    """Load MMLU data from CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"MMLU row {index} has fewer than 6 columns: {row}")
            question, choice1, choice2, choice3, choice4, answer = [item.strip() for item in row[:6]]
            label = answer.upper()
            if label not in "ABCD":
                if index == 0:
                    continue
                raise ValueError(f"Invalid MMLU answer at row {index}: {answer}")
            rows.append(
                {
                    "id": str(index),
                    "question": question,
                    "choices": [choice1, choice2, choice3, choice4],
                    "label": "ABCD".index(label),
                }
            )
    return rows


def load_wino(path: str) -> list[dict]:
    """Load Wino data from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(data):
        rows.append(
            {
                "id": str(index),
                "sentence": item["sentence"].strip(),
                "choices": [item["option1"].strip(), item["option2"].strip()],
                "label": int(item["answer"]) - 1,
            }
        )
    return rows


def load_boolq(path: str) -> list[dict]:
    """Load BoolQ data from JSONL."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "id": str(index),
                    "question": item["question"].strip(),
                    "passage": item["passage"].strip(),
                    "choices": ["Yes", "No"],
                    "label": 0 if item["answer"] else 1,
                }
            )
    return rows


def format_copa_prompt(row: dict) -> str:
    """Build a zero-shot two-choice COPA prompt."""
    return (
        "Carefully read the passage below and answer the question by choosing between A and B.\n\n"
        "Use the following rules:\n"
        "- Base your answer on the passage rather than external knowledge.\n"
        "- Consider both choices before deciding.\n"
        "- Prefer the choice that is directly supported or most reasonably implied by the passage.\n"
        "- Avoid assumptions that are not supported by the passage.\n"
        "- Pay special attention to negation, conditions, comparisons, causal relations, and temporal order.\n"
        "- If the passage does not explicitly state the answer, select the choice requiring the least additional assumption.\n\n"
        f"Passage:\n{row['premise']}\n\n"
        f"Question:\nBased on the passage above, what is the most reasonable {row['asks_for']}?\n\n"
        f"A: {row['choices'][0]}\n"
        f"B: {row['choices'][1]}\n\n"
        "Respond with exactly one character: A or B.\n\n"
        "Answer:"
    )


def format_storycloze_prompt(row: dict) -> str:
    """Build a zero-shot two-choice StoryCloze prompt."""
    story = "\n".join(
        f"{index}. {sentence}"
        for index, sentence in enumerate(row["sentences"], start=1)
    )
    return (
        "Read the four-sentence story carefully and choose the ending that provides "
        "the most natural, coherent, and plausible continuation.\n\n"
        "Consider:\n"
        "- consistency with the events and characters in the story;\n"
        "- causal and temporal coherence;\n"
        "- the characters' likely goals, actions, and reactions;\n"
        "- ordinary commonsense plausibility;\n"
        "- which ending requires fewer unsupported assumptions.\n\n"
        "Reject an ending if it contradicts the story, introduces an abrupt or unexplained "
        "development, or is less naturally connected to the preceding events.\n\n"
        f"Story:\n{story}\n\n"
        f"A: {row['choices'][0]}\n"
        f"B: {row['choices'][1]}\n\n"
        "Respond with exactly one letter: A or B.\n"
        "Answer:"
    )


def format_mmlu_prompt(row: dict) -> str:
    """Build a zero-shot four-choice MMLU prompt."""
    return (
        "Read the question carefully and choose the single best answer from A, B, C, and D.\n\n"
        "When answering:\n"
        "- Determine exactly what the question is asking.\n"
        "- Evaluate all four choices before making a decision.\n"
        "- Eliminate choices that are incorrect, irrelevant, contradictory, or less directly supported.\n"
        "- Pay attention to negation, qualifiers, exceptions, comparisons, and subtle wording differences.\n"
        "- If multiple choices seem plausible, choose the one that most directly and completely answers the question "
        "while requiring the fewest unsupported assumptions.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"A: {row['choices'][0]}\n"
        f"B: {row['choices'][1]}\n"
        f"C: {row['choices'][2]}\n"
        f"D: {row['choices'][3]}\n\n"
        "Respond with exactly one letter: A, B, C, or D.\n"
        "Answer:"
    )


def format_wino_prompt(row: dict) -> str:
    """Build a zero-shot two-choice Wino prompt."""
    return (
        "Read the sentence carefully and choose the option that best completes the blank.\n\n"
        "Consider:\n"
        "- the meaning of the entire sentence;\n"
        "- grammatical correctness and sentence structure;\n"
        "- semantic and contextual coherence;\n"
        "- natural word usage, collocations, and idiomatic expressions.\n\n"
        "Reject an option if it makes the sentence grammatically incorrect, awkward, "
        "contradictory, or less natural in context. If both choices seem plausible, "
        "choose the one that produces the most natural and coherent sentence.\n\n"
        f"Sentence:\n{row['sentence']}\n\n"
        f"A: {row['choices'][0]}\n"
        f"B: {row['choices'][1]}\n\n"
        "Respond with exactly one letter: A or B.\n"
        "Answer:"
    )


def format_boolq_prompt(row: dict) -> str:
    """Build a zero-shot two-choice BoolQ prompt."""
    return (
        "Read the passage carefully and answer the question with Yes or No.\n\n"
        "Use only the information stated or clearly implied by the passage. "
        "Do not introduce unsupported assumptions or outside knowledge. "
        "Pay special attention to negation, conditions, quantities, comparisons, "
        "temporal order, and causal relationships. "
        "Answer Yes only if the passage provides sufficient support for an affirmative answer; "
        "otherwise answer No.\n\n"
        f"Passage:\n{row['passage']}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Respond with exactly one word: Yes or No.\n"
        "Answer:"
    )


def encode_prompt(tokenizer, prompt: str, device):
    return tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def masked_score(model, tokenizer, prefix_ids, answer: str, normalize: bool) -> float:
    """Score candidate answers using log probabilities at masked positions."""
    answer_ids = tokenizer(
        " " + answer.strip(),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    if answer_ids.numel() == 0:
        return -math.inf

    mask_ids = torch.full_like(answer_ids, model.config.mask_token_id)
    input_ids = torch.cat([prefix_ids, mask_ids], dim=1)
    logits = model(input_ids, attention_mask=torch.ones_like(input_ids)).logits
    start = prefix_ids.size(1)
    logprobs = F.log_softmax(logits[:, start:], dim=-1)
    score = logprobs.gather(-1, answer_ids.unsqueeze(-1)).sum().item()
    return score / answer_ids.size(1) if normalize else score


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: list[str], label: int, normalize: bool) -> dict:
    """Score candidate answers and compute selection accuracy."""
    prefix_ids = encode_prompt(tokenizer, prompt, model.device)
    option_scores = [
        masked_score(model, tokenizer, prefix_ids, choice, normalize)
        for choice in choices
    ]
    predicted = max(range(len(option_scores)), key=option_scores.__getitem__)
    value = int(predicted == label)
    return {
        "predicted": predicted,
        "option_scores": option_scores,
        "mc1": {"acc": value, "score": value},
    }


def score_copa(model, tokenizer, row: dict, normalize: bool) -> dict:
    return score_choices(
        model,
        tokenizer,
        format_copa_prompt(row),
        row["choices"],
        row["label"],
        normalize,
    )


def score_storycloze(model, tokenizer, row: dict, normalize: bool) -> dict:
    return score_choices(
        model,
        tokenizer,
        format_storycloze_prompt(row),
        row["choices"],
        row["label"],
        normalize,
    )


def score_mmlu(model, tokenizer, row: dict, normalize: bool) -> dict:
    return score_choices(
        model,
        tokenizer,
        format_mmlu_prompt(row),
        row["choices"],
        row["label"],
        normalize,
    )


def score_wino(model, tokenizer, row: dict, normalize: bool) -> dict:
    return score_choices(
        model,
        tokenizer,
        format_wino_prompt(row),
        row["choices"],
        row["label"],
        normalize,
    )


def score_boolq(model, tokenizer, row: dict, normalize: bool) -> dict:
    return score_choices(
        model,
        tokenizer,
        format_boolq_prompt(row),
        row["choices"],
        row["label"],
        normalize,
    )


def evaluate(args) -> dict:
    """Load the selected dataset, evaluate MC1 with the model, and save per-sample results."""
    if args.dataset == "copa":
        rows = load_copa(args.data_path)
        score_row = score_copa
        description = "Evaluating COPA"
    elif args.dataset == "storycloze":
        rows = load_storycloze(args.data_path)
        score_row = score_storycloze
        description = "Evaluating StoryCloze"
    elif args.dataset == "mmlu":
        rows = load_mmlu(args.data_path)
        score_row = score_mmlu
        description = "Evaluating MMLU"
    elif args.dataset == "wino":
        rows = load_wino(args.data_path)
        score_row = score_wino
        description = "Evaluating Wino"
    elif args.dataset == "boolq":
        rows = load_boolq(args.data_path)
        score_row = score_boolq
        description = "Evaluating BoolQ"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model(args.model_dir)
    normalize = not args.no_length_normalization
    results = []
    correct = 0
    progress = tqdm(rows, desc=description, unit="sample")

    for number, row in enumerate(progress, start=1):
        scores = score_row(model, tokenizer, row, normalize)
        correct += scores["mc1"]["acc"]
        progress.set_postfix(sample=f"{number}/{len(rows)}", mc1=f"{correct / number:.4f}")
        results.append({**row, **scores})

    accuracy = correct / len(rows) if rows else 0.0
    report = {
        "config": {
            "dataset": args.dataset,
            "data_path": args.data_path,
            "model_dir": args.model_dir,
            "max_samples": args.max_samples,
            "length_normalization": normalize,
        },
        "summary": {
            "num_samples": len(rows),
            "mc1": {"acc": accuracy, "score": accuracy},
        },
        "results": results,
    }

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DLLM on BoolQ, COPA, StoryCloze, MMLU, or Wino with MC1.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="copa")
    parser.add_argument("--data_path")
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--output_path")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no_length_normalization",
        action="store_true",
        help="Use summed log-probability instead of mean log-probability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_data_path, default_output_path = default_paths(args.dataset)
    if args.data_path is None:
        args.data_path = default_data_path
    if args.output_path is None:
        args.output_path = default_output_path

    report = evaluate(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
