from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_TRUTHFULQA_PATH = Path(
    "./row_data/TruthfulQA_v1.csv"
)


def _split_answers(text: str) -> list[str]:
    return [answer.strip() for answer in text.split(";") if answer.strip()]


def load_truthfulqa(path: str | Path = DEFAULT_TRUTHFULQA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "Type": "type",
            "Category": "category",
            "Question": "question",
            "Best Answer": "best_answer",
            "Correct Answers": "correct_answers",
            "Incorrect Answers": "incorrect_answers",
            "Source": "source",
        }
    )
    df["correct_answers"] = df["correct_answers"].apply(_split_answers)
    df["incorrect_answers"] = df["incorrect_answers"].apply(_split_answers)
    return df


if __name__ == "__main__":
    df = load_truthfulqa()
    print(f"Loaded {len(df)} examples")
    print(df.head(1).to_dict("records")[0])
