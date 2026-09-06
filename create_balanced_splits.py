"""
Create class-balanced CSV splits by downsampling the majority class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42
LABEL_NAMES = {
    0: "Real",
    1: "Manipulated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance train/val/test CSV files by downsampling the majority class in each split."
    )
    parser.add_argument("--train_csv", required=True, type=Path, help="Path to train.csv.")
    parser.add_argument("--val_csv", required=True, type=Path, help="Path to val.csv.")
    parser.add_argument("--test_csv", required=True, type=Path, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for balanced CSV outputs.")
    return parser.parse_args()


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return stable binary class counts using dissertation label names."""
    if "label" not in df.columns:
        raise ValueError("Input CSV is missing required column: label")

    counts = df["label"].value_counts().to_dict()
    return {
        LABEL_NAMES[0]: int(counts.get(0, 0)),
        LABEL_NAMES[1]: int(counts.get(1, 0)),
    }


def validate_labels(df: pd.DataFrame, split_name: str) -> None:
    labels = set(df["label"].dropna().astype(int).unique().tolist())
    unexpected = labels.difference({0, 1})
    if unexpected:
        raise ValueError(f"{split_name}.csv contains labels outside {{0, 1}}: {sorted(unexpected)}")


def balance_split(df: pd.DataFrame, split_name: str, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Balance one split independently by downsampling the majority class.

    Rows are sampled only from the current split, so no samples are mixed across
    train/val/test. All columns, including group_id, are preserved unchanged.
    """
    validate_labels(df, split_name)

    before_counts = class_counts(df)
    real_df = df[df["label"].astype(int) == 0]
    manipulated_df = df[df["label"].astype(int) == 1]

    minority_count = min(len(real_df), len(manipulated_df))
    warnings: list[str] = []

    if minority_count == 0:
        balanced_df = df.copy()
        warnings.append(
            f"{split_name}: cannot balance because one class has zero samples; wrote original split unchanged."
        )
    else:
        real_balanced = real_df.sample(n=minority_count, random_state=seed)
        manipulated_balanced = manipulated_df.sample(n=minority_count, random_state=seed)
        balanced_df = (
            pd.concat([real_balanced, manipulated_balanced], axis=0)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

    after_counts = class_counts(balanced_df)
    dropped_rows = int(len(df) - len(balanced_df))

    summary = {
        "before": before_counts,
        "after": after_counts,
        "rows_before": int(len(df)),
        "rows_after": int(len(balanced_df)),
        "dropped_rows": dropped_rows,
        "warnings": warnings,
    }
    return balanced_df, summary


def save_summary(summary: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "balanced_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_split_summary(split_name: str, split_summary: dict[str, Any]) -> None:
    before = split_summary["before"]
    after = split_summary["after"]
    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"Before: Real={before['Real']} | Manipulated={before['Manipulated']}")
    print(f"After:  Real={after['Real']} | Manipulated={after['Manipulated']}")
    print(f"Dropped rows: {split_summary['dropped_rows']}")
    for warning in split_summary["warnings"]:
        print(f"Warning: {warning}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_inputs = {
        "train": args.train_csv.expanduser().resolve(),
        "val": args.val_csv.expanduser().resolve(),
        "test": args.test_csv.expanduser().resolve(),
    }

    summary: dict[str, Any] = {
        "seed": SEED,
        "method": "independent_per_split_majority_downsampling",
        "outputs": {},
        "splits": {},
    }

    for split_name, csv_path in split_inputs.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"{split_name}.csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        balanced_df, split_summary = balance_split(df, split_name, SEED)

        output_path = output_dir / f"{split_name}_balanced.csv"
        balanced_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        summary["outputs"][f"{split_name}_balanced_csv"] = str(output_path)
        summary["splits"][split_name] = split_summary
        print_split_summary(split_name, split_summary)

    save_summary(summary, output_dir)
    print(f"\nSaved balanced CSVs and summary to: {output_dir}")


if __name__ == "__main__":
    main()
