"""Build the curated model-training dataset from validated Silver features."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "market_risk_features.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "gold" / "risk_training_dataset.parquet"
FEATURE_COLUMNS = ["price", "volume", "rolling_avg_volume", "daily_volatility"]
TARGET_COLUMN = "target_high_risk"


def build_gold_dataset(
    input_path: str | os.PathLike = DEFAULT_INPUT_PATH,
    output_path: str | os.PathLike = DEFAULT_OUTPUT_PATH,
    *,
    risk_label_threshold: float | None = None,
) -> str:
    """Select model inputs and add a stable, recorded binary target label."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Silver feature dataset not found: {source}")

    frame = pd.read_parquet(source)
    required = ["timestamp", "ticker", *FEATURE_COLUMNS, "composite_risk_feature"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Silver dataset is missing required columns: {missing}")

    numeric_columns = [*FEATURE_COLUMNS, "composite_risk_feature"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", "ticker", *numeric_columns]).copy()
    if frame.empty:
        raise ValueError("Silver dataset has no valid rows for model training.")

    if risk_label_threshold is None:
        override = os.getenv("RISK_LABEL_THRESHOLD", "").strip()
        risk_label_threshold = (
            float(override)
            if override
            else float(frame["composite_risk_feature"].median())
        )
    if not np.isfinite(risk_label_threshold):
        raise ValueError("Risk label threshold must be finite.")

    frame[TARGET_COLUMN] = (
        frame["composite_risk_feature"] > risk_label_threshold
    ).astype("int8")
    class_counts = frame[TARGET_COLUMN].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "Gold training data must contain at least two examples from each risk class."
        )

    gold_columns = [
        "timestamp",
        "ticker",
        *FEATURE_COLUMNS,
        "composite_risk_feature",
        TARGET_COLUMN,
    ]
    gold = frame[gold_columns].copy()
    gold["risk_label_threshold"] = float(risk_label_threshold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(destination, index=False)
    print(
        f"[Gold] Training dataset written: {len(gold)} rows -> {destination}; "
        f"label threshold={risk_label_threshold:.6f}; classes={class_counts.to_dict()}"
    )
    return str(destination)


if __name__ == "__main__":
    build_gold_dataset()
