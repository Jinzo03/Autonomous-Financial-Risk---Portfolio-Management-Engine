"""Register Feast definitions and load the latest Silver features online."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore

from src.data_pipeline.feast_repo.features import market_risk_fv, ticker_entity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "src" / "data_pipeline" / "feast_repo"
SILVER_DATASET = PROJECT_ROOT / "data" / "silver" / "market_risk_features.parquet"


def materialize_features() -> None:
    """Apply the repository and materialize available batch rows to SQLite."""
    if not SILVER_DATASET.is_file():
        raise FileNotFoundError(f"Silver feature dataset not found: {SILVER_DATASET}")

    timestamps = pd.to_datetime(
        pd.read_parquet(SILVER_DATASET, columns=["timestamp"])["timestamp"],
        errors="coerce",
    ).dropna()
    if timestamps.empty:
        raise ValueError("Silver feature dataset has no usable event timestamps.")
    start_date = timestamps.min().to_pydatetime()
    end_date = (timestamps.max() + pd.Timedelta(microseconds=1)).to_pydatetime()

    store = FeatureStore(repo_path=str(FEAST_REPO))
    store.apply([ticker_entity, market_risk_fv])
    store.materialize(start_date=start_date, end_date=end_date)
    print(
        f"[Feast] Materialized online features from {SILVER_DATASET} "
        f"({start_date} through {end_date})"
    )


if __name__ == "__main__":
    materialize_features()
