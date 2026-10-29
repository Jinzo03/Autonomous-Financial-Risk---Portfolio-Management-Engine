from datetime import timedelta
from pathlib import Path
from feast import (
    Entity,
    Field,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.types import Float64, Int64, String

# Define the primary key entity
ticker_entity = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    join_keys=["ticker"],
    description="Stock ticker symbol",
)

# Source pointing to our Gold/Silver storage
batch_source = FileSource(
    name="market_risk_source",
    # Resolve from the project root, not the Feast repository directory.
    path=str(Path(__file__).resolve().parents[3] / "data" / "silver" / "market_risk_features.parquet"),
    timestamp_field="timestamp",
)

# Feature view consumed by downstream MLOps models
market_risk_fv = FeatureView(
    name="market_risk_features",
    entities=[ticker_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="price", dtype=Float64),
        Field(name="volume", dtype=Int64),
        Field(name="rolling_avg_volume", dtype=Float64),
        Field(name="composite_risk_feature", dtype=Float64),
    ],
    online=True,
    source=batch_source,
    tags={"team": "quant_risk"},
)
