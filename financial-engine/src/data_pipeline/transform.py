import os
import pandas as pd
import numpy as np

def compute_silver_features(input_path: str = "data/bronze/raw_market_ticks.parquet", output_path: str = "data/silver/market_risk_features.parquet") -> str:
    """Reads Bronze layer data, applies cleaning, and outputs Silver metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_parquet(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["ticker", "timestamp"])
    
    # Compute rolling risk features
    df["rolling_avg_volume"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    df["price_change_ratio"] = (
        df.groupby("ticker")["price"]
        .transform(lambda x: x.pct_change().fillna(0))
    )
    df["composite_risk_feature"] = np.clip(
        (df["daily_volatility"] * 0.6 + np.abs(df["price_change_ratio"]) * 0.4),
        0.0,
        1.0
    )
    
    df.to_parquet(output_path, index=False)
    print(f"[Silver] Features transformed: {len(df)} records -> {output_path}")
    return output_path

if __name__ == "__main__":
    compute_silver_features()