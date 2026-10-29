import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_mock_market_data(num_records: int = 1000) -> pd.DataFrame:
    """Generates synthetic stock market tick data for testing."""
    np.random.seed(42)
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
    
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=i) for i in range(num_records)]
    
    data = {
        "timestamp": np.random.choice(timestamps, num_records),
        "ticker": np.random.choice(tickers, num_records),
        "price": np.round(np.random.uniform(100, 500, num_records), 2),
        "volume": np.random.randint(1000, 50000, num_records),
        "daily_volatility": np.round(np.random.uniform(0.01, 0.08, num_records), 4),
    }
    df = pd.DataFrame(data)
    return df.sort_values(by="timestamp").reset_index(drop=True)

def compute_silver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rolling risk metrics used downstream by classical ML models."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["ticker", "timestamp"])
    
    # Calculate rolling 5-period average volume and price volatility score
    df["rolling_avg_volume"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    df["price_change_ratio"] = (
        df.groupby("ticker")["price"]
        .transform(lambda x: x.pct_change().fillna(0))
    )
    df["composite_risk_feature"] = (
        df["daily_volatility"] * 0.6 + np.abs(df["price_change_ratio"]) * 0.4
    )
    
    return df

if __name__ == "__main__":
    os.makedirs("data/silver", exist_ok=True)
    raw_df = generate_mock_market_data(1000)
    silver_df = compute_silver_features(raw_df)
    
    # Save as Parquet (representing Silver Delta store)
    silver_df.to_parquet("data/silver/market_risk_features.parquet", index=False)
    print(f"Successfully processed {len(silver_df)} records into Silver layer.")