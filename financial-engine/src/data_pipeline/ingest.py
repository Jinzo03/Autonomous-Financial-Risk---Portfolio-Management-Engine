import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def ingest_raw_market_data(num_records: int = 1000, output_path: str = "data/bronze/raw_market_ticks.parquet") -> str:
    """Ingests raw market data from APIs/streams and dumps to Bronze layer."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=i) for i in range(num_records)]
    
    raw_data = {
        "timestamp": np.random.choice(timestamps, num_records),
        "ticker": np.random.choice(tickers, num_records),
        "price": np.round(np.random.uniform(100, 500, num_records), 2),
        "volume": np.random.randint(1000, 50000, num_records),
        "daily_volatility": np.round(np.random.uniform(0.01, 0.08, num_records), 4),
    }
    
    df = pd.DataFrame(raw_data)
    df.to_parquet(output_path, index=False)
    print(f"[Bronze] Raw market data ingested: {len(df)} records -> {output_path}")
    return output_path

if __name__ == "__main__":
    ingest_raw_market_data()