from feast import FeatureStore
import pandas as pd

def test_online_lookup():
    # Point Feast to the feature store directory
    store = FeatureStore(repo_path="src/data_pipeline/feast_repo")
    
    # Fetch real-time feature vector for downstream MLOps consumption
    entity_rows = [
        {"ticker": "AAPL"},
        {"ticker": "NVDA"}
    ]
    
    features_to_fetch = [
        "market_risk_features:price",
        "market_risk_features:rolling_avg_volume",
        "market_risk_features:composite_risk_feature"
    ]
    
    response = store.get_online_features(
        entity_rows=entity_rows,
        features=features_to_fetch
    ).to_dict()
    
    df = pd.DataFrame(response)
    print("\n--- Feast Online Feature Store Lookup Success ---")
    print(df)

if __name__ == "__main__":
    test_online_lookup()