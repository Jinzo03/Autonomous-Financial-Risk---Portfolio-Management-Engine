import pandas as pd
import sys

def validate_silver_data(file_path: str) -> bool:
    """Validates schema, null values, and numeric ranges on Silver data."""
    df = pd.read_parquet(file_path)
    
    validation_passed = True
    
    # Rule 1: Key fields cannot be null
    if df[["timestamp", "ticker", "price"]].isnull().sum().sum() > 0:
        print("Validation Failed: Null values detected in key columns.")
        validation_passed = False
        
    # Rule 2: Price and Volume must be positive
    if (df["price"] <= 0).any() or (df["volume"] < 0).any():
        print("Validation Failed: Non-positive price or volume found.")
        validation_passed = False

    # Rule 3: Composite risk score must be bounded [0.0, 1.0]
    if not df["composite_risk_feature"].between(0.0, 1.0).all():
        print("Validation Failed: Risk metrics fell outside expected bounds [0.0, 1.0].")
        validation_passed = False

    if validation_passed:
        print("Data Quality Gate Passed: All assertions succeeded.")
    return validation_passed

if __name__ == "__main__":
    is_valid = validate_silver_data("data/silver/market_risk_features.parquet")
    if not is_valid:
        sys.exit(1)