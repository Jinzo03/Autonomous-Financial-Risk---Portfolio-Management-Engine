import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.xgboost
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_risk_model(data_path: str | os.PathLike = PROJECT_ROOT / "data" / "silver" / "market_risk_features.parquet"):
    """Trains an XGBoost risk model and logs metrics/artifacts to MLflow."""
    # Set MLflow experiment target
    mlflow.set_experiment("market_risk_assessment")
    
    # Load dataset generated during DataOps phase
    df = pd.read_parquet(Path(data_path).expanduser().resolve())
    
    # Feature engineering: create binary target variable (High Risk if composite_risk > 0.08)
    df["target_high_risk"] = (df["composite_risk_feature"] > 0.08).astype(int)
    
    feature_cols = ["price", "volume", "rolling_avg_volume", "daily_volatility"]
    X = df[feature_cols]
    y = df["target_high_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model Hyperparameters
    params = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    with mlflow.start_run() as run:
        # Train XGBoost Classifier
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        # Log Hyperparameters & Metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", auc)
        
        # Log and register model artifact
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="risk_xgb_model",
            registered_model_name="XGBoost_Risk_Scorer"
        )
        
        run_id = run.info.run_id
        print(f"MLflow Run Successful! Run ID: {run_id}")
        print(f"Metrics -> Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        return run_id

if __name__ == "__main__":
    train_risk_model()
