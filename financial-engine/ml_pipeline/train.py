import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "market_risk_features.parquet"
)

EXPERIMENT_NAME = "market_risk_assessment"
MODEL_NAME = "XGBoost_Risk_Scorer"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

# Set RISK_LABEL_THRESHOLD to override the data-driven median cutoff.
RISK_LABEL_THRESHOLD_OVERRIDE = os.getenv("RISK_LABEL_THRESHOLD")

FEATURE_COLUMNS = [
    "price",
    "volume",
    "rolling_avg_volume",
    "daily_volatility",
]

TARGET_COLUMN = "target_high_risk"

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Dataset Validation
# ============================================================

def load_and_validate_data(data_path: Path) -> pd.DataFrame:

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_parquet(data_path)

    required_columns = FEATURE_COLUMNS + [
        "composite_risk_feature"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert model inputs to numeric values
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Replace infinities with NaN
    df[required_columns] = df[
        required_columns
    ].replace([np.inf, -np.inf], np.nan)

    initial_rows = len(df)

    # Remove rows with missing required values
    df = df.dropna(
        subset=required_columns
    ).copy()

    logger.info(
        "Removed %d invalid rows.",
        initial_rows - len(df)
    )

    if df.empty:
        raise ValueError(
            "No valid training samples available."
        )

    # Use the median composite risk as the default decision boundary so the
    # target is approximately balanced. An environment override is useful for
    # experiments that need a fixed, comparable threshold.
    risk_label_threshold = (
        float(RISK_LABEL_THRESHOLD_OVERRIDE)
        if RISK_LABEL_THRESHOLD_OVERRIDE is not None
        else float(df["composite_risk_feature"].median())
    )

    # Construct binary classification target
    df[TARGET_COLUMN] = (
        df["composite_risk_feature"]
        > risk_label_threshold
    ).astype(int)
    df.attrs["risk_label_threshold"] = risk_label_threshold

    class_counts = df[TARGET_COLUMN].value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "Training requires both HIGH and LOW risk classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class must contain at least two samples."
        )

    logger.info(
        "Dataset validated: %d samples.",
        len(df)
    )

    logger.info(
        "Class distribution: %s",
        class_counts.to_dict()
    )

    return df


# ============================================================
# Model Training
# ============================================================

def train_risk_model(
    data_path: str | os.PathLike = DEFAULT_DATA_PATH
) -> str:

    data_path = Path(
        data_path
    ).expanduser().resolve()

    # Configure MLflow
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # Load validated dataset
    df = load_and_validate_data(data_path)
    risk_label_threshold = df.attrs["risk_label_threshold"]

    X = df[FEATURE_COLUMNS].astype("float64")
    y = df[TARGET_COLUMN]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Weight the positive class based on the training split. With a median
    # cutoff this should be near 1, but it also handles ties or overrides.
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Training split must contain both HIGH and LOW risk samples."
        )

    scale_pos_weight = negative_count / positive_count

    # XGBoost hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    with mlflow.start_run(
        run_name="xgboost_risk_training"
    ) as run:

        logger.info(
            "Starting MLflow run: %s",
            run.info.run_id
        )

        # ----------------------------------------------------
        # 1. Log configuration
        # ----------------------------------------------------

        mlflow.log_params(params)

        mlflow.log_params({
            "risk_label_threshold": risk_label_threshold,
            "positive_training_samples": positive_count,
            "negative_training_samples": negative_count,
            "test_size": 0.2,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
        })

        mlflow.set_tags({
            "model_type": "XGBoost",
            "task": "binary_classification",
            "domain": "financial_risk",
            "pipeline_stage": "training",
        })

        # ----------------------------------------------------
        # 2. Train model
        # ----------------------------------------------------

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ----------------------------------------------------
        # 3. Evaluate model
        # ----------------------------------------------------

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = {
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions,
                zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test, probabilities
            ),
        }

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------
        # 4. Log confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        mlflow.log_dict(
            {
                "labels": ["LOW", "HIGH"],
                "matrix": cm.tolist(),
            },
            "evaluation/confusion_matrix.json"
        )

        # ----------------------------------------------------
        # 5. Model signature
        # ----------------------------------------------------

        input_example = X_train.head(5)

        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        # ----------------------------------------------------
        # 6. Log and register model
        # ----------------------------------------------------

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="risk_xgb_model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
            model_format="ubj",
            metadata={
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "risk_label_threshold": risk_label_threshold,
            },
        )

        # ----------------------------------------------------
        # 7. Final logging
        # ----------------------------------------------------

        logger.info(
            "Model registered successfully: %s",
            MODEL_NAME
        )

        logger.info(
            "Logged model URI: %s",
            model_info.model_uri
        )

        for metric_name, value in metrics.items():
            logger.info(
                "%s: %.4f",
                metric_name,
                value
            )

        return run.info.run_id


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    train_risk_model()
