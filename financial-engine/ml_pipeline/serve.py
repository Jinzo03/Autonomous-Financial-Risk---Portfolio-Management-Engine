import os
import logging

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import mlflow
import mlflow.xgboost

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from feast import FeatureStore


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

FEAST_REPO_PATH = Path(
    os.getenv(
        "FEAST_REPO_PATH",
        "src/data_pipeline/feast_repo"
    )
).resolve()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "XGBoost_Risk_Scorer"
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion"
)

RISK_THRESHOLD = float(
    os.getenv("RISK_THRESHOLD", "0.5")
)

DAILY_VOLATILITY_BASELINE = float(
    os.getenv("DAILY_VOLATILITY_BASELINE", "0.035")
)

FEATURE_COLUMNS = [
    "price",
    "volume",
    "rolling_avg_volume",
    "daily_volatility"
]

FEAST_FEATURES = [
    "market_risk_features:price",
    "market_risk_features:volume",
    "market_risk_features:rolling_avg_volume"
]


# ============================================================
# Pydantic Schemas
# ============================================================

class RiskRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {"ticker": "AAPL"}
            ]
        }
    )

    ticker: str = Field(
        min_length=1,
        max_length=20,
        description="Stock ticker symbol"
    )


class RiskResponse(BaseModel):
    ticker: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_category: str
    features_used: dict[str, float]


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting MLOps Risk Prediction Service...")

    app.state.feature_store = None
    app.state.model = None

    try:
        # Initialize Feast Feature Store
        app.state.feature_store = FeatureStore(
            repo_path=str(FEAST_REPO_PATH)
        )

        logger.info("Feast Feature Store initialized.")

        # Configure MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load registered model using alias
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        app.state.model = mlflow.xgboost.load_model(
            model_uri
        )

        logger.info(
            "MLflow model loaded successfully: %s",
            model_uri
        )

    except Exception:
        logger.exception(
            "Failed to initialize prediction service."
        )

        # Fail startup instead of serving an unusable application
        raise

    yield

    # Shutdown cleanup
    logger.info("Shutting down MLOps Risk Prediction Service...")

    app.state.model = None
    app.state.feature_store = None


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="MLOps Risk Prediction Microservice",
    description=(
        "Financial risk prediction API powered by "
        "FastAPI, Feast, MLflow and XGBoost."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# Health Endpoints
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MLOps Risk Serving Engine"
    }


@app.get("/ready")
def readiness_check(request: Request):

    model_ready = request.app.state.model is not None

    feature_store_ready = (
        request.app.state.feature_store is not None
    )

    if not model_ready or not feature_store_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not ready."
        )

    return {
        "status": "ready",
        "model_loaded": model_ready,
        "feature_store_initialized": feature_store_ready
    }


# ============================================================
# Feature Retrieval
# ============================================================

def fetch_online_features(
    feature_store: FeatureStore,
    ticker: str
) -> dict[str, float]:

    try:
        response = feature_store.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[
                {"ticker": ticker}
            ]
        ).to_dict()

    except Exception as exc:
        logger.exception(
            "Feast feature retrieval failed for %s",
            ticker
        )

        raise HTTPException(
            status_code=503,
            detail="Feature store temporarily unavailable."
        ) from exc

    required_features = [
        "price",
        "volume",
        "rolling_avg_volume"
    ]

    features = {}

    for feature_name in required_features:

        values = response.get(feature_name)

        if not values or values[0] is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Missing feature '{feature_name}' "
                    f"for ticker '{ticker}'."
                )
            )

        try:
            value = float(values[0])

            if not pd.notna(value):
                raise ValueError("Non-finite feature value")

            if value in (float("inf"), float("-inf")):
                raise ValueError("Infinite feature value")

            features[feature_name] = value

        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid feature '{feature_name}' "
                    f"for ticker '{ticker}'."
                )
            ) from exc

    return features


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict_risk",
    response_model=RiskResponse
)
def predict_risk(
    payload: RiskRequest,
    request: Request
):

    model = request.app.state.model
    feature_store = request.app.state.feature_store

    if model is None or feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not initialized."
        )

    ticker = payload.ticker.upper()

    # --------------------------------------------------------
    # 1. Retrieve features from Feast
    # --------------------------------------------------------

    features = fetch_online_features(
        feature_store=feature_store,
        ticker=ticker
    )

    # --------------------------------------------------------
    # 2. Prepare model input
    # --------------------------------------------------------

    model_features = {
        **features,
        "daily_volatility": DAILY_VOLATILITY_BASELINE
    }

    input_df = pd.DataFrame(
        [model_features],
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------------
    # 3. Run XGBoost inference
    # --------------------------------------------------------

    try:
        probabilities = model.predict_proba(input_df)

        risk_probability = float(
            probabilities[0][1]
        )

        if not 0.0 <= risk_probability <= 1.0:
            raise ValueError(
                "Model returned an invalid probability."
            )

    except Exception as exc:
        logger.exception(
            "Model inference failed for ticker %s",
            ticker
        )

        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        ) from exc

    # --------------------------------------------------------
    # 4. Risk classification
    # --------------------------------------------------------

    risk_category = (
        "HIGH"
        if risk_probability >= RISK_THRESHOLD
        else "LOW"
    )

    logger.info(
        "Prediction completed | ticker=%s | risk=%.4f | category=%s",
        ticker,
        risk_probability,
        risk_category
    )

    return RiskResponse(
        ticker=ticker,
        risk_probability=round(
            risk_probability,
            4
        ),
        risk_category=risk_category,
        features_used=model_features
    )


# ============================================================
# Local Development
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )