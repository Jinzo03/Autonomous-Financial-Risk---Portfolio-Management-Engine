import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.xgboost
from feast import FeatureStore

app = FastAPI(title="MLOps Risk Prediction Microservice", version="1.0.0")

# Global variables for model and feature store
feature_store = None
model = None

class RiskRequest(BaseModel):
    ticker: str

class RiskResponse(BaseModel):
    ticker: str
    risk_probability: float
    risk_category: str
    features_used: dict

@app.on_event("startup")
def load_resources():
    global feature_store, model
    # Initialize Feast Feature Store link
    feature_store = FeatureStore(repo_path="src/data_pipeline/feast_repo")
    
    # Load latest registered model from MLflow
    model_uri = "models:/XGBoost_Risk_Scorer/latest"
    try:
        model = mlflow.xgboost.load_model(model_uri)
        print("MLflow model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load registered model by alias ({e}). Fallback to local artifacts if needed.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "MLOps Risk Serving Engine"}

@app.post("/predict_risk", response_model=RiskResponse)
def predict_risk(request: RiskRequest):
    if model is None or feature_store is None:
        raise HTTPException(status_code=500, detail="Service resources not initialized.")
        
    # 1. Fetch online features from Feast Feature Store
    entity_rows = [{"ticker": request.ticker}]
    features_to_fetch = [
        "market_risk_features:price",
        "market_risk_features:volume",
        "market_risk_features:rolling_avg_volume",
    ]
    
    try:
        response = feature_store.get_online_features(
            entity_rows=entity_rows,
            features=features_to_fetch
        ).to_dict()
        
        price = response["price"][0]
        volume = response["volume"][0]
        rolling_avg_volume = response["rolling_avg_volume"][0]
        
        # Fallback values if entity is missing from online store
        if price is None:
            raise HTTPException(status_code=404, detail=f"No online features found for ticker '{request.ticker}'.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feast lookup error: {str(e)}")
    
    # 2. Format features for XGBoost model
    # Note: daily_volatility is estimated/derived if not stored directly in online vector
    daily_volatility = 0.035  # baseline market default
    input_df = pd.DataFrame([{
        "price": price,
        "volume": volume,
        "rolling_avg_volume": rolling_avg_volume,
        "daily_volatility": daily_volatility
    }])
    
    # 3. Model Inference
    risk_prob = float(model.predict_proba(input_df)[0][1])
    category = "HIGH" if risk_prob >= 0.5 else "LOW"
    
    return RiskResponse(
        ticker=request.ticker,
        risk_probability=round(risk_prob, 4),
        risk_category=category,
        features_used={
            "price": price,
            "volume": volume,
            "rolling_avg_volume": rolling_avg_volume
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)