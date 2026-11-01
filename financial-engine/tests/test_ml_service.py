"""API tests for the ML risk service, with Feast and MLflow mocked."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import serve


class FakeOnlineFeatureResponse:
    def __init__(self, values):
        self.values = values

    def to_dict(self):
        return self.values


class FakeFeatureStore:
    def __init__(self):
        self.values = {
            "price": [331.35],
            "volume": [37593],
            "rolling_avg_volume": [40744.6],
        }
        self.request = None
        self.error = None

    def get_online_features(self, *, features, entity_rows):
        self.request = {"features": features, "entity_rows": entity_rows}
        if self.error:
            raise self.error
        return FakeOnlineFeatureResponse(self.values)


class FakeModel:
    def __init__(self):
        self.probabilities = [[0.2, 0.8]]
        self.error = None
        self.input_frame = None

    def predict_proba(self, input_frame):
        self.input_frame = input_frame.copy()
        if self.error:
            raise self.error
        return self.probabilities


class MLServiceTests(unittest.TestCase):
    def setUp(self):
        self.feature_store = FakeFeatureStore()
        self.model = FakeModel()

        patches = (
            patch.object(serve, "FeatureStore", return_value=self.feature_store),
            patch.object(serve.mlflow, "set_tracking_uri"),
            patch.object(
                serve.mlflow.xgboost,
                "load_model",
                return_value=self.model,
            ),
        )
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

        self.client = TestClient(serve.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_ready_endpoint_reports_loaded_dependencies(self):
        response = self.client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])
        self.assertTrue(response.json()["feature_store_initialized"])

    def test_prediction_uses_online_features_and_uppercases_ticker(self):
        response = self.client.post(
            "/predict_risk",
            json={"ticker": " aapl "},
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["ticker"], "AAPL")
        self.assertEqual(body["risk_category"], "HIGH")
        self.assertEqual(body["risk_probability"], 0.8)
        self.assertEqual(
            self.feature_store.request["entity_rows"],
            [{"ticker": "AAPL"}],
        )
        self.assertEqual(
            list(self.model.input_frame.columns),
            serve.FEATURE_COLUMNS,
        )
        self.assertEqual(
            self.model.input_frame.iloc[0]["daily_volatility"],
            serve.DAILY_VOLATILITY_BASELINE,
        )

    def test_blank_ticker_is_rejected(self):
        response = self.client.post(
            "/predict_risk",
            json={"ticker": "   "},
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_online_feature_returns_not_found(self):
        self.feature_store.values["volume"] = [None]

        response = self.client.post(
            "/predict_risk",
            json={"ticker": "AAPL"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("volume", response.json()["detail"])

    def test_feature_store_error_returns_service_unavailable(self):
        self.feature_store.error = RuntimeError("store unavailable")

        response = self.client.post(
            "/predict_risk",
            json={"ticker": "AAPL"},
        )

        self.assertEqual(response.status_code, 503)

    def test_model_error_returns_internal_server_error(self):
        self.model.error = RuntimeError("inference failed")

        response = self.client.post(
            "/predict_risk",
            json={"ticker": "AAPL"},
        )

        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
