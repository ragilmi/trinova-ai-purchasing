"""
test_model_comparison.py
Unit and integration tests for Linear Regression baseline, Random Forest, and XGBoost
model comparison and FastAPI endpoint preservation using standard unittest library.
"""

import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_comparison import (
    evaluate_classifier_metrics,
    evaluate_linear_regression_baseline,
    run_model_comparison,
)
from app.services.train_model import train

client = TestClient(app)


class TestModelComparison(unittest.TestCase):

    def test_evaluate_linear_regression_baseline(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.95])

        metrics = evaluate_linear_regression_baseline(y_true, y_score, threshold=0.50)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("auc_roc", metrics)
        self.assertIsNone(metrics["log_loss"])
        self.assertEqual(metrics["log_loss_str"].strip(), "N/A")
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_evaluate_classifier_metrics(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.95])

        metrics = evaluate_classifier_metrics(y_true, y_pred, y_prob)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("auc_roc", metrics)
        self.assertIn("log_loss", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_run_model_comparison(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.choice([0, 1], size=100))
        X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.Series(np.random.choice([0, 1], size=20))

        result = run_model_comparison(X_train, y_train, X_test, y_test)

        self.assertIn("linear_regression", result)
        self.assertIn("random_forest", result)
        self.assertIn("xgboost", result)
        self.assertIn("best_model", result)
        self.assertIn(result["best_model"], ["Linear Regression", "Random Forest", "XGBoost Classifier"])
        self.assertIn("f1_score", result["linear_regression"])
        self.assertIn("f1_score", result["random_forest"])
        self.assertIn("f1_score", result["xgboost"])

    def test_train_end_to_end(self):
        metrics = train()

        self.assertIn("samples_trained", metrics)
        self.assertIn("samples_tested", metrics)
        self.assertIn("model_comparison", metrics)
        comp = metrics["model_comparison"]
        self.assertIn("linear_regression", comp)
        self.assertIn("random_forest", comp)
        self.assertIn("xgboost", comp)
        self.assertIn(comp["best_model"], ["Linear Regression", "Random Forest", "XGBoost Classifier"])

    def test_api_health_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_ready", data)

    def test_api_predict_endpoint(self):
        payload = {
            "supplier_id": 99,
            "supplier_price": 450000,
            "lead_time_days": 4,
            "claim_rate": 0.03,
            "on_time_rate": 0.92,
            "order_frequency": 18,
        }
        response = client.post("/predict/supplier-risk", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["supplier_id"], 99)
        self.assertIn(data["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertTrue(0.0 <= data["delay_probability"] <= 1.0)
        self.assertTrue(0 <= data["late_probability"] <= 100)


if __name__ == "__main__":
    unittest.main()
