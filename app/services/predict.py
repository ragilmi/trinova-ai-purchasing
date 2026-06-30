"""
predict.py
Loads the trained XGBoost model and runs supplier-risk predictions.
"""

import logging
from typing import Literal

from xgboost import XGBClassifier

from app.services.preprocess import preprocess_single
from app.services.train_model import load_model

logger = logging.getLogger(__name__)

# Module-level model cache — loaded once on first prediction call
_model: XGBClassifier | None = None


def get_model() -> XGBClassifier:
    """Return the cached model, loading it from disk on first access."""
    global _model
    if _model is None:
        logger.info("Loading model into memory...")
        _model = load_model()
        logger.info("Model loaded successfully.")
    return _model


def invalidate_model_cache() -> None:
    """Force the next prediction call to reload the model from disk.

    Should be called after a successful retraining so the service
    always uses the latest artifact.
    """
    global _model
    _model = None
    logger.info("Model cache invalidated — will reload on next prediction.")


def _resolve_risk_level(probability: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """Map a delay probability to a human-readable risk level.

    Thresholds:
        LOW    : probability < 0.30
        MEDIUM : 0.30 <= probability < 0.60
        HIGH   : probability >= 0.60
    """
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    return "HIGH"


def predict_supplier_risk(input_data: dict) -> dict:
    """Run a single supplier-risk prediction.

    Args:
        input_data: Dict with keys matching SupplierInput:
            - supplier_price   (float)
            - lead_time_days   (int)
            - claim_rate       (float, 0–1)
            - on_time_rate     (float, 0–1)
            - order_frequency  (int)

    Returns:
        Dict:
            {
                "risk_level":         "LOW" | "MEDIUM" | "HIGH",
                "delay_probability":  float,   # 0.0 – 1.0
                "late_probability":   int,     # percentage 0–100
            }
    """
    model = get_model()

    X = preprocess_single(input_data)
    proba = model.predict_proba(X)[0]

    # proba[1] = probability of class 1 (late_delivery = True)
    delay_prob = float(round(proba[1], 4))
    risk_level = _resolve_risk_level(delay_prob)
    late_pct = int(round(delay_prob * 100))

    logger.debug(
        "Prediction — risk_level=%s, delay_probability=%.4f", risk_level, delay_prob
    )

    return {
        "risk_level": risk_level,
        "delay_probability": delay_prob,
        "late_probability": late_pct,
    }
