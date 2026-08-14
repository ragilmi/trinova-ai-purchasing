"""
predict.py
Runs supplier-risk predictions using the configured production model.

The active model (XGBoost or Linear Regression) is selected entirely through
the PURCHASING_AI_MODEL environment variable — see model_factory.py.

Public API is unchanged:
    get_model()                 -> returns active production model (backward compat)
    invalidate_model_cache()    -> clears model cache after retraining
    predict_supplier_risk(dict) -> returns {risk_level, delay_probability, late_probability}
"""

import logging
from typing import Literal

from app.services.preprocess import preprocess_single
from app.services.model_factory import (
    MODEL_LINEAR_REGRESSION,
    get_active_model_name,
    get_production_model,
    invalidate_production_model_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible public helpers
# ---------------------------------------------------------------------------

def get_model():
    """Return the cached production model.

    Backward-compatible wrapper around model_factory.get_production_model().
    Callers that previously imported get_model() from predict.py continue
    to work without changes.
    """
    return get_production_model()


def invalidate_model_cache() -> None:
    """Clear the production model cache after retraining.

    Delegates to model_factory.invalidate_production_model_cache() so that
    both the factory cache and any future get_model() / predict calls pick up
    the newly saved artifact.
    """
    invalidate_production_model_cache()
    logger.info("[AI] Model cache invalidated — will reload on next prediction.")


# ---------------------------------------------------------------------------
# Risk level resolver (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_supplier_risk(input_data: dict) -> dict:
    """Run a single supplier-risk prediction using the active production model.

    Both XGBoost and Linear Regression pass through the identical shared
    preprocessing pipeline (preprocess_single) so the feature vector is
    always the same regardless of which model is active.

    Args:
        input_data: Dict with keys matching SupplierInput:
            - supplier_price   (float)
            - lead_time_days   (int)
            - claim_rate       (float, 0–1)
            - on_time_rate     (float, 0–1)
            - order_frequency  (int)

    Returns:
        Dict — response schema is IDENTICAL regardless of active model:
            {
                "risk_level":         "LOW" | "MEDIUM" | "HIGH",
                "delay_probability":  float,   # 0.0 – 1.0
                "late_probability":   int,     # percentage 0–100
            }
    """
    model      = get_production_model()
    model_name = get_active_model_name()

    # Shared preprocessing — identical feature vector for both models
    X = preprocess_single(input_data)

    if model_name == MODEL_LINEAR_REGRESSION:
        # Linear Regression produces a continuous score; clip to [0, 1]
        # so that _resolve_risk_level() and the percentage conversion are
        # consistent with XGBoost's predict_proba output range.
        import numpy as np
        raw_score  = float(model.predict(X)[0])
        delay_prob = float(round(float(np.clip(raw_score, 0.0, 1.0)), 4))
    else:
        # XGBoost: use predict_proba — class 1 = late_delivery
        proba      = model.predict_proba(X)[0]
        delay_prob = float(round(proba[1], 4))

    risk_level = _resolve_risk_level(delay_prob)
    late_pct   = int(round(delay_prob * 100))

    logger.debug(
        "[AI:%s] Prediction — risk_level=%s, delay_probability=%.4f",
        model_name, risk_level, delay_prob,
    )

    return {
        "risk_level":        risk_level,
        "delay_probability": delay_prob,
        "late_probability":  late_pct,
    }
