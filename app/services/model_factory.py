"""
model_factory.py
Centralized production model selection for the Trinova Purchasing AI service.

Configuration
-------------
Set the environment variable PURCHASING_AI_MODEL to one of:

    xgboost           — XGBoost Classifier  (default)
    linear_regression — Linear Regression Classifier

Example (.env or Railway environment variable):

    PURCHASING_AI_MODEL=xgboost
    PURCHASING_AI_MODEL=linear_regression

Design rules
------------
- ONE authoritative configuration source (this file reads the env var once).
- NO silent fallback: an invalid value or a missing artifact raises immediately.
- Both models share the exact same preprocessing pipeline (preprocess_single).
- The factory is the only place that knows which model is active.
- predict.py calls get_production_model() and never references model types directly.
"""

import logging
import os
from pathlib import Path
from typing import Union

import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ── Allowed configuration values ──────────────────────────────────────────────
MODEL_XGBOOST           = "xgboost"
MODEL_LINEAR_REGRESSION = "linear_regression"
ALLOWED_MODELS          = {MODEL_XGBOOST, MODEL_LINEAR_REGRESSION}

# ── Artifact paths (resolved relative to this file so they work anywhere) ─────
_MODELS_DIR = Path(__file__).parent.parent / "models"
XGBOOST_MODEL_PATH           = _MODELS_DIR / "xgboost_model.pkl"
LINEAR_REGRESSION_MODEL_PATH = _MODELS_DIR / "linear_regression_model.pkl"

# Type alias for the two supported production model types
ProductionModel = Union[XGBClassifier, LinearRegression]


def get_selected_model_name() -> str:
    """Read and validate the PURCHASING_AI_MODEL environment variable.

    Returns:
        Normalised model name string (one of ALLOWED_MODELS).

    Raises:
        ValueError: If the value is set to an unrecognised string.
    """
    raw = os.getenv("PURCHASING_AI_MODEL", MODEL_XGBOOST).strip().lower()

    if raw not in ALLOWED_MODELS:
        raise ValueError(
            f"[AI] Invalid PURCHASING_AI_MODEL value: '{raw}'. "
            f"Allowed values: {sorted(ALLOWED_MODELS)}. "
            "Fix your environment configuration and restart the service."
        )

    return raw


def get_artifact_path(model_name: str) -> Path:
    """Return the artifact path for a given model name.

    Args:
        model_name: One of ALLOWED_MODELS.

    Returns:
        Path to the corresponding .pkl file.
    """
    if model_name == MODEL_XGBOOST:
        return XGBOOST_MODEL_PATH
    if model_name == MODEL_LINEAR_REGRESSION:
        return LINEAR_REGRESSION_MODEL_PATH
    raise ValueError(f"[AI] Unknown model name: '{model_name}'")


def load_production_model(model_name: str) -> ProductionModel:
    """Load the artifact for the specified model from disk.

    Args:
        model_name: One of ALLOWED_MODELS.

    Returns:
        Loaded model instance (XGBClassifier or LinearRegression).

    Raises:
        FileNotFoundError: If the artifact does not exist on disk.
        ValueError:        If model_name is not in ALLOWED_MODELS.
    """
    artifact_path = get_artifact_path(model_name)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"[AI] Model artifact not found: '{artifact_path}'. "
            f"Train the '{model_name}' model first and ensure the artifact is present."
        )

    logger.info("[AI] Loading %s artifact from: %s", model_name, artifact_path)
    model = joblib.load(artifact_path)
    logger.info("[AI] %s model loaded successfully.", model_name)
    return model


def validate_production_model(model: ProductionModel, model_name: str) -> None:
    """Sanity-check that the loaded model can perform inference.

    Runs a single dummy prediction to catch corrupt artifacts early at startup
    rather than on the first real request.

    Args:
        model:      Loaded model instance.
        model_name: Label used in log/error messages.

    Raises:
        RuntimeError: If the model cannot predict on the expected 8-feature input.
    """
    import numpy as np

    # 8 features: supplier_price, lead_time_days, claim_rate, on_time_rate,
    #             order_frequency, price_per_day, risk_composite, frequency_reliability
    dummy = np.array([[300000, 5, 0.03, 0.90, 12, 60000, 0.13, 10.8]])

    try:
        if model_name == MODEL_XGBOOST:
            model.predict_proba(dummy)
        else:
            # Linear Regression: predict() + threshold
            model.predict(dummy)
    except Exception as exc:
        raise RuntimeError(
            f"[AI] Artifact validation failed for '{model_name}': {exc}. "
            "The artifact may be corrupt or incompatible. Retrain the model."
        ) from exc


# ── Module-level production model cache ───────────────────────────────────────
# Loaded once at startup (via validate_and_load_production_model) and cached
# for the lifetime of the process. Invalidated explicitly after retraining.

_production_model: ProductionModel | None = None
_production_model_name: str | None = None


def validate_and_load_production_model() -> tuple[ProductionModel, str]:
    """Full startup sequence: read config → validate → load artifact → smoke-test.

    Called once from main.py lifespan. Populates the module-level cache so
    subsequent calls to get_production_model() are instant (no disk I/O).

    Returns:
        (model, model_name) tuple.

    Raises:
        ValueError:        Bad PURCHASING_AI_MODEL value.
        FileNotFoundError: Artifact file missing.
        RuntimeError:      Artifact loaded but prediction smoke-test failed.
    """
    global _production_model, _production_model_name

    model_name    = get_selected_model_name()
    artifact_path = get_artifact_path(model_name)
    display_name  = (
        "XGBoost" if model_name == MODEL_XGBOOST else "Linear Regression"
    )

    logger.info("")
    logger.info("AI MODEL CONFIGURATION")
    logger.info("----------------------")
    logger.info("Production Model : %s", display_name)
    logger.info("Model Artifact   : %s", artifact_path)

    model = load_production_model(model_name)
    validate_production_model(model, model_name)

    _production_model      = model
    _production_model_name = model_name

    logger.info("Status           : Loaded Successfully")
    logger.info("")

    return model, model_name


def get_production_model() -> ProductionModel:
    """Return the cached production model.

    Falls back to loading from disk if the cache is empty (e.g. first call
    before startup validation ran, or after invalidate_production_model_cache()).

    Returns:
        Active production model instance.
    """
    global _production_model, _production_model_name

    if _production_model is None:
        model_name = get_selected_model_name()
        _production_model_name = model_name
        _production_model = load_production_model(model_name)
        validate_production_model(_production_model, model_name)

    return _production_model


def get_active_model_name() -> str:
    """Return the currently active production model name (for logging)."""
    if _production_model_name is not None:
        return _production_model_name
    return get_selected_model_name()


def invalidate_production_model_cache() -> None:
    """Clear the production model cache.

    Call this after retraining so the next prediction reloads from disk.
    """
    global _production_model, _production_model_name
    _production_model      = None
    _production_model_name = None
    logger.info("[AI] Production model cache invalidated — will reload on next prediction.")
