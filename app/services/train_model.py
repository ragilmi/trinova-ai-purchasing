"""
train_model.py
Trains an XGBoost classifier on the supplier dataset and persists the model.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.services.preprocess import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    load_dataset,
    preprocess_dataframe,
)

logger = logging.getLogger(__name__)

# Default path where the trained model artifact is saved
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_model.pkl"

# Columns the caller must supply when sending raw rows
REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all 6 evaluation metrics for one split.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Hard predictions (0/1).
        y_prob: Predicted probabilities for class 1.

    Returns:
        Dict with keys: log_loss, mse, mae, r2, accuracy, auc_roc
    """
    return {
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "mse":      round(float(mean_squared_error(y_true, y_pred)), 4),
        "mae":      round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":       round(float(r2_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "auc_roc":  round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def train(
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    append_to_existing: bool = False,
    model_output_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train the XGBoost supplier-risk model.

    Data source priority:
      1. ``dataframe``        — caller passes a DataFrame directly (live data from backend)
      2. ``csv_path``         — path to a CSV on disk
      3. bundled dataset      — app/datasets/supplier_training.csv (fallback)

    When ``append_to_existing`` is True and a ``dataframe`` is supplied, the new
    rows are merged with the bundled dataset before training, so historical data
    is never discarded.

    Args:
        csv_path:            Path to a training CSV. Defaults to bundled dataset.
        dataframe:           Pre-built DataFrame of new training rows from the ERP backend.
        append_to_existing:  When True, merge ``dataframe`` with the bundled dataset.
        model_output_path:   Where to save the .pkl artifact. Defaults to MODEL_PATH.
        test_size:           Fraction of data held out for evaluation.
        random_state:        Seed for reproducibility.

    Returns:
        Dict with keys:
            samples_trained, samples_tested, model_path, data_source,
            train_metrics: { log_loss, mse, mae, r2, accuracy, auc_roc },
            test_metrics:  { log_loss, mse, mae, r2, accuracy, auc_roc },
    """
    model_output_path = Path(model_output_path or MODEL_PATH)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve the training DataFrame
    # ------------------------------------------------------------------
    if dataframe is not None:
        # Normalize column names before validation
        dataframe = dataframe.copy()
        dataframe.columns = [c.strip().lower() for c in dataframe.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
        if missing:
            raise ValueError(
                f"Supplied dataframe is missing required columns: {missing}. "
                f"Required: {REQUIRED_COLUMNS}"
            )

        if append_to_existing:
            bundled = load_dataset()
            df = pd.concat([bundled, dataframe], ignore_index=True)
            data_source = f"live_data ({len(dataframe)} new rows) + bundled ({len(bundled)} rows)"
        else:
            df = dataframe.copy()
            data_source = f"live_data ({len(df)} rows)"

        logger.info("Using live dataframe: %s", data_source)
    else:
        logger.info("Loading dataset from disk...")
        df = load_dataset(csv_path)
        data_source = str(csv_path) if csv_path else "bundled_dataset"
        logger.info("Dataset loaded: %d rows from %s", len(df), data_source)

    X, y = preprocess_dataframe(df)

    if y is None:
        raise ValueError("Training dataset must contain the 'late_delivery' label column.")

    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} rows. Need at least 10 samples to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Split — train: %d rows, test: %d rows", len(X_train), len(X_test))

    model = XGBClassifier(
        # Tree complexity — kept shallow to prevent memorizing small datasets
        n_estimators=500,       # high ceiling; early stopping will cut this down
        max_depth=3,            # reduced from 4 — shallower trees generalize better
        min_child_weight=3,     # require at least 3 samples in a leaf
        # Learning rate
        learning_rate=0.05,
        # Subsampling — adds randomness, reduces overfitting
        subsample=0.7,          # reduced from 0.8
        colsample_bytree=0.7,   # reduced from 0.8
        colsample_bylevel=0.7,  # additional column sampling per level
        # Regularization
        reg_alpha=0.5,          # L1 — drives less-important feature weights to zero
        reg_lambda=2.0,         # L2 — penalizes large weights
        gamma=0.1,              # minimum loss reduction to make a split
        # Misc
        eval_metric="logloss",
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=20,   # stop if test log-loss doesn't improve for 20 rounds
    )

    # ------------------------------------------------------------------
    # Evaluate on training set (detects overfitting when compared to test)
    # ------------------------------------------------------------------
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = _compute_metrics(y_train, train_pred, train_prob)

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_pred, test_prob)

    logger.info(
        "Train — log_loss=%.4f, mse=%.4f, mae=%.4f, r2=%.4f, accuracy=%.4f, auc_roc=%.4f",
        *train_metrics.values(),
    )
    logger.info(
        "Test  — log_loss=%.4f, mse=%.4f, mae=%.4f, r2=%.4f, accuracy=%.4f, auc_roc=%.4f",
        *test_metrics.values(),
    )

    joblib.dump(model, model_output_path)
    best_round = getattr(model, "best_iteration", model.n_estimators)
    logger.info("Model saved to %s (best round: %d)", model_output_path, best_round)

    return {
        "samples_trained": len(X_train),
        "samples_tested":  len(X_test),
        "model_path":      str(model_output_path),
        "data_source":     data_source,
        "best_round":      best_round,
        "train_metrics":   train_metrics,
        "test_metrics":    test_metrics,
    }


def load_model(model_path: str | Path | None = None) -> XGBClassifier:
    """Load a previously trained model from disk.

    Args:
        model_path: Path to the .pkl file. Defaults to MODEL_PATH.

    Returns:
        Loaded XGBClassifier instance.

    Raises:
        FileNotFoundError: If no model artifact exists at the given path.
    """
    model_path = Path(model_path or MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Call POST /train to train the model first."
        )

    return joblib.load(model_path)
