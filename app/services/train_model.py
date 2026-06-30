"""
train_model.py
Trains an XGBoost classifier on the supplier dataset and persists the model.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

from app.services.preprocess import load_dataset, preprocess_dataframe, FEATURE_COLUMNS, LABEL_COLUMN

logger = logging.getLogger(__name__)

# Default path where the trained model artifact is saved
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_model.pkl"

# Columns the caller must supply when sending raw rows
REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]


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
        Dict with training metrics:
            {
                "accuracy": float,
                "roc_auc": float,
                "samples_trained": int,
                "samples_tested": int,
                "model_path": str,
                "classification_report": str,
                "data_source": str,
            }
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

    logger.info(
        "Split — train: %d rows, test: %d rows", len(X_train), len(X_test)
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    report = classification_report(y_test, y_pred, target_names=["on_time", "late"])

    logger.info("Accuracy : %.4f", accuracy)
    logger.info("ROC-AUC  : %.4f", roc_auc)
    logger.info("Report:\n%s", report)

    joblib.dump(model, model_output_path)
    logger.info("Model saved to %s", model_output_path)

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "samples_trained": len(X_train),
        "samples_tested": len(X_test),
        "model_path": str(model_output_path),
        "classification_report": report,
        "data_source": data_source,
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
