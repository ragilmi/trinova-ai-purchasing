"""
train_model.py
Trains an XGBoost classifier on the supplier dataset and persists the model.

Split strategy (chronological / temporal):
  - Data is sorted by `transaction_date` when that column exists, or kept in
    original row order (which is also chronological for this dataset).
  - The first 80 % of rows become the training set; the last 20 % become the
    held-out test set.  This mirrors real procurement conditions where the
    model learns from the past and is evaluated on future transactions.
  - Within the training set, TimeSeriesSplit (5 folds) is used for cross-
    validation so that validation windows always follow training windows —
    no future data ever leaks into earlier folds.

   train_test_split(shuffle=True)  — NOT used (would leak future into past)
  time-based 80/20 slice          — chronologically honest
  TimeSeriesSplit CV on train set — forward-only fold expansion
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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from app.services.preprocess import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    load_dataset,
    normalize_columns,
    preprocess_dataframe,
)

logger = logging.getLogger(__name__)

# Default path where the trained model artifact is saved
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_model.pkl"

# Columns the caller must supply when sending raw rows
REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

# Number of folds used for TimeSeriesSplit cross-validation on the training set
CV_N_SPLITS: int = 5


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


def _time_based_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = "transaction_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / test sets using temporal ordering.

    If ``date_col`` exists, the DataFrame is sorted by that column first.
    Otherwise the current row order is used as-is (this dataset is already
    in chronological order by supplier_id / procurement date).

    The split is a simple slice — first (1 - test_size) rows go to train,
    the remaining rows go to test.  No shuffling is ever applied.

    Args:
        df:        Full dataset DataFrame.
        test_size: Fraction reserved for the test set (default 0.20).
        date_col:  Optional column name to sort by before splitting.

    Returns:
        (train_df, test_df) — both are DataFrame slices, index preserved.
    """
    if date_col in df.columns:
        logger.info("Sorting dataset by '%s' for time-based split.", date_col)
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        logger.info(
            "Column '%s' not found — using existing row order as temporal proxy.",
            date_col,
        )

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return train_df, test_df


def _run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    n_splits: int = CV_N_SPLITS,
) -> dict:
    """Run TimeSeriesSplit cross-validation on the training set.

    Each fold trains a fresh XGBClassifier with the same hyper-parameters
    used for the final model.  The fold windows expand forward in time:

        Fold 1: train[0..t1]  → validate[t1..t2]
        Fold 2: train[0..t2]  → validate[t2..t3]
        ...

    This guarantees that the validation split never sees data that would
    have been in the future relative to the training window.

    Args:
        X_train:      Feature matrix (training portion only).
        y_train:      Label series (training portion only).
        model_params: XGBClassifier keyword arguments.
        n_splits:     Number of CV folds (default 5).

    Returns:
        Dict with per-fold metric lists and their averages:
          fold_metrics: list of dicts (one per fold)
          avg_accuracy, avg_auc_roc, avg_log_loss,
          avg_mse, avg_mae, avg_r2
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []

    logger.info("Starting TimeSeriesSplit CV — %d folds on %d training samples.", n_splits, len(X_train))

    for fold_num, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        fold_model = XGBClassifier(**model_params)
        fold_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=20,
        )

        val_pred = fold_model.predict(X_val)
        val_prob = fold_model.predict_proba(X_val)[:, 1]
        metrics  = _compute_metrics(y_val, val_pred, val_prob)
        fold_metrics.append({"fold": fold_num, **metrics})

        logger.info(
            "  Fold %d/%d — accuracy=%.4f  auc_roc=%.4f  log_loss=%.4f  "
            "train_size=%d  val_size=%d",
            fold_num, n_splits,
            metrics["accuracy"], metrics["auc_roc"], metrics["log_loss"],
            len(X_tr), len(X_val),
        )

    # Aggregate across folds
    keys = ("log_loss", "mse", "mae", "r2", "accuracy", "auc_roc")
    averages = {
        f"avg_{k}": round(float(np.mean([f[k] for f in fold_metrics])), 4)
        for k in keys
    }

    logger.info(
        "CV summary — avg_accuracy=%.4f  avg_auc_roc=%.4f  avg_log_loss=%.4f",
        averages["avg_accuracy"],
        averages["avg_auc_roc"],
        averages["avg_log_loss"],
    )

    return {"fold_metrics": fold_metrics, **averages}


def train(
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    append_to_existing: bool = False,
    model_output_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_splits: int = CV_N_SPLITS,
) -> dict:
    """Train the XGBoost supplier-risk model.

    Split strategy
    --------------
    This function uses a **time-based split** instead of a random split to
    respect the chronological nature of procurement data:

    1. The full dataset is sorted by ``transaction_date`` (if present) or
       kept in existing row order.
    2. The first 80 % of rows form the **training set**.
    3. The remaining 20 % form the **held-out test set**.
    4. **TimeSeriesSplit cross-validation** (``n_cv_splits`` folds) is then
       run *on the training set only* to tune and validate the model without
       any future-data leakage.
    5. A final model is fit on the full training set and evaluated on the
       test set.

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
        test_size:           Fraction of data held out for evaluation (time-based).
        random_state:        Seed for reproducibility.
        n_cv_splits:         Number of TimeSeriesSplit folds (default 5).

    Returns:
        Dict with keys:
            samples_trained, samples_tested, model_path, data_source, best_round,
            split_method,
            train_metrics:  { log_loss, mse, mae, r2, accuracy, auc_roc },
            test_metrics:   { log_loss, mse, mae, r2, accuracy, auc_roc },
            cv_results:     {
                fold_metrics: [ {fold, log_loss, mse, mae, r2, accuracy, auc_roc}, ... ],
                avg_log_loss, avg_mse, avg_mae, avg_r2, avg_accuracy, avg_auc_roc
            }
    """
    model_output_path = Path(model_output_path or MODEL_PATH)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve the training DataFrame
    # ------------------------------------------------------------------
    if dataframe is not None:
        dataframe = normalize_columns(dataframe)

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

    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} rows. Need at least 10 samples to train."
        )

    # ------------------------------------------------------------------
    # Time-based 80/20 split  (no shuffling — chronological order)
    # ------------------------------------------------------------------
    train_df, test_df = _time_based_split(df, test_size=test_size)
    logger.info(
        "Time-based split — train: %d rows (%.0f%%), test: %d rows (%.0f%%)",
        len(train_df), (1 - test_size) * 100,
        len(test_df),  test_size * 100,
    )

    X_train, y_train = preprocess_dataframe(train_df)
    X_test,  y_test  = preprocess_dataframe(test_df)

    if y_train is None or y_test is None:
        raise ValueError("Training dataset must contain the 'late_delivery' label column.")

    # ------------------------------------------------------------------
    # XGBoost hyper-parameters (shared between CV folds and final model)
    # ------------------------------------------------------------------
    model_params = dict(
        n_estimators=500,
        max_depth=3,
        min_child_weight=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        eval_metric="logloss",
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # TimeSeriesSplit cross-validation on the TRAINING SET only
    # ------------------------------------------------------------------
    min_samples_for_cv = n_cv_splits + 1
    if len(X_train) >= min_samples_for_cv:
        cv_results = _run_cv(X_train, y_train, model_params, n_splits=n_cv_splits)
    else:
        logger.warning(
            "Training set too small for %d-fold CV (%d rows). Skipping CV.",
            n_cv_splits, len(X_train),
        )
        cv_results = {
            "fold_metrics": [],
            "avg_log_loss": None,
            "avg_mse":      None,
            "avg_mae":      None,
            "avg_r2":       None,
            "avg_accuracy": None,
            "avg_auc_roc":  None,
        }

    # ------------------------------------------------------------------
    # Final model — fit on the full training set
    # ------------------------------------------------------------------
    model = XGBClassifier(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=20,
    )

    # ------------------------------------------------------------------
    # Evaluate on training set (detects overfitting when compared to test)
    # ------------------------------------------------------------------
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = _compute_metrics(y_train, train_pred, train_prob)

    # ------------------------------------------------------------------
    # Evaluate on held-out test set (the most-recent 20 % of transactions)
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
        "split_method":    "time_based_80_20",
        "train_metrics":   train_metrics,
        "test_metrics":    test_metrics,
        "cv_results":      cv_results,
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
