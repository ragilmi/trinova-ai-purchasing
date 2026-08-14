"""
model_comparison.py
Handles training, evaluation, and comparison between Linear Regression (academic baseline),
Random Forest Classifier, and XGBoost Classifier for supplier late-delivery risk prediction.

Linear Regression Baseline:
  - Produces continuous regression scores: y_score = model.predict(X_test)
  - Converted to binary decisions using a fixed threshold of 0.50: y_pred = (y_score >= 0.50).astype(int)
  - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are evaluated on thresholded y_pred.
  - ROC-AUC is evaluated directly on continuous y_score.
  - Log Loss is marked as N/A (Linear Regression outputs are continuous scores, not calibrated probabilities).

Ensemble Classifiers (Random Forest & XGBoost):
  - Predict hard binary classes via model.predict(X_test) for Accuracy, Precision, Recall, F1.
  - Predict continuous probabilities via model.predict_proba(X_test)[:, 1] for ROC-AUC and Log Loss.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when running as a standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def evaluate_classifier_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Calculate classification metrics for probabilistic classifiers (Random Forest, XGBoost)."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(loss, 4),
        "log_loss_str": f"{loss:8.4f}",
        "_raw_acc": acc,
        "_raw_prec": prec,
        "_raw_rec": rec,
        "_raw_f1": f1,
        "_raw_auc": auc,
        "_raw_loss": loss,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_linear_regression_baseline(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Evaluate Linear Regression as an academic regression baseline.

    - y_score contains continuous regression predictions (probability-like scores, not calibrated probabilities).
    - Binary decision rule: y_pred = 1 if y_score >= threshold else 0.
    - Accuracy, Precision, Recall, F1, and Confusion Matrix are evaluated on y_pred.
    - ROC-AUC is evaluated on continuous y_score.
    - Log Loss is marked N/A.
    """
    y_pred = (y_score >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_score))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "log_loss": None,
        "log_loss_str": "     N/A",
        "_raw_acc": acc,
        "_raw_prec": prec,
        "_raw_rec": rec,
        "_raw_f1": f1,
        "_raw_auc": auc,
        "_raw_loss": float("inf"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def run_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    xgb_model: XGBClassifier | None = None,
    dataset_info: dict | None = None,
) -> dict:
    """Train Linear Regression Baseline, Random Forest, and XGBoost Classifier on X_train/y_train,
    evaluate on X_test/y_test, and display a model comparison report.

    Args:
        X_train: Feature matrix for training.
        y_train: Binary target labels for training.
        X_test: Feature matrix for testing.
        y_test: Binary target labels for testing.
        xgb_model: Optional pre-fitted XGBoost model. If None, a new XGBClassifier will be trained.
        dataset_info: Optional dictionary containing dataset summary info.

    Returns:
        Dict containing comparison metrics and best model selection.
    """
    dataset_info = dataset_info or {}
    dataset_name = dataset_info.get("dataset_name", "supplier_training.csv")
    total_products = dataset_info.get("total_suppliers") or dataset_info.get("total_products", len(X_train) + len(X_test))
    products_evaluated = dataset_info.get("suppliers_evaluated") or dataset_info.get("products_evaluated", len(X_test))
    evaluation_method = dataset_info.get("evaluation_method", "Time-based Hold-out 80/20")
    training_window = dataset_info.get("training_window", f"80% historical data ({len(X_train)} samples)")
    testing_window = dataset_info.get("testing_window", f"20% test split ({len(X_test)} samples)")

    models_dict = {}

    # ------------------------------------------------------------------
    # 1. Train & Evaluate Linear Regression Academic Baseline
    # ------------------------------------------------------------------
    lr_status = "Training completed successfully."
    try:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_score_lr = lr_model.predict(X_test)
        lr_metrics = evaluate_linear_regression_baseline(y_test, y_score_lr, threshold=0.50)
    except Exception as exc:
        logger.error("Linear Regression baseline training/evaluation failed: %s", exc)
        lr_status = f"Failed: {exc}"
        lr_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc_roc": 0.0, "log_loss": None, "log_loss_str": "     N/A",
            "_raw_acc": 0.0, "_raw_prec": 0.0, "_raw_rec": 0.0, "_raw_f1": 0.0, "_raw_auc": 0.0, "_raw_loss": float("inf"),
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "threshold": 0.50,
        }
    models_dict["Linear Regression"] = lr_metrics

    # ------------------------------------------------------------------
    # 2. Train & Evaluate Random Forest Classifier
    # ------------------------------------------------------------------
    rf_status = "Training completed successfully."
    try:
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_classifier_metrics(y_test, y_pred_rf, y_prob_rf)
    except Exception as exc:
        logger.error("Random Forest training/evaluation failed: %s", exc)
        rf_status = f"Failed: {exc}"
        rf_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc_roc": 0.0, "log_loss": 99.0, "log_loss_str": " 99.0000",
            "_raw_acc": 0.0, "_raw_prec": 0.0, "_raw_rec": 0.0, "_raw_f1": 0.0, "_raw_auc": 0.0, "_raw_loss": 99.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    models_dict["Random Forest"] = rf_metrics

    # ------------------------------------------------------------------
    # 3. Train & Evaluate XGBoost Classifier
    # ------------------------------------------------------------------
    xgb_status = "Training completed successfully."
    try:
        if xgb_model is None:
            xgb_model = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                min_child_weight=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=20,
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_classifier_metrics(y_test, y_pred_xgb, y_prob_xgb)
    except Exception as exc:
        logger.error("XGBoost training/evaluation failed: %s", exc)
        xgb_status = f"Failed: {exc}"
        xgb_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc_roc": 0.0, "log_loss": 99.0, "log_loss_str": " 99.0000",
            "_raw_acc": 0.0, "_raw_prec": 0.0, "_raw_rec": 0.0, "_raw_f1": 0.0, "_raw_auc": 0.0, "_raw_loss": 99.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    models_dict["XGBoost Classifier"] = xgb_metrics

    # ------------------------------------------------------------------
    # 4. Unrounded Full-Precision Winner Determination & Metric Highlights
    # ------------------------------------------------------------------
    candidates = ["Linear Regression", "Random Forest", "XGBoost Classifier"]

    best_acc_model = max(candidates, key=lambda m: models_dict[m]["_raw_acc"])
    best_acc_val = models_dict[best_acc_model]["accuracy"]

    best_prec_model = max(candidates, key=lambda m: models_dict[m]["_raw_prec"])
    best_prec_val = models_dict[best_prec_model]["precision"]

    best_rec_model = max(candidates, key=lambda m: models_dict[m]["_raw_rec"])
    best_rec_val = models_dict[best_rec_model]["recall"]

    best_f1_model = max(candidates, key=lambda m: models_dict[m]["_raw_f1"])
    best_f1_val = models_dict[best_f1_model]["f1_score"]

    best_auc_model = max(candidates, key=lambda m: models_dict[m]["_raw_auc"])
    best_auc_val = models_dict[best_auc_model]["auc_roc"]

    # Log Loss winner evaluated only among probabilistic classifiers
    prob_candidates = ["Random Forest", "XGBoost Classifier"]
    best_loss_model = min(prob_candidates, key=lambda m: models_dict[m]["_raw_loss"])
    best_loss_val = models_dict[best_loss_model]["log_loss"]

    # Overall Best Model based on Highest F1
    overall_best_model = best_f1_model
    overall_best_f1 = best_f1_val

    # Improvements vs Linear Regression Baseline (calculated using unrounded values)
    f1_diff_rf = rf_metrics["_raw_f1"] - lr_metrics["_raw_f1"]
    f1_diff_xgb = xgb_metrics["_raw_f1"] - lr_metrics["_raw_f1"]

    acc_diff_rf = rf_metrics["_raw_acc"] - lr_metrics["_raw_acc"]
    acc_diff_xgb = xgb_metrics["_raw_acc"] - lr_metrics["_raw_acc"]

    auc_diff_rf = rf_metrics["_raw_auc"] - lr_metrics["_raw_auc"]
    auc_diff_xgb = xgb_metrics["_raw_auc"] - lr_metrics["_raw_auc"]

    # ------------------------------------------------------------------
    # 5. Generate & Output Terminal Report
    # ------------------------------------------------------------------
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    report = f"""
====================================================================================================
TRINOVA PURCHASING AI SUPPLIER
LATE-DELIVERY MODEL COMPARISON
====================================================================================================

DATASET INFORMATION
----------------------------------------------------------------------------------------------------
Dataset                  : {dataset_name}
Total Records            : {total_products}
Training Samples         : {training_window}
Testing Samples          : {testing_window}
Validation               : {evaluation_method}
Target                   : late_delivery
Class 0                  : On Time
Class 1                  : Late Delivery

====================================================================================================
MODEL TRAINING
====================================================================================================

[1/3] Training Linear Regression Baseline...
      {lr_status}

[2/3] Training Random Forest Classifier...
      {rf_status}

[3/3] Training XGBoost Classifier...
      {xgb_status}

====================================================================================================
LINEAR REGRESSION BASELINE
====================================================================================================

Model Output             : Continuous regression score
Classification Threshold : 0.50

Classification Rule:
    score >= 0.50 -> Late Delivery (1)
    score <  0.50 -> On Time (0)

IMPORTANT:
Linear Regression outputs are continuous scores, not calibrated probabilities.

====================================================================================================
MODEL PERFORMANCE
====================================================================================================

Model                  Accuracy  Precision  Recall   F1 Score   ROC-AUC  Log Loss
----------------------------------------------------------------------------------------------------
Linear Regression        {lr_metrics['accuracy']:8.4f}   {lr_metrics['precision']:8.4f}  {lr_metrics['recall']:8.4f}   {lr_metrics['f1_score']:8.4f}   {lr_metrics['auc_roc']:8.4f}  {lr_metrics['log_loss_str']}
Random Forest            {rf_metrics['accuracy']:8.4f}   {rf_metrics['precision']:8.4f}  {rf_metrics['recall']:8.4f}   {rf_metrics['f1_score']:8.4f}   {rf_metrics['auc_roc']:8.4f}  {rf_metrics['log_loss_str']}
XGBoost Classifier       {xgb_metrics['accuracy']:8.4f}   {xgb_metrics['precision']:8.4f}  {xgb_metrics['recall']:8.4f}   {xgb_metrics['f1_score']:8.4f}   {xgb_metrics['auc_roc']:8.4f}  {xgb_metrics['log_loss_str']}
----------------------------------------------------------------------------------------------------

CONFUSION MATRICES
----------------------------------------------------------------------------------------------------
Linear Regression     : TN={lr_metrics['tn']}, FP={lr_metrics['fp']} | FN={lr_metrics['fn']}, TP={lr_metrics['tp']} (threshold=0.50)
Random Forest         : TN={rf_metrics['tn']}, FP={rf_metrics['fp']} | FN={rf_metrics['fn']}, TP={rf_metrics['tp']}
XGBoost Classifier      : TN={xgb_metrics['tn']}, FP={xgb_metrics['fp']} | FN={xgb_metrics['fn']}, TP={xgb_metrics['tp']}

====================================================================================================
BEST RESULTS
====================================================================================================

Best Accuracy           : {best_acc_model} ({best_acc_val:.4f})
Best Precision          : {best_prec_model} ({best_prec_val:.4f})
Best Recall             : {best_rec_model} ({best_rec_val:.4f})
Best F1 Score           : {best_f1_model} ({best_f1_val:.4f})
Best ROC-AUC            : {best_auc_model} ({best_auc_val:.4f})
Best Log Loss           : {best_loss_model} ({best_loss_val:.4f})

Overall Best Model
(highest F1)            : {overall_best_model} ({overall_best_f1:.4f})

====================================================================================================
MODEL COMPARISON (vs Linear Regression Baseline)
====================================================================================================

F1 Score Improvement
  - Random Forest       : {f1_diff_rf:+.4f}
  - XGBoost Classifier  : {f1_diff_xgb:+.4f}

Accuracy Improvement
  - Random Forest       : {acc_diff_rf:+.4f}
  - XGBoost Classifier  : {acc_diff_xgb:+.4f}

ROC-AUC Improvement
  - Random Forest       : {auc_diff_rf:+.4f}
  - XGBoost Classifier  : {auc_diff_xgb:+.4f}

====================================================================================================
TRAINING COMPLETED
====================================================================================================
"""
    # Print report to stdout for terminal view
    print(report, flush=True)
    logger.info("3-Model comparison complete. Overall Best Model: %s (F1: %.4f)", overall_best_model, overall_best_f1)

    return {
        "linear_regression": lr_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "best_model": overall_best_model,
        "best_f1": overall_best_f1,
        "best_accuracy": best_acc_val,
        "best_auc_roc": best_auc_val,
        "best_accuracy_model": best_acc_model,
        "best_precision_model": best_prec_model,
        "best_recall_model": best_rec_model,
        "best_f1_model": best_f1_model,
        "best_auc_model": best_auc_model,
        "best_loss_model": best_loss_model,
        "f1_improvement_rf": round(f1_diff_rf, 4),
        "f1_improvement_xgb": round(f1_diff_xgb, 4),
        "accuracy_improvement_rf": round(acc_diff_rf, 4),
        "accuracy_improvement_xgb": round(acc_diff_xgb, 4),
        "auc_improvement_rf": round(auc_diff_rf, 4),
        "auc_improvement_xgb": round(auc_diff_xgb, 4),
    }
