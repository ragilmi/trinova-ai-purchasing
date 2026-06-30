import io
import logging
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from app.services.predict import invalidate_model_cache, predict_supplier_risk
from app.services.train_model import REQUIRED_COLUMNS, train
from app.services.preprocess import normalize_columns
from app.services.ahp_topsis import (
    DEFAULT_AHP_MATRIX,
    CRITERION_NAMES,
    rank_suppliers,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

class SupplierInput(BaseModel):
    supplier_id: int | None = Field(
        default=None,
        description="Optional supplier ID — echoed back in the response for traceability.",
        examples=[12],
    )
    supplier_price: float = Field(
        ...,
        gt=0,
        description="Unit price or contract value in local currency.",
        examples=[500000],
    )
    lead_time_days: int = Field(
        ...,
        ge=1,
        description="Agreed lead time in calendar days.",
        examples=[3],
    )
    claim_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical claim / defect rate (0 = no claims, 1 = all orders claimed).",
        examples=[0.02],
    )
    on_time_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical on-time delivery rate (0 = never on time, 1 = always on time).",
        examples=[0.95],
    )
    order_frequency: int = Field(
        ...,
        ge=1,
        description="Number of orders placed with this supplier per year.",
        examples=[24],
    )

    @field_validator("claim_rate", "on_time_rate")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Rate must be between 0.0 and 1.0")
        return v


class SupplierRiskResponse(BaseModel):
    supplier_id: int | None = None
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    delay_probability: float = Field(
        description="Predicted probability of a late delivery (0.0 – 1.0)."
    )
    late_probability: int = Field(
        description="Same as delay_probability expressed as a percentage (0 – 100)."
    )


class TrainRequest(BaseModel):
    csv_path: str | None = Field(
        default=None,
        description=(
            "Absolute or relative path to a custom training CSV on the server. "
            "Leave empty to use the bundled dataset."
        ),
    )


class TrainFromRowsRequest(BaseModel):
    """Send new ERP transaction rows directly as JSON for retraining."""

    rows: list[dict] = Field(
        ...,
        description=(
            "List of training records exported from the ERP database. "
            f"Each record must contain: {REQUIRED_COLUMNS}."
        ),
        examples=[
            [
                {
                    "supplier_price": 320000,
                    "lead_time_days": 5,
                    "claim_rate": 0.05,
                    "on_time_rate": 0.88,
                    "order_frequency": 12,
                    "late_delivery": 1,
                }
            ]
        ],
    )
    append_to_existing: bool = Field(
        default=True,
        description=(
            "When True (default), the new rows are merged with the bundled dataset "
            "before training so historical data is preserved. "
            "Set to False to train only on the supplied rows."
        ),
    )


class SplitMetrics(BaseModel):
    """Evaluation metrics for one data split (train or test)."""
    log_loss: float = Field(description="Log-Loss (cross-entropy). Lower is better.")
    mse:      float = Field(description="Mean Squared Error. Lower is better.")
    mae:      float = Field(description="Mean Absolute Error. Lower is better.")
    r2:       float = Field(description="R² coefficient of determination. Closer to 1 is better.")
    accuracy: float = Field(description="Classification accuracy (0–1).")
    auc_roc:  float = Field(description="Area Under the ROC Curve (0–1). Higher is better.")


class FoldMetrics(BaseModel):
    """Evaluation metrics for a single TimeSeriesSplit CV fold."""
    fold:     int
    log_loss: float
    mse:      float
    mae:      float
    r2:       float
    accuracy: float
    auc_roc:  float


class CVResults(BaseModel):
    """Aggregated results from TimeSeriesSplit cross-validation on the training set.

    Cross-validation is performed *only* on the training set (first 80 % of
    data in chronological order).  Each fold expands forward in time so that
    validation rows always come after training rows — preventing any future
    data leakage.
    """
    fold_metrics:  list[FoldMetrics] = Field(
        description="Per-fold validation metrics."
    )
    avg_log_loss:  float | None = Field(None, description="Mean log-loss across folds.")
    avg_mse:       float | None = Field(None, description="Mean MSE across folds.")
    avg_mae:       float | None = Field(None, description="Mean MAE across folds.")
    avg_r2:        float | None = Field(None, description="Mean R² across folds.")
    avg_accuracy:  float | None = Field(None, description="Mean accuracy across folds.")
    avg_auc_roc:   float | None = Field(None, description="Mean AUC-ROC across folds.")


class TrainResponse(BaseModel):
    message:         str
    split_method:    str = Field(
        description=(
            "Splitting strategy used. 'time_based_80_20' means the first 80 % of "
            "rows (chronologically) were used for training and the last 20 % for "
            "testing — no random shuffling applied."
        )
    )
    samples_trained: int
    samples_tested:  int
    best_round:      int = Field(description="Number of trees used after early stopping.")
    model_path:      str
    data_source:     str
    train_metrics:   SplitMetrics
    test_metrics:    SplitMetrics
    cv_results:      CVResults = Field(
        description=(
            "TimeSeriesSplit cross-validation results computed on the training set. "
            "Use avg_accuracy / avg_auc_roc to compare model iterations without "
            "touching the held-out test set."
        )
    )

@router.post(
    "/supplier-risk",
    response_model=SupplierRiskResponse,
    summary="Predict supplier delay risk",
    description=(
        "Accepts supplier performance metrics and returns a predicted risk level "
        "(LOW / MEDIUM / HIGH) together with the probability of a late delivery."
    ),
)
def predict_supplier_risk_endpoint(payload: SupplierInput) -> SupplierRiskResponse:
    """POST /predict/supplier-risk"""
    try:
        result = predict_supplier_risk(payload.model_dump(exclude={"supplier_id"}))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction.",
        ) from exc

    return SupplierRiskResponse(supplier_id=payload.supplier_id, **result)

# ── Batch predict models ──────────────────────────────────────────────────────

class BatchSupplierInput(BaseModel):
    """One supplier entry for batch prediction."""
    supplier_id:     int | None = Field(default=None, examples=[1])
    supplier_name:   str | None = Field(default=None, examples=["PT Supplier A"])
    supplier_price:  float      = Field(..., gt=0,    examples=[500000])
    lead_time_days:  int        = Field(..., ge=1,    examples=[3])
    claim_rate:      float      = Field(..., ge=0.0, le=1.0, examples=[0.02])
    on_time_rate:    float      = Field(..., ge=0.0, le=1.0, examples=[0.95])
    order_frequency: int        = Field(..., ge=1,    examples=[24])

    @field_validator("claim_rate", "on_time_rate")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Rate must be between 0.0 and 1.0")
        return v


class BatchPredictRequest(BaseModel):
    """List of suppliers to score in one call."""
    suppliers: list[BatchSupplierInput] = Field(
        ...,
        description="One or more supplier entries to run through the XGBoost model.",
    )


class SupplierPredictResult(BaseModel):
    """ML result for a single supplier — returned before any ranking."""
    supplier_id:       int | None
    supplier_name:     str | None
    supplier_price:    float
    lead_time_days:    int
    claim_rate:        float
    on_time_rate:      float
    order_frequency:   int
    risk_level:        Literal["LOW", "MEDIUM", "HIGH"]
    delay_probability: float
    late_probability:  int


class BatchPredictResponse(BaseModel):
    """Raw ML results for all suppliers — no ranking applied yet."""
    total:   int
    results: list[SupplierPredictResult]


# ── AHP-TOPSIS rank models ────────────────────────────────────────────────────

class AhpMatrixRequest(BaseModel):
    """
    Optional custom 5×5 AHP pairwise comparison matrix.
    Row / column order: delay_probability, on_time_rate, claim_rate,
                        supplier_price, order_frequency.
    Leave empty to use the default matrix.
    """
    matrix: list[list[float]] | None = Field(
        default=None,
        description=(
            "5×5 positive reciprocal matrix. "
            "Rows/cols: delay_probability, on_time_rate, claim_rate, "
            "supplier_price, order_frequency."
        ),
    )


class RankSupplierInput(BaseModel):
    """
    One ML-scored supplier ready for AHP-TOPSIS ranking.
    These fields come directly from SupplierPredictResult.
    """
    supplier_id:       int | None
    supplier_name:     str | None = None
    supplier_price:    float
    lead_time_days:    int
    claim_rate:        float
    on_time_rate:      float
    order_frequency:   int
    risk_level:        Literal["LOW", "MEDIUM", "HIGH"]
    delay_probability: float
    late_probability:  int


class RankRequest(BaseModel):
    """Send ML-scored suppliers + optional AHP matrix to get a TOPSIS ranking."""
    suppliers:  list[RankSupplierInput] = Field(
        ...,
        description="Supplier list produced by POST /predict/all-suppliers.",
    )
    ahp_matrix: AhpMatrixRequest = Field(
        default_factory=AhpMatrixRequest,
        description="Optional custom AHP pairwise matrix. Omit to use defaults.",
    )


class RankedSupplierResult(BaseModel):
    """One ranked supplier — ML fields + TOPSIS score and rank."""
    supplier_id:       int | None
    supplier_name:     str | None
    supplier_price:    float
    lead_time_days:    int
    claim_rate:        float
    on_time_rate:      float
    order_frequency:   int
    risk_level:        Literal["LOW", "MEDIUM", "HIGH"]
    delay_probability: float
    late_probability:  int
    topsis_score:      float
    topsis_rank:       int


class RankResponse(BaseModel):
    """AHP-TOPSIS ranking result."""
    total:             int
    consistency_ratio: float = Field(
        description="AHP consistency ratio (CR). Should be ≤ 0.10."
    )
    ahp_weights:       dict[str, float] = Field(
        description="Derived criterion weights (sum ≈ 1.0)."
    )
    ranked_suppliers:  list[RankedSupplierResult]


# ── Batch predict router ──────────────────────────────────────────────────────

batch_router = APIRouter(prefix="/predict", tags=["Batch Prediction"])


@batch_router.post(
    "/all-suppliers",
    response_model=BatchPredictResponse,
    summary="Batch-predict risk for multiple suppliers (ML results only, no ranking)",
    description=(
        "Accepts a list of supplier feature vectors, runs each through the trained "
        "XGBoost model, and returns raw ML results (risk_level + delay_probability). "
        "Call POST /rank/ahp-topsis next to rank these results."
    ),
    status_code=status.HTTP_200_OK,
)
def batch_predict_endpoint(payload: BatchPredictRequest) -> BatchPredictResponse:
    """POST /predict/all-suppliers"""
    if not payload.suppliers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="suppliers list must not be empty.",
        )

    results: list[SupplierPredictResult] = []

    for entry in payload.suppliers:
        features = entry.model_dump(exclude={"supplier_id", "supplier_name"})
        try:
            ml_result = predict_supplier_risk(features)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Prediction failed for supplier_id=%s: %s", entry.supplier_id, exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed for supplier_id={entry.supplier_id}: {exc}",
            ) from exc

        results.append(
            SupplierPredictResult(
                supplier_id=entry.supplier_id,
                supplier_name=entry.supplier_name,
                supplier_price=entry.supplier_price,
                lead_time_days=entry.lead_time_days,
                claim_rate=entry.claim_rate,
                on_time_rate=entry.on_time_rate,
                order_frequency=entry.order_frequency,
                **ml_result,
            )
        )

    logger.info("Batch prediction complete — %d suppliers scored.", len(results))
    return BatchPredictResponse(total=len(results), results=results)


# ── AHP-TOPSIS rank router ────────────────────────────────────────────────────

rank_router = APIRouter(prefix="/rank", tags=["Ranking"])


@rank_router.post(
    "/ahp-topsis",
    response_model=RankResponse,
    summary="Rank ML-scored suppliers using AHP-TOPSIS",
    description=(
        "Takes the output of POST /predict/all-suppliers (ML results) and ranks "
        "suppliers using AHP-derived weights and the TOPSIS method. "
        "The AHP pairwise matrix can be customised; leave it empty to use the "
        "built-in defaults (delay_probability weighted highest)."
    ),
    status_code=status.HTTP_200_OK,
)
def rank_ahp_topsis_endpoint(payload: RankRequest) -> RankResponse:
    """POST /rank/ahp-topsis"""
    if not payload.suppliers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="suppliers list must not be empty.",
        )

    # Build AHP matrix
    ahp_matrix: np.ndarray | None = None
    if payload.ahp_matrix.matrix is not None:
        try:
            ahp_matrix = np.array(payload.ahp_matrix.matrix, dtype=float)
            if ahp_matrix.shape != (5, 5):
                raise ValueError(
                    f"AHP matrix must be 5×5, got {ahp_matrix.shape}."
                )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid AHP matrix: {exc}",
            ) from exc

    supplier_dicts = [s.model_dump() for s in payload.suppliers]

    try:
        ranking = rank_suppliers(supplier_dicts, ahp_matrix=ahp_matrix)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("AHP-TOPSIS ranking failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ranking failed. Check server logs for details.",
        ) from exc

    ranked = [RankedSupplierResult(**row) for row in ranking["ranked_suppliers"]]

    return RankResponse(
        total=len(ranked),
        consistency_ratio=ranking["consistency_ratio"],
        ahp_weights=ranking["ahp_weights"],
        ranked_suppliers=ranked,
    )


train_router = APIRouter(prefix="/train", tags=["Training"])


def _run_train_and_respond(metrics: dict) -> TrainResponse:
    """Shared helper: invalidate cache and build the response."""
    invalidate_model_cache()

    raw_cv = metrics.get("cv_results", {})
    cv = CVResults(
        fold_metrics=[FoldMetrics(**f) for f in raw_cv.get("fold_metrics", [])],
        avg_log_loss=raw_cv.get("avg_log_loss"),
        avg_mse=raw_cv.get("avg_mse"),
        avg_mae=raw_cv.get("avg_mae"),
        avg_r2=raw_cv.get("avg_r2"),
        avg_accuracy=raw_cv.get("avg_accuracy"),
        avg_auc_roc=raw_cv.get("avg_auc_roc"),
    )

    return TrainResponse(
        message="Model trained and saved successfully.",
        split_method=metrics.get("split_method", "time_based_80_20"),
        samples_trained=metrics["samples_trained"],
        samples_tested=metrics["samples_tested"],
        best_round=metrics["best_round"],
        model_path=metrics["model_path"],
        data_source=metrics["data_source"],
        train_metrics=SplitMetrics(**metrics["train_metrics"]),
        test_metrics=SplitMetrics(**metrics["test_metrics"]),
        cv_results=cv,
    )


@train_router.post(
    "",
    response_model=TrainResponse,
    summary="Retrain using bundled dataset or server-side CSV path",
    description=(
        "Triggers a full model retrain. Optionally supply an absolute CSV path "
        "on the server. Leave the body empty to use the bundled dataset."
    ),
    status_code=status.HTTP_200_OK,
)
def train_model_endpoint(
    payload: TrainRequest = TrainRequest(),
) -> TrainResponse:
    """POST /train"""
    try:
        metrics = train(csv_path=payload.csv_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed. Check server logs for details.",
        ) from exc

    return _run_train_and_respond(metrics)


@train_router.post(
    "/from-rows",
    response_model=TrainResponse,
    summary="Retrain using live JSON rows from the ERP backend",
    description=(
        "The .NET ERP backend queries the database (PO, GR, Purchase Return, Catalog) "
        "and sends the aggregated rows here as JSON. The model retrains immediately on "
        "this fresh data. Set `append_to_existing=true` (default) to merge with the "
        "bundled historical dataset."
    ),
    status_code=status.HTTP_200_OK,
)
def train_from_rows_endpoint(payload: TrainFromRowsRequest) -> TrainResponse:
    """POST /train/from-rows — called by the .NET backend with live ERP data."""
    try:
        df = pd.DataFrame(payload.rows)
        df = normalize_columns(df)

        metrics = train(dataframe=df, append_to_existing=payload.append_to_existing)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Training from rows failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed. Check server logs for details.",
        ) from exc

    return _run_train_and_respond(metrics)


@train_router.post(
    "/from-csv-upload",
    response_model=TrainResponse,
    summary="Retrain by uploading a CSV file exported from the ERP database",
    description=(
        "Upload a CSV file directly. The .NET backend can export training data "
        "from SQL and POST it here as a multipart file. "
        "Required columns: supplier_price, lead_time_days, claim_rate, "
        "on_time_rate, order_frequency, late_delivery."
    ),
    status_code=status.HTTP_200_OK,
)
async def train_from_csv_upload_endpoint(
    file: UploadFile = File(..., description="CSV file exported from the ERP database"),
    append_to_existing: bool = True,
) -> TrainResponse:
    """POST /train/from-csv-upload"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df = normalize_columns(df)

        metrics = train(dataframe=df, append_to_existing=append_to_existing)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Training from CSV upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed. Check server logs for details.",
        ) from exc

    return _run_train_and_respond(metrics)
