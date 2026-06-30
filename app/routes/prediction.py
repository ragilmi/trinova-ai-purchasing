import io
import logging
from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from app.services.predict import invalidate_model_cache, predict_supplier_risk
from app.services.train_model import REQUIRED_COLUMNS, train

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


class TrainResponse(BaseModel):
    message: str
    accuracy: float
    roc_auc: float
    samples_trained: int
    samples_tested: int
    model_path: str
    classification_report: str
    data_source: str

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

train_router = APIRouter(prefix="/train", tags=["Training"])


def _run_train_and_respond(metrics: dict) -> TrainResponse:
    """Shared helper: invalidate cache and build the response."""
    invalidate_model_cache()
    return TrainResponse(message="Model trained and saved successfully.", **metrics)


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

        # Normalize column names: strip whitespace and lowercase
        df.columns = [c.strip().lower() for c in df.columns]

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

        # Normalize column names: strip whitespace and lowercase
        # so headers like " Supplier_Price " still match
        df.columns = [c.strip().lower() for c in df.columns]

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
