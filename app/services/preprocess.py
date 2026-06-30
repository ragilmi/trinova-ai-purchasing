"""
preprocess.py
Handles feature engineering and data preprocessing for the supplier risk model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Feature columns expected by the model — order matters
FEATURE_COLUMNS = [
    "supplier_price",
    "lead_time_days",
    "claim_rate",
    "on_time_rate",
    "order_frequency",
]

LABEL_COLUMN = "late_delivery"


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the training CSV from disk.

    Args:
        csv_path: Explicit path to the CSV. Defaults to the bundled
                  datasets/supplier_training.csv.

    Returns:
        Raw DataFrame.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "supplier_training.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {csv_path}")

    return pd.read_csv(csv_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model signal.

    New columns:
    - price_per_day        : supplier_price / lead_time_days  (cost efficiency)
    - risk_composite       : claim_rate + (1 - on_time_rate)  (combined risk signal)
    - frequency_reliability: order_frequency * on_time_rate   (volume-weighted reliability)

    Args:
        df: DataFrame that must contain FEATURE_COLUMNS.

    Returns:
        DataFrame with added engineered columns.
    """
    df = df.copy()

    # Avoid division by zero for lead_time_days
    df["price_per_day"] = df["supplier_price"] / df["lead_time_days"].replace(0, 1)

    # Composite risk score: high claim rate + low on-time rate = worse
    df["risk_composite"] = df["claim_rate"] + (1.0 - df["on_time_rate"])

    # High frequency + high on-time = reliable
    df["frequency_reliability"] = df["order_frequency"] * df["on_time_rate"]

    return df


def get_all_feature_columns() -> list[str]:
    """Return the full feature list including engineered columns."""
    return FEATURE_COLUMNS + [
        "price_per_day",
        "risk_composite",
        "frequency_reliability",
    ]


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Full preprocessing pipeline for a raw DataFrame.

    Applies feature engineering, selects model columns, and optionally
    returns the label series if present.

    Args:
        df: Raw DataFrame (from CSV or API payload converted to DataFrame).

    Returns:
        (X, y) where y is None when no label column exists.
    """
    df = engineer_features(df)

    feature_cols = get_all_feature_columns()
    X = df[feature_cols].astype(float)

    y = None
    if LABEL_COLUMN in df.columns:
        y = df[LABEL_COLUMN].astype(int)

    return X, y


def preprocess_single(input_dict: dict) -> pd.DataFrame:
    """Preprocess a single prediction payload (from the API).

    Args:
        input_dict: Dict matching SupplierInput fields.

    Returns:
        Single-row DataFrame ready for model.predict().
    """
    df = pd.DataFrame([input_dict])
    X, _ = preprocess_dataframe(df)
    return X
