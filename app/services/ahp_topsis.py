"""
ahp_topsis.py
AHP-TOPSIS supplier ranking based on XGBoost ML prediction results.

Flow:
  1. XGBoost produces a delay_probability (0–1) for each supplier.
  2. AHP derives criterion weights from a pairwise comparison matrix.
  3. TOPSIS ranks suppliers: closer to the ideal best, farther from the ideal worst.

Criteria (5 columns, all sourced from ERP aggregation + ML output):
  - delay_probability  : XGBoost output — lower is better  (cost/benefit: cost)
  - on_time_rate       : ERP metric    — higher is better  (cost/benefit: benefit)
  - claim_rate         : ERP metric    — lower is better   (cost/benefit: cost)
  - supplier_price     : ERP metric    — lower is better   (cost/benefit: cost)
  - order_frequency    : ERP metric    — higher is better  (cost/benefit: benefit)
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criterion configuration
# ---------------------------------------------------------------------------

# Criteria included in ranking and their direction.
# "benefit" = higher is better, "cost" = lower is better.
CRITERIA: list[tuple[str, Literal["benefit", "cost"]]] = [
    ("delay_probability", "cost"),
    ("on_time_rate",      "benefit"),
    ("claim_rate",        "cost"),
    ("supplier_price",    "cost"),
    ("order_frequency",   "benefit"),
]

CRITERION_NAMES = [c[0] for c in CRITERIA]
CRITERION_TYPES = {c[0]: c[1] for c in CRITERIA}

# ---------------------------------------------------------------------------
# Default AHP pairwise comparison matrix
# ---------------------------------------------------------------------------
# Scale: 1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme importance
#
# Interpretation of these defaults:
#   delay_probability  is the most critical signal (ML output) — weighted highest
#   on_time_rate       is the second most important ERP metric
#   claim_rate         moderately important
#   supplier_price     slightly less than claim_rate
#   order_frequency    least important (informational, not a direct risk driver)
#
# Row/col order matches CRITERION_NAMES above.
DEFAULT_AHP_MATRIX = np.array([
    #  delay_prob  on_time  claim   price   freq
    [1.0,          3.0,     5.0,    5.0,    7.0],   # delay_probability
    [1/3,          1.0,     3.0,    3.0,    5.0],   # on_time_rate
    [1/5,          1/3,     1.0,    1.0,    3.0],   # claim_rate
    [1/5,          1/3,     1.0,    1.0,    3.0],   # supplier_price
    [1/7,          1/5,     1/3,    1/3,    1.0],   # order_frequency
], dtype=float)

# Saaty's Random Index table (n = 1..10)
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ---------------------------------------------------------------------------
# AHP helpers
# ---------------------------------------------------------------------------

def compute_ahp_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Derive AHP priority weights from a pairwise comparison matrix.

    Uses the eigenvector method (normalise each column, then row-average).

    Args:
        matrix: n×n positive reciprocal matrix.

    Returns:
        (weights, consistency_ratio) where weights sum to 1.0.

    Raises:
        ValueError: If the matrix is not square, or CR > 0.10 (inconsistent).
    """
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"AHP matrix must be square, got shape {matrix.shape}.")

    # Step 1 – column normalisation
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums

    # Step 2 – priority vector (row means of normalised matrix)
    weights = norm.mean(axis=1)

    # Step 3 – consistency check
    weighted_sum = matrix @ weights
    lambda_vec   = weighted_sum / weights
    lambda_max   = lambda_vec.mean()
    ci           = (lambda_max - n) / (n - 1)
    ri           = _RI.get(n, 1.49)
    cr           = ci / ri if ri > 0 else 0.0

    logger.debug("AHP λ_max=%.4f, CI=%.4f, CR=%.4f", lambda_max, ci, cr)

    if cr > 0.10:
        logger.warning(
            "AHP consistency ratio %.4f exceeds 0.10. "
            "The pairwise comparison matrix may need revision.",
            cr,
        )

    return weights, round(cr, 4)


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

def topsis_rank(
    df: pd.DataFrame,
    weights: np.ndarray,
    criteria: list[tuple[str, Literal["benefit", "cost"]]],
) -> pd.DataFrame:
    """Rank rows using TOPSIS.

    Steps:
      1. Build decision matrix X from selected criterion columns.
      2. Normalise: r_ij = x_ij / sqrt(sum(x_ij^2)).
      3. Weight:    v_ij = w_j * r_ij.
      4. Determine ideal best (A+) and ideal worst (A−) per criterion.
      5. Compute Euclidean distances d+ and d−.
      6. Closeness coefficient: C_i = d− / (d+ + d−).
      7. Rank by C_i descending (highest C = best supplier).

    Args:
        df:       DataFrame that must contain all criterion columns.
        weights:  1-D array of AHP weights (must sum ≈ 1, length == len(criteria)).
        criteria: List of (column_name, "benefit"|"cost") tuples.

    Returns:
        Input DataFrame with three new columns appended:
            topsis_score (float, 0–1), topsis_rank (int, 1 = best).
    """
    names = [c[0] for c in criteria]
    types = {c[0]: c[1] for c in criteria}

    X = df[names].values.astype(float)

    # --- Step 2: vector normalisation ---
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms[norms == 0] = 1.0          # guard against zero columns
    R = X / norms

    # --- Step 3: weighted normalised matrix ---
    V = R * weights

    # --- Step 4: ideal best / worst ---
    A_plus  = np.where(
        [types[n] == "benefit" for n in names],
        V.max(axis=0),
        V.min(axis=0),
    )
    A_minus = np.where(
        [types[n] == "benefit" for n in names],
        V.min(axis=0),
        V.max(axis=0),
    )

    # --- Step 5: distances ---
    d_plus  = np.sqrt(((V - A_plus)  ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    # --- Step 6: closeness coefficient ---
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-9         # guard against identical rows
    C = d_minus / denom

    result = df.copy()
    result["topsis_score"] = np.round(C, 6)
    result["topsis_rank"]  = pd.Series(C).rank(ascending=False, method="min").astype(int).values

    return result.sort_values("topsis_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rank_suppliers(
    suppliers: list[dict],
    ahp_matrix: np.ndarray | None = None,
) -> dict:
    """Full AHP-TOPSIS pipeline for a list of ML-scored suppliers.

    Expected keys per supplier dict (all produced by the ML batch predict step):
        supplier_id       (int)
        supplier_name     (str, optional)
        delay_probability (float)   — from XGBoost
        risk_level        (str)     — from XGBoost
        late_probability  (int)     — from XGBoost
        on_time_rate      (float)   — from ERP
        claim_rate        (float)   — from ERP
        supplier_price    (float)   — from ERP
        order_frequency   (int)     — from ERP

    Args:
        suppliers:  List of supplier dicts (ML predict results + ERP fields).
        ahp_matrix: Optional custom n×n pairwise matrix. Uses DEFAULT_AHP_MATRIX
                    if not provided.

    Returns:
        {
            "ahp_weights":          { criterion: weight, ... },
            "consistency_ratio":    float,
            "ranked_suppliers":     [ { supplier fields + topsis_score + topsis_rank }, ... ]
        }
    """
    if not suppliers:
        raise ValueError("suppliers list must not be empty.")

    matrix = ahp_matrix if ahp_matrix is not None else DEFAULT_AHP_MATRIX
    weights, cr = compute_ahp_weights(matrix)

    df = pd.DataFrame(suppliers)

    # Ensure all criterion columns exist; fill missing numeric cols with 0
    for col, _ in CRITERIA:
        if col not in df.columns:
            logger.warning("Column '%s' missing from supplier data — defaulting to 0.", col)
            df[col] = 0.0

    ranked_df = topsis_rank(df, weights, CRITERIA)

    # Build weight summary
    ahp_weights_dict = {
        name: round(float(w), 6)
        for name, w in zip(CRITERION_NAMES, weights)
    }

    ranked_list = ranked_df.to_dict(orient="records")

    logger.info(
        "AHP-TOPSIS complete — %d suppliers ranked, CR=%.4f", len(ranked_list), cr
    )

    return {
        "ahp_weights":       ahp_weights_dict,
        "consistency_ratio": cr,
        "ranked_suppliers":  ranked_list,
    }
