"""
src/features/behavioral.py
==========================
Temporal and behavioural anomaly features.

Fraud is rarely random in time.  Card-not-present fraud clusters overnight
when cardholder monitoring is weakest; account-takeover attacks often occur
immediately after credential theft.  Behavioural anomalies — a suddenly
10× larger purchase than a cardholder's norm — are among the strongest
real-time fraud signals in production systems.

Feature groups produced
-----------------------
1. **Temporal** (from TransactionDT, a second-level integer delta)
   - ``hour_of_day``         : 0–23
   - ``day_of_week``         : 0 (Mon) – 6 (Sun)
   - ``is_night``            : bool — hour in {0,1,2,3,4,5}
   - ``is_weekend``          : bool — day in {5,6}
   - ``day_of_dataset``      : which calendar day of the dataset (day index)

2. **Amount features**
   - ``log1p_amt``           : log(1 + TransactionAmt) — stabilises the
                               heavy-tailed distribution for tree models
   - ``amt_cents``           : fractional part of TransactionAmt (fraud
                               transactions often end in .00 or .99)
   - ``is_round_amount``     : TransactionAmt % 10 == 0

3. **Spend-spike features** (entity historical mean from *training data only*)
   - ``amt_to_card1_mean_ratio``         : amt / historical mean per card
   - ``amt_to_addr1_mean_ratio``         : amt / historical mean per addr1
   - ``amt_to_P_emaildomain_mean_ratio`` : amt / historical mean per domain

   IMPORTANT — data leakage note
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Computing a mean over the entire dataset (including future rows) would
   leak future information into training examples.  To avoid this:

   * During training, call ``fit_amount_encoders(train_df)`` to build
     per-entity mean tables from the training set only.
   * During validation and test, pass the *fitted* ``encoders`` dict to
     ``compute_behavioral_features(df, encoders=encoders)`` so that the
     same training-time statistics are applied.

   The ``pipeline.build_features()`` orchestrator handles this automatically.

4. **Intra-entity deviation**
   - ``amt_zscore_card1``    : (amt − mean) / std, clipped to [−5, 5]
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Entities for which we compute amount ratios
_AMOUNT_ENTITIES: list[str] = ["card1", "addr1", "P_emaildomain"]

# Hours considered "night" (local time is unknown, so we use the relative
# hour derived from TransactionDT which tracks dataset-wide time-of-day)
_NIGHT_HOURS: frozenset[int] = frozenset(range(0, 6))   # 00:00–05:59

# Seconds in a day / week
_SECS_PER_HOUR = 3_600
_SECS_PER_DAY  = 86_400


# ---------------------------------------------------------------------------
# 1. Temporal features
# ---------------------------------------------------------------------------

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from ``TransactionDT``.

    ``TransactionDT`` is a second-level integer delta from an undisclosed
    reference timestamp.  We recover relative hour-of-day and day-of-week
    by taking modular arithmetic.  Because the absolute epoch is unknown,
    hour_of_day and day_of_week are *relative* (i.e., consistent across the
    dataset but may be shifted by a fixed offset from wall-clock time).
    This is fine — what matters is the *pattern*, not the exact hour label.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TransactionDT`` (integer, seconds).

    Returns
    -------
    pd.DataFrame
        ``df`` with temporal feature columns appended.
    """
    if "TransactionDT" not in df.columns:
        raise KeyError("behavioral: 'TransactionDT' not found in DataFrame.")

    dt = df["TransactionDT"]

    out = df.copy()
    out["hour_of_day"]   = ((dt // _SECS_PER_HOUR) % 24).astype(np.int8)
    out["day_of_week"]   = ((dt // _SECS_PER_DAY) % 7).astype(np.int8)
    out["day_of_dataset"] = (dt  // _SECS_PER_DAY).astype(np.int16)

    # Boolean flags (stored as int8 to save memory and work cleanly with trees)
    out["is_night"]   = out["hour_of_day"].isin(_NIGHT_HOURS).astype(np.int8)
    out["is_weekend"] = out["day_of_week"].isin({5, 6}).astype(np.int8)

    logger.debug(
        "Temporal: hour_of_day range=[%d,%d]  day_of_week range=[%d,%d]",
        out["hour_of_day"].min(), out["hour_of_day"].max(),
        out["day_of_week"].min(), out["day_of_week"].max(),
    )
    return out


# ---------------------------------------------------------------------------
# 2. Amount features
# ---------------------------------------------------------------------------

def compute_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive static features from ``TransactionAmt``.

    These require no historical lookups and are safe to compute on any split.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TransactionAmt`` (float).

    Returns
    -------
    pd.DataFrame
        ``df`` with amount feature columns appended.
    """
    if "TransactionAmt" not in df.columns:
        raise KeyError("behavioral: 'TransactionAmt' not found in DataFrame.")

    amt = df["TransactionAmt"].astype(np.float64)

    out = df.copy()
    out["log1p_amt"]      = np.log1p(amt).astype(np.float32)
    out["amt_cents"]      = (amt % 1.0).round(2).astype(np.float32)
    out["is_round_amount"] = ((amt % 10.0) == 0.0).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# 3. Spend-spike encoder (fit on train, apply to any split)
# ---------------------------------------------------------------------------

def fit_amount_encoders(
    train_df: pd.DataFrame,
    entity_cols: list[str] = _AMOUNT_ENTITIES,
) -> Dict[str, pd.Series]:
    """
    Compute per-entity historical mean and std of ``TransactionAmt``
    **using the training set only**.

    Call this once on the training partition and pass the returned ``encoders``
    dict to ``compute_spend_spike_features`` for val and test sets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    entity_cols : list of str
        Entities for which to compute statistics.

    Returns
    -------
    dict
        Keys: ``"{entity}_mean"`` and ``"{entity}_std"``.
        Values: pd.Series indexed by entity value.
    """
    encoders: Dict[str, pd.Series] = {}

    for col in entity_cols:
        if col not in train_df.columns:
            logger.warning("behavioral: encoder col '%s' missing — skipping.", col)
            continue

        grp = train_df.groupby(col, observed=True)["TransactionAmt"]
        encoders[f"{col}_mean"] = grp.mean().astype(np.float32)
        encoders[f"{col}_std"]  = grp.std().fillna(0).astype(np.float32)

        logger.info(
            "Amount encoder fitted for '%s': %d unique values.",
            col, len(encoders[f"{col}_mean"]),
        )

    # Global fallbacks (used for unseen entities at inference time)
    global_mean = train_df["TransactionAmt"].mean()
    global_std  = train_df["TransactionAmt"].std()
    encoders["_global_mean"] = float(global_mean)
    encoders["_global_std"]  = float(global_std)

    return encoders


def compute_spend_spike_features(
    df: pd.DataFrame,
    encoders: Dict[str, pd.Series],
    entity_cols: list[str] = _AMOUNT_ENTITIES,
) -> pd.DataFrame:
    """
    Compute amount-ratio and z-score features using pre-fitted ``encoders``.

    For each entity:
    - ``amt_to_{entity}_mean_ratio`` : TransactionAmt / historical mean
      (values >> 1 indicate a spend spike; < 1 a subdued transaction)
    - ``amt_zscore_{entity}``        : (amt − mean) / std, clipped to [−5, 5]

    Unseen entity values at inference time fall back to the global mean/std
    computed from the training set.

    Parameters
    ----------
    df : pd.DataFrame
    encoders : dict
        Output of ``fit_amount_encoders``.
    entity_cols : list of str

    Returns
    -------
    pd.DataFrame
        ``df`` with spike feature columns appended.
    """
    if "TransactionAmt" not in df.columns:
        raise KeyError("behavioral: 'TransactionAmt' not found in DataFrame.")

    out = df.copy()
    amt = out["TransactionAmt"].astype(np.float64)

    global_mean = encoders.get("_global_mean", 1.0)
    global_std  = encoders.get("_global_std",  1.0)

    for col in entity_cols:
        if col not in df.columns:
            logger.warning("behavioral: spike col '%s' missing — skipping.", col)
            continue

        mean_key = f"{col}_mean"
        std_key  = f"{col}_std"

        if mean_key not in encoders:
            logger.warning(
                "behavioral: encoder for '%s' not found — using global stats.", col
            )
            entity_mean = pd.Series(global_mean, index=df.index)
            entity_std  = pd.Series(global_std,  index=df.index)
        else:
            # Map entity value → historical mean/std.
            # .map() on a categorical Series inherits the categorical dtype,
            # so we must cast to float64 BEFORE .fillna() — otherwise pandas
            # raises TypeError when trying to insert a float into a categorical.
            entity_mean = (
                df[col]
                .map(encoders[mean_key])
                .astype(np.float64)          # break out of categorical dtype first
                .fillna(global_mean)
            )
            entity_std = (
                df[col]
                .map(encoders[std_key])
                .astype(np.float64)          # same fix for std series
                .fillna(global_std)
            )

        # Ratio: how many × the historical average is this transaction?
        # Add small epsilon to denominator to avoid division by zero on new cards
        ratio_col = f"amt_to_{col}_mean_ratio"
        out[ratio_col] = (amt / entity_mean.clip(lower=1e-6)).astype(np.float32)

        # Z-score: signed deviation in standard-deviation units
        zscore_col = f"amt_zscore_{col}"
        zscore = (amt - entity_mean) / entity_std.clip(lower=1e-6)
        out[zscore_col] = zscore.clip(-5, 5).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# 4. Unified entry point
# ---------------------------------------------------------------------------

def compute_behavioral_features(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, pd.Series]] = None,
    entity_cols: list[str] = _AMOUNT_ENTITIES,
) -> pd.DataFrame:
    """
    Compute all behavioural features in one call.

    If ``encoders`` is ``None``, spend-spike features are computed using
    statistics derived from ``df`` itself.

    .. warning::
       Passing ``encoders=None`` on validation or test data causes data
       leakage (the mean includes the current row's split).  Always pass
       pre-fitted encoders when working outside of training data.
       ``pipeline.build_features()`` handles this correctly.

    Parameters
    ----------
    df : pd.DataFrame
    encoders : dict, optional
        Output of ``fit_amount_encoders(train_df)``.  If None, statistics
        are derived from ``df`` (training-time only).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    logger.info("Behavioral: computing temporal features …")
    out = compute_temporal_features(df)

    logger.info("Behavioral: computing amount features …")
    out = compute_amount_features(out)

    if encoders is None:
        logger.info(
            "Behavioral: no encoders provided — fitting on current df "
            "(only correct if this is the training set)."
        )
        encoders = fit_amount_encoders(out, entity_cols=entity_cols)

    logger.info("Behavioral: computing spend-spike features …")
    out = compute_spend_spike_features(out, encoders=encoders, entity_cols=entity_cols)

    new_cols = [
        c for c in out.columns
        if c not in df.columns
    ]
    logger.info("Behavioral: added %d features: %s", len(new_cols), new_cols)
    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from src.ingestion.loader import load_and_merge
    from src.ingestion.splitter import time_aware_split

    df   = load_and_merge()
    train, val, test = time_aware_split(df)

    # Correct workflow: fit encoders on train, apply to all splits
    encoders = fit_amount_encoders(train)
    train_f  = compute_behavioral_features(train, encoders=encoders)
    val_f    = compute_behavioral_features(val,   encoders=encoders)

    behav_cols = [c for c in train_f.columns if c not in df.columns]
    print(f"\nAdded {len(behav_cols)} behavioural features:")
    print(train_f[["TransactionID", "isFraud"] + behav_cols].head(10).to_string())
