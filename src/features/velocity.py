"""
src/features/velocity.py
========================
Rolling velocity features for fraud detection.

Velocity captures *how much activity an entity (card, address, email) has
produced in a recent time window*.  Fraudsters typically burst through a
compromised card quickly before it is blocked — so a card that fires 30
transactions in 1 hour is a much stronger fraud signal than the same card
doing 30 over a week.

Feature naming convention
--------------------------
  vel_{entity}_{window}_{stat}

  entity  : card1 | addr1 | P_emaildomain
  window  : 1h | 24h | 7d
  stat    : count | sum | mean_amt

Examples
--------
  vel_card1_1h_count      — how many transactions on this card in the last hour
  vel_addr1_24h_sum       — total $$ volume from this billing region last 24 h
  vel_P_emaildomain_7d_mean_amt — avg $ from this email domain last 7 days

Implementation notes
--------------------
* ``TransactionDT`` is a second-level integer delta.  We convert it to a
  ``pd.Timestamp`` using an arbitrary epoch so pandas time-based rolling
  (e.g. "3600s") works correctly.

* ``closed="left"`` on the rolling window means the current transaction is
  NOT included in its own window — this prevents data leakage and correctly
  models "history up to but not including this transaction".

* ``min_periods=0`` means a window returns 0 (not NaN) when there are no
  prior transactions — e.g. the very first transaction on a new card gets a
  count of 0, which is an informative signal in itself.

* We use ``groupby(...).apply(_roll)`` rather than the more exotic
  ``groupby().rolling()`` chaining.  The latter has edge-case behaviour
  with duplicate timestamps (multiple transactions in the same second on
  the same card) in some pandas versions.  The apply approach is
  unambiguous and produces correct results.

Performance
-----------
For 590k rows across ~13.5k card1 groups the apply loop runs in
roughly 30–90 s depending on hardware.  For production scale (billions of
rows), replace with PySpark ``Window.rangeBetween`` or a Polars ``group_by``
+ ``rolling``.  The interface of this module is designed to be a drop-in
replacement: swap the implementation, keep the column names.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Arbitrary epoch — only the relative deltas matter for rolling windows
_SURROGATE_EPOCH = pd.Timestamp("2000-01-01 00:00:00")

# Window definitions: label → seconds
VELOCITY_WINDOWS: Dict[str, int] = {
    "1h":  3_600,
    "24h": 86_400,
    "7d":  604_800,
}

# Entities to aggregate over
ENTITY_COLS: List[str] = ["card1", "addr1", "P_emaildomain"]


# ---------------------------------------------------------------------------
# Core rolling helper
# ---------------------------------------------------------------------------

def _rolling_entity_window(
    df: pd.DataFrame,
    entity_col: str,
    window_sec: int,
    window_label: str,
) -> pd.DataFrame:
    """
    Compute rolling count, sum, and mean of ``TransactionAmt`` for one
    (entity, window) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Full (or split) dataset.  Must contain ``TransactionID``,
        ``entity_col``, ``TransactionDT``, and ``TransactionAmt``.
    entity_col : str
        Column to group by (e.g. "card1").
    window_sec : int
        Window size in seconds.
    window_label : str
        Short label used in the output column names (e.g. "1h").

    Returns
    -------
    pd.DataFrame
        Three-column frame: ``TransactionID`` + the three new feature columns.
        Merge back to the original df on ``TransactionID``.
    """
    prefix     = f"vel_{entity_col}_{window_label}"
    count_col  = f"{prefix}_count"
    sum_col    = f"{prefix}_sum"
    mean_col   = f"{prefix}_mean_amt"
    window_str = f"{window_sec}s"

    # ── lightweight working copy ──────────────────────────────────────────
    required = ["TransactionID", entity_col, "TransactionDT", "TransactionAmt"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"velocity: required columns missing from DataFrame: {missing}"
        )

    work = df[required].copy()

    # Fill NaN in entity column with a sentinel so NaN rows form their own
    # "unknown" group rather than being silently dropped.
    # Categorical columns (produced by loader.reduce_mem_usage) require the
    # new category to be registered BEFORE fillna — otherwise pandas raises
    # TypeError: "Cannot setitem on a Categorical with a new category".
    if pd.api.types.is_categorical_dtype(work[entity_col]):
        if "__UNKNOWN__" not in work[entity_col].cat.categories:
            work[entity_col] = work[entity_col].cat.add_categories("__UNKNOWN__")
    work[entity_col] = work[entity_col].fillna("__UNKNOWN__")

    # Convert second-delta to datetime for time-based rolling
    work["_dt"] = _SURROGATE_EPOCH + pd.to_timedelta(work["TransactionDT"], unit="s")

    # Sort within each entity by time (required for rolling correctness)
    work = work.sort_values([entity_col, "_dt"]).reset_index(drop=True)

    # ── inner rolling function applied per group ──────────────────────────
    def _roll(group: pd.DataFrame) -> pd.DataFrame:
        """Roll over a single entity group sorted by _dt."""
        amt_indexed = group.set_index("_dt")["TransactionAmt"]

        roll = amt_indexed.rolling(
            window=window_str,
            closed="left",     # exclude the current row → no lookahead leakage
            min_periods=0,     # return 0 (not NaN) for the first transaction
        )

        cnt = roll.count().values
        sm  = roll.sum().values
        # Mean: guard against zero count (first transaction → count=0)
        mn  = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)

        out = group[["TransactionID"]].copy()
        out[count_col] = cnt.astype(np.float32)
        out[sum_col]   = sm.astype(np.float32)
        out[mean_col]  = mn.astype(np.float32)
        return out

    # Select only the columns _roll needs — this excludes the grouping column
    # from the sub-DataFrame passed to apply, suppressing the FutureWarning
    # about groupby operating on grouping columns (pandas >= 2.2 compat).
    _cols_for_roll = ["TransactionID", "_dt", "TransactionAmt"]
    result = (
        work.groupby(entity_col, observed=True, sort=False, group_keys=False)[_cols_for_roll]
        .apply(_roll)
    )

    return result[["TransactionID", count_col, sum_col, mean_col]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_velocity_features(
    df: pd.DataFrame,
    entity_cols: Sequence[str] = ENTITY_COLS,
    windows: Dict[str, int]   = VELOCITY_WINDOWS,
) -> pd.DataFrame:
    """
    Compute all velocity features and merge them back to ``df``.

    Iterates over every (entity, window) combination and calls
    ``_rolling_entity_window`` for each.  Results are joined back via
    ``TransactionID`` so the original row order is preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset produced by the ingestion layer.
    entity_cols : sequence of str
        Entity columns to aggregate over.  Defaults to ``ENTITY_COLS``.
    windows : dict
        Mapping of window label → seconds.  Defaults to ``VELOCITY_WINDOWS``.

    Returns
    -------
    pd.DataFrame
        Original ``df`` with ``len(entity_cols) × len(windows) × 3``
        new feature columns appended.  Row order is unchanged.

    Examples
    --------
    >>> from src.features.velocity import compute_velocity_features
    >>> df_feat = compute_velocity_features(df)
    >>> [c for c in df_feat.columns if c.startswith("vel_")]
    ['vel_card1_1h_count', 'vel_card1_1h_sum', ...]
    """
    out = df.copy()
    total_combos = len(entity_cols) * len(windows)
    done = 0

    for entity_col in entity_cols:
        if entity_col not in df.columns:
            logger.warning("velocity: '%s' not in DataFrame — skipping.", entity_col)
            continue

        for window_label, window_sec in windows.items():
            done += 1
            logger.info(
                "[%d/%d] Computing velocity: entity=%s  window=%s",
                done, total_combos, entity_col, window_label,
            )

            feat_df = _rolling_entity_window(
                df=df,
                entity_col=entity_col,
                window_sec=window_sec,
                window_label=window_label,
            )

            # Left-join on TransactionID — preserves original row order and
            # handles any entity columns that were entirely NaN
            out = out.merge(feat_df, on="TransactionID", how="left")

    new_cols = [c for c in out.columns if c.startswith("vel_")]
    logger.info(
        "Velocity: added %d features.  NaN check: %s",
        len(new_cols),
        out[new_cols].isnull().any().any(),
    )
    return out


# ---------------------------------------------------------------------------
# Feature name registry (used by pipeline.py for validation)
# ---------------------------------------------------------------------------

def expected_feature_names(
    entity_cols: Sequence[str] = ENTITY_COLS,
    windows: Dict[str, int]   = VELOCITY_WINDOWS,
) -> List[str]:
    """Return the list of column names this module will produce."""
    names = []
    for entity in entity_cols:
        for wlabel in windows:
            prefix = f"vel_{entity}_{wlabel}"
            names += [f"{prefix}_count", f"{prefix}_sum", f"{prefix}_mean_amt"]
    return names


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from src.ingestion.loader import load_and_merge

    df = load_and_merge()
    df_small = df.head(5_000)  # quick smoke test on a slice
    result = compute_velocity_features(df_small)

    vel_cols = [c for c in result.columns if c.startswith("vel_")]
    print(f"\nAdded {len(vel_cols)} velocity features.")
    print(result[["TransactionID", "isFraud"] + vel_cols[:9]].head(10).to_string())
