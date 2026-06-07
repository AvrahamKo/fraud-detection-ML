"""
src/ingestion/loader.py
=======================
Loads, merges, and memory-optimises the IEEE-CIS Fraud Detection dataset.

Design decisions
----------------
* Left join (transactions ← identity): not every transaction has identity
  enrichment.  A left join is the production-correct choice — missing
  identity rows are filled with NaN and treated as sparse features downstream.
* Memory optimisation: the raw CSVs together occupy ~1.7 GB.  Downcasting
  int64 → int32/int16/int8 and float64 → float32 cuts this by ~55 %.
* Object columns are cast to pandas `category` dtype, saving additional RAM
  and speeding up groupby operations in feature engineering.
* All paths are resolved relative to the project root so the module works
  regardless of the caller's working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

DEFAULT_TRANSACTION_PATH = _RAW_DIR / "train_transaction.csv"
DEFAULT_IDENTITY_PATH    = _RAW_DIR / "train_identity.csv"


# ---------------------------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------------------------

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns and cast object columns to ``category``.

    The function inspects each column's dtype and selects the smallest
    numeric type that can represent the observed min/max without overflow.
    Object columns (strings) are cast to ``category`` dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (modified in-place for efficiency, but a reference
        to the same object is returned for chaining).
    verbose : bool
        Whether to log before/after memory usage.

    Returns
    -------
    pd.DataFrame
        The same dataframe with downcasted dtypes.
    """
    start_mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            df[col] = df[col].astype("category")

        elif col_type.name == "bool":
            df[col] = df[col].astype(np.int8)

        elif np.issubdtype(col_type, np.integer):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min  and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            # else: leave as int64

        elif np.issubdtype(col_type, np.floating):
            c_min, c_max = df[col].min(), df[col].max()
            if (c_min >= np.finfo(np.float32).min and
                    c_max <= np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)
            # else: leave as float64

    end_mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction  = 100.0 * (start_mem_mb - end_mem_mb) / start_mem_mb

    if verbose:
        logger.info(
            "Memory: %.2f MB → %.2f MB  (%.1f %% reduction)",
            start_mem_mb, end_mem_mb, reduction,
        )
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_transactions(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read ``train_transaction.csv`` into a DataFrame.

    Parameters
    ----------
    path : Path, optional
        Override the default raw-data path.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path) if path else DEFAULT_TRANSACTION_PATH
    logger.info("Loading transactions from: %s", path)
    df = pd.read_csv(path)
    logger.info("Transactions loaded  →  shape: %s", df.shape)
    return df


def load_identity(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read ``train_identity.csv`` into a DataFrame.

    Parameters
    ----------
    path : Path, optional
        Override the default raw-data path.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path) if path else DEFAULT_IDENTITY_PATH
    logger.info("Loading identity from: %s", path)
    df = pd.read_csv(path)
    logger.info("Identity loaded      →  shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_tables(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join transactions with identity on ``TransactionID``.

    A left join is the correct semantic here:
    - Every transaction must appear in the output (even without identity).
    - Identity columns for unmatched transactions will be NaN — this is
      modelled explicitly in downstream feature engineering (e.g. a binary
      ``has_identity`` flag).

    Parameters
    ----------
    transactions : pd.DataFrame
    identity     : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged frame with all transaction columns followed by identity columns.
    """
    logger.info("Merging on TransactionID (left join) …")
    merged = transactions.merge(identity, on="TransactionID", how="left")

    n_with_identity = identity["TransactionID"].isin(
        transactions["TransactionID"]
    ).sum()
    join_rate = n_with_identity / len(transactions)

    logger.info(
        "Identity enrichment rate: %d / %d transactions  (%.1f %%)",
        n_with_identity, len(transactions), 100.0 * join_rate,
    )
    logger.info("Merged shape: %s", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_merge(
    transaction_path: Optional[Path] = None,
    identity_path: Optional[Path] = None,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: load → merge → (optionally) optimise memory.

    This is the primary entry point for all downstream modules.

    Parameters
    ----------
    transaction_path : Path, optional
        Path to ``train_transaction.csv``.  Defaults to ``data/raw/``.
    identity_path : Path, optional
        Path to ``train_identity.csv``.  Defaults to ``data/raw/``.
    optimize_memory : bool
        If True, downcast numeric dtypes and convert object columns to
        ``category`` to reduce RAM usage by ~55 %.

    Returns
    -------
    pd.DataFrame
        Merged, ready-to-use dataset.

    Examples
    --------
    >>> from src.ingestion.loader import load_and_merge
    >>> df = load_and_merge()
    >>> df.shape
    (590540, 434)
    """
    transactions = load_transactions(transaction_path)
    identity     = load_identity(identity_path)
    df           = merge_tables(transactions, identity)

    if optimize_memory:
        logger.info("Optimising memory …")
        df = reduce_mem_usage(df)

    logger.info("Ingestion complete. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI entry point — run directly to validate ingestion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_and_merge()
    print("\n--- Dataset overview ---")
    print(f"Shape          : {df.shape}")
    print(f"Fraud rate     : {df['isFraud'].mean():.4%}")
    print(f"Null pct (mean): {df.isnull().mean().mean():.4%}")
    print(f"\nDtypes summary:\n{df.dtypes.value_counts()}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
