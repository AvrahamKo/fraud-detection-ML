"""
src/ingestion/splitter.py
=========================
Time-aware train / validation / test split for the IEEE-CIS dataset.

Why time-aware?
---------------
``TransactionDT`` is a second-level time-delta from an undisclosed reference
date.  Fraudsters adapt — a random split would expose the model to future
temporal patterns during training, inflating validation metrics and producing
models that collapse on live traffic.

A chronological split (sort → slice) correctly simulates:
  * Training on older transactions
  * Validating on slightly newer ones (hyper-parameter tuning)
  * Testing on the most recent slice (hold-out evaluation)

Default partition sizes
-----------------------
  +-----------+------+
  | Partition | Frac |
  +===========+======+
  | train     | 60 % |
  | val       | 20 % |
  | test      | 20 % |
  +-----------+------+

The fractions are configurable via ``SplitConfig``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
_PROCESSED_DIR   = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """
    Hyper-parameters for the time-aware split.

    Attributes
    ----------
    train_frac : float
        Fraction of rows allocated to training (default 0.60).
    val_frac : float
        Fraction of rows allocated to validation (default 0.20).
        The test fraction is implicit: ``1 - train_frac - val_frac``.
    time_col : str
        Column used for chronological sorting (default ``"TransactionDT"``).
    label_col : str
        Target label column used only for logging fraud rates per split.
    """
    train_frac : float = 0.60
    val_frac   : float = 0.20
    time_col   : str   = "TransactionDT"
    label_col  : str   = "isFraud"

    def __post_init__(self) -> None:
        if not (0 < self.train_frac < 1):
            raise ValueError("train_frac must be in (0, 1)")
        if not (0 < self.val_frac < 1):
            raise ValueError("val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac ({self.train_frac}) + val_frac ({self.val_frac}) "
                f"must be < 1.0 to leave room for the test set."
            )

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def time_aware_split(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort ``df`` by ``config.time_col`` and slice into train / val / test.

    No shuffling is performed. The returned DataFrames have a reset integer
    index so that downstream iloc-based access is safe.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged dataset produced by ``loader.load_and_merge()``.
    config : SplitConfig
        Partition fractions and column names.

    Returns
    -------
    train, val, test : pd.DataFrame
        Three non-overlapping, chronologically ordered partitions.

    Raises
    ------
    KeyError
        If ``config.time_col`` is not a column in ``df``.
    ValueError
        If ``config`` fractions do not satisfy the constraint.

    Examples
    --------
    >>> from src.ingestion.splitter import time_aware_split, SplitConfig
    >>> train, val, test = time_aware_split(df)
    >>> assert test[config.time_col].min() >= val[config.time_col].max()
    """
    if config.time_col not in df.columns:
        raise KeyError(
            f"Time column '{config.time_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns[:10])} …"
        )

    logger.info(
        "Sorting %d rows by '%s' …", len(df), config.time_col
    )
    df_sorted = df.sort_values(config.time_col).reset_index(drop=True)

    n          = len(df_sorted)
    train_end  = int(n * config.train_frac)
    val_end    = int(n * (config.train_frac + config.val_frac))

    train = df_sorted.iloc[:train_end].copy()
    val   = df_sorted.iloc[train_end:val_end].copy()
    test  = df_sorted.iloc[val_end:].copy()

    _log_split_summary(train, val, test, config)
    _assert_no_overlap(train, val, test, config.time_col)

    return train, val, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: SplitConfig,
) -> None:
    """Log row counts, time ranges, and fraud rates for each partition."""
    total = len(train) + len(val) + len(test)
    rows = [
        ("Train", train),
        ("Val",   val),
        ("Test",  test),
    ]
    logger.info("─" * 72)
    logger.info("%-6s  %8s  %8s  %12s  %12s  %8s", "Split",
                "Rows", "Frac%", "DT_min", "DT_max", "FraudRate")
    logger.info("─" * 72)
    for name, split in rows:
        t_min      = split[config.time_col].min()
        t_max      = split[config.time_col].max()
        frac_pct   = 100.0 * len(split) / total
        fraud_rate = split[config.label_col].mean() if config.label_col in split.columns else float("nan")
        logger.info(
            "%-6s  %8,d  %7.1f%%  %12,d  %12,d  %7.3f%%",
            name, len(split), frac_pct, int(t_min), int(t_max),
            100.0 * fraud_rate,
        )
    logger.info("─" * 72)


def _assert_no_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str,
) -> None:
    """
    Sanity check: val must start at or after train ends;
    test must start at or after val ends.

    Compares val.min() against train.MAX (not train.min) — the only way to
    confirm there is no temporal overlap between adjacent partitions.

    Raises
    ------
    ValueError
        If any adjacent split pair has a temporal overlap.
    """
    train_max = train[time_col].max()
    val_min   = val[time_col].min()
    val_max   = val[time_col].max()
    test_min  = test[time_col].min()

    if val_min < train_max:
        raise ValueError(
            f"Temporal overlap between train and val: "
            f"val starts at {val_min} but train ends at {train_max}."
        )
    if test_min < val_max:
        raise ValueError(
            f"Temporal overlap between val and test: "
            f"test starts at {test_min} but val ends at {val_max}."
        )
    logger.info(
        "Overlap check passed — train ends at %d, val starts at %d, "
        "val ends at %d, test starts at %d.",
        train_max, val_min, val_max, test_min,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = _PROCESSED_DIR,
    fmt: str = "parquet",
) -> Dict[str, Path]:
    """
    Persist train / val / test splits to disk.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    out_dir : Path
        Destination directory (created if it doesn't exist).
    fmt : str
        ``"parquet"`` (default, columnar, fast) or ``"csv"``.

    Returns
    -------
    dict
        Mapping of split name → saved file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        if fmt == "parquet":
            p = out_dir / f"{name}.parquet"
            split.to_parquet(p, index=False)
        elif fmt == "csv":
            p = out_dir / f"{name}.csv"
            split.to_csv(p, index=False)
        else:
            raise ValueError(f"Unsupported format: '{fmt}'. Choose 'parquet' or 'csv'.")
        paths[name] = p
        logger.info("Saved %s → %s", name, p)

    return paths


def load_splits(
    in_dir: Path = _PROCESSED_DIR,
    fmt: str = "parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously persisted splits from disk.

    Parameters
    ----------
    in_dir : Path
        Directory containing ``train``, ``val``, ``test`` files.
    fmt : str
        ``"parquet"`` or ``"csv"``.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    in_dir = Path(in_dir)
    splits = {}
    for name in ("train", "val", "test"):
        p = in_dir / f"{name}.{fmt}"
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\n"
                f"Run the ingestion pipeline first: python -m src.ingestion.splitter"
            )
        splits[name] = pd.read_parquet(p) if fmt == "parquet" else pd.read_csv(p)
        logger.info("Loaded %-5s ← %s  shape=%s", name, p, splits[name].shape)
    return splits["train"], splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# CLI — run end-to-end ingestion + split in one command
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.ingestion.loader import load_and_merge

    logger.info("=== Ingestion + Split Pipeline ===")
    df = load_and_merge()

    config = SplitConfig()          # 60 / 20 / 20
    train, val, test = time_aware_split(df, config)

    paths = save_splits(train, val, test)
    logger.info("All splits saved.  Paths: %s", paths)
