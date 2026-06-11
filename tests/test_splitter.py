"""
tests/test_splitter.py
======================
Unit tests for src/ingestion/splitter.py
"""
import pytest
import pandas as pd
import numpy as np

from src.ingestion.splitter import (
    SplitConfig,
    time_aware_split,
    _assert_no_overlap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "TransactionAmt": rng.uniform(10, 500, size=n).astype(np.float32),
        "isFraud": rng.integers(0, 2, size=n).astype(np.int8),
    })


# ---------------------------------------------------------------------------
# SplitConfig validation
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SplitConfig()
    assert cfg.train_frac == 0.60
    assert cfg.val_frac   == 0.20
    assert abs(cfg.test_frac - 0.20) < 1e-9


def test_config_rejects_bad_fractions():
    with pytest.raises(ValueError):
        SplitConfig(train_frac=0.0)
    with pytest.raises(ValueError):
        SplitConfig(val_frac=1.0)
    with pytest.raises(ValueError):
        SplitConfig(train_frac=0.8, val_frac=0.3)  # sum > 1


# ---------------------------------------------------------------------------
# time_aware_split: shape and ordering
# ---------------------------------------------------------------------------

def test_split_shapes():
    df = _make_df(100)
    train, val, test = time_aware_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) == 60
    assert len(val)   == 20
    assert len(test)  == 20


def test_split_is_chronological():
    df = _make_df(200)
    train, val, test = time_aware_split(df)
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max()   <= test["TransactionDT"].min()


def test_split_missing_time_col():
    df = _make_df().drop(columns=["TransactionDT"])
    with pytest.raises(KeyError):
        time_aware_split(df)


# ---------------------------------------------------------------------------
# _assert_no_overlap: correct boundary logic (the bug that was fixed)
# ---------------------------------------------------------------------------

def test_no_overlap_passes_for_clean_splits():
    df = _make_df(150)
    train, val, test = time_aware_split(df)
    # Should not raise
    _assert_no_overlap(train, val, test, "TransactionDT")


def test_no_overlap_raises_when_val_starts_before_train_ends():
    """val.min < train.max must raise — this is the bug that was fixed."""
    df = _make_df(150)
    train, val, test = time_aware_split(df)

    # Artificially inject overlap: prepend a train row into val
    overlap_row = train.iloc[[-1]].copy()  # last train row
    val_bad = pd.concat([overlap_row, val], ignore_index=True)
    val_bad = val_bad.sort_values("TransactionDT").reset_index(drop=True)

    with pytest.raises(ValueError, match="Temporal overlap between train and val"):
        _assert_no_overlap(train, val_bad, test, "TransactionDT")


def test_no_overlap_raises_when_test_starts_before_val_ends():
    df = _make_df(150)
    train, val, test = time_aware_split(df)

    overlap_row = val.iloc[[-1]].copy()
    test_bad = pd.concat([overlap_row, test], ignore_index=True)
    test_bad = test_bad.sort_values("TransactionDT").reset_index(drop=True)

    with pytest.raises(ValueError, match="Temporal overlap between val and test"):
        _assert_no_overlap(train, val, test_bad, "TransactionDT")
