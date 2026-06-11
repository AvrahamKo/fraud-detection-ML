"""
tests/test_behavioral.py
========================
Unit tests for src/features/behavioral.py
"""
import numpy as np
import pandas as pd
import pytest

from src.features.behavioral import (
    compute_temporal_features,
    compute_amount_features,
    fit_amount_encoders,
    compute_spend_spike_features,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "TransactionID":  np.arange(n),
        "TransactionDT":  np.sort(rng.integers(0, 2_000_000, size=n)),
        "TransactionAmt": rng.uniform(10, 1000, size=n).astype(np.float32),
        "card1":          rng.integers(1000, 9999, size=n).astype(np.int16),
        "addr1":          rng.integers(100, 999, size=n).astype(np.int16),
        "P_emaildomain":  pd.Categorical(
            rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], size=n)
        ),
    })


# ---------------------------------------------------------------------------
# compute_temporal_features
# ---------------------------------------------------------------------------

def test_temporal_features_added():
    df = _make_df()
    out = compute_temporal_features(df)
    for col in ["hour_of_day", "day_of_week", "is_night", "is_weekend", "day_of_dataset"]:
        assert col in out.columns, f"Missing column: {col}"


def test_hour_of_day_range():
    df = _make_df()
    out = compute_temporal_features(df)
    assert out["hour_of_day"].between(0, 23).all()


def test_day_of_week_range():
    df = _make_df()
    out = compute_temporal_features(df)
    assert out["day_of_week"].between(0, 6).all()


def test_temporal_missing_col():
    df = _make_df().drop(columns=["TransactionDT"])
    with pytest.raises(KeyError):
        compute_temporal_features(df)


# ---------------------------------------------------------------------------
# compute_amount_features
# ---------------------------------------------------------------------------

def test_amount_features_added():
    df = _make_df()
    out = compute_amount_features(df)
    for col in ["log1p_amt", "amt_cents", "is_round_amount"]:
        assert col in out.columns


def test_log1p_amt_nonnegative():
    df = _make_df()
    out = compute_amount_features(df)
    assert (out["log1p_amt"] >= 0).all()


def test_amt_cents_range():
    df = _make_df()
    out = compute_amount_features(df)
    assert out["amt_cents"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# fit_amount_encoders + compute_spend_spike_features
# ---------------------------------------------------------------------------

def test_encoders_fitted_keys():
    df = _make_df()
    encoders = fit_amount_encoders(df)
    assert "card1_mean" in encoders
    assert "addr1_mean" in encoders
    assert "_global_mean" in encoders


def test_spend_spike_no_categorical_error():
    """
    Regression test: mapping float mean onto a categorical column must not
    raise 'Cannot setitem on a Categorical with a new category'.
    """
    df = _make_df()
    encoders = fit_amount_encoders(df)
    out = compute_spend_spike_features(df, encoders)
    assert "amt_to_card1_mean_ratio" in out.columns
    assert "amt_zscore_card1" in out.columns


def test_zscore_clipped():
    df = _make_df()
    encoders = fit_amount_encoders(df)
    out = compute_spend_spike_features(df, encoders)
    for col in out.columns:
        if "zscore" in col:
            assert out[col].between(-5, 5).all(), f"{col} exceeds [-5, 5]"
