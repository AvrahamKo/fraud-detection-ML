"""
tests/test_velocity.py
======================
Unit tests for src/features/velocity.py
"""
import numpy as np
import pandas as pd

from src.features.velocity import compute_velocity_features


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_df(categorical: bool = False) -> pd.DataFrame:
    df = pd.DataFrame({
        "TransactionID":  [1, 2, 3, 4, 5],
        "TransactionDT":  [0, 1800, 3700, 90_000, 90_100],   # seconds
        "TransactionAmt": [100.0, 200.0, 50.0, 300.0, 75.0],
        "card1": pd.Categorical(["A", "A", "A", "B", "B"])
            if categorical else ["A", "A", "A", "B", "B"],
        "addr1":          ["X", "X", "Y", "X", "Y"],
        "P_emaildomain":  ["g.com", "g.com", "y.com", "g.com", "y.com"],
    })
    return df


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

def test_expected_columns_created():
    df = _make_df()
    out = compute_velocity_features(df, entity_cols=["card1"], windows={"1h": 3600})
    assert "vel_card1_1h_count" in out.columns
    assert "vel_card1_1h_sum"   in out.columns
    assert "vel_card1_1h_mean_amt" in out.columns


def test_27_features_total():
    df = _make_df()
    out = compute_velocity_features(df)
    vel_cols = [c for c in out.columns if c.startswith("vel_")]
    assert len(vel_cols) == 27   # 3 entities × 3 windows × 3 stats


# ---------------------------------------------------------------------------
# closed="left" — current transaction excluded from its own window
# ---------------------------------------------------------------------------

def test_first_transaction_count_is_zero():
    """The very first transaction on a card has no history → count should be 0."""
    df = _make_df()
    out = compute_velocity_features(df, entity_cols=["card1"], windows={"1h": 3600})
    # Row 0: first transaction of card "A" — nothing before it
    assert out.loc[out["TransactionID"] == 1, "vel_card1_1h_count"].iloc[0] == 0.0


def test_window_excludes_current_row():
    """Transaction at t=3700s is outside the 1h window of the tx at t=3700s."""
    df = _make_df()
    out = compute_velocity_features(df, entity_cols=["card1"], windows={"1h": 3600})
    # TX ID=3 (card A, t=3700): only TX at t=1800 is within [100, 3700)
    # TX at t=0 is at delta=3700s — outside the 1h window
    row = out.loc[out["TransactionID"] == 3, "vel_card1_1h_count"].iloc[0]
    assert row == 1.0


# ---------------------------------------------------------------------------
# Categorical entity column — regression test for the NaN sentinel bug
# ---------------------------------------------------------------------------

def test_categorical_entity_no_error():
    """
    Regression: categorical card1 + NaN fillna used to raise
    'Cannot setitem on a Categorical with a new category'.
    """
    df = _make_df(categorical=True)
    # Inject a NaN
    df.loc[0, "card1"] = np.nan
    out = compute_velocity_features(df, entity_cols=["card1"], windows={"1h": 3600})
    assert "vel_card1_1h_count" in out.columns
