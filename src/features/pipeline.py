"""
src/features/pipeline.py
========================
Master feature engineering pipeline.

This module is the single entry point for all downstream consumers
(model training, evaluation, notebooks).  It orchestrates the three
feature modules in the correct order and handles the train/val/test
leakage boundary correctly.

Usage — training workflow
--------------------------
::

    from src.ingestion.splitter import load_splits
    from src.features.pipeline import FeaturePipeline

    train, val, test = load_splits()

    pipeline = FeaturePipeline()
    pipeline.fit(train)                          # fits all encoders on train only

    train_feat = pipeline.transform(train)       # apply to train
    val_feat   = pipeline.transform(val)         # same encoders — no leakage
    test_feat  = pipeline.transform(test)

    # Or in one convenience call:
    train_feat, val_feat, test_feat = pipeline.fit_transform_splits(train, val, test)

Design principles
-----------------
* **Single fit, multiple transforms.**  All stateful encoders (amount means,
  address modes) are fitted once on the training partition and stored in
  ``self.encoders_``.  Calling ``transform`` on val/test uses these stored
  values, preventing any future information from leaking.

* **Idempotent ``transform``.**  The same DataFrame can be passed through
  ``transform`` multiple times without side effects (we always work on a
  copy).

* **Graceful degradation.**  Each feature module guards its own columns.
  If a column is missing (e.g. ``R_emaildomain`` absent in some datasets),
  the module logs a warning and skips that feature rather than raising.

* **Validation.**  After building features, ``_validate_output`` checks for
  unexpected all-NaN columns and logs a warning — useful for catching
  upstream data quality regressions.

Feature inventory (as of current modules)
------------------------------------------
Velocity   : 27 features (3 entities × 3 windows × 3 stats)
Behavioral : 13 features (5 temporal + 3 amount + 5 spend-spike)
Identity   : 13 features (6 email + 4 address + 2 presence)
               Total ≈ 53 engineered features added to the base ~434 columns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.velocity import (
    ENTITY_COLS as _VEL_ENTITIES,
    VELOCITY_WINDOWS as _VEL_WINDOWS,
    compute_velocity_features,
)
from src.features.behavioral import (
    _AMOUNT_ENTITIES as _BEH_ENTITIES,
    compute_behavioral_features,
    fit_amount_encoders,
)
from src.features.identity_mismatch import (
    compute_identity_features,
    fit_address_encoders,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Controls which feature modules are active and their parameters.

    Attributes
    ----------
    run_velocity : bool
        Enable velocity (rolling window) features.  Slowest module.
    run_behavioral : bool
        Enable temporal and spend-spike features.
    run_identity : bool
        Enable email mismatch and address consistency features.
    velocity_entity_cols : list[str]
        Entities for velocity aggregation.
    velocity_windows : dict
        Window label → seconds.
    behavioral_entity_cols : list[str]
        Entities for amount mean/std encoding.
    address_entity_col : str
        Card entity column for address consistency.
    address_cols : tuple[str, ...]
        Address columns to check consistency for.
    label_col : str
        Target column name (excluded from feature matrix).
    """
    run_velocity           : bool       = True
    run_behavioral         : bool       = True
    run_identity           : bool       = True

    velocity_entity_cols   : List[str]  = field(default_factory=lambda: list(_VEL_ENTITIES))
    velocity_windows       : Dict[str, int] = field(default_factory=lambda: dict(_VEL_WINDOWS))

    behavioral_entity_cols : List[str]  = field(default_factory=lambda: list(_BEH_ENTITIES))

    address_entity_col     : str        = "card1"
    address_cols           : Tuple[str, ...] = ("addr1", "addr2")

    label_col              : str        = "isFraud"


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Stateful feature engineering pipeline.

    Attributes
    ----------
    config : PipelineConfig
    encoders_ : dict
        Populated after ``fit()`` is called.  Contains all stateful
        lookups (amount means/stds, address modes/n-uniques).
    feature_names_ : list[str]
        Column names added by the pipeline (populated after first transform).
    is_fitted_ : bool
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config       = config or PipelineConfig()
        self.encoders_    : Dict = {}
        self.feature_names_: List[str] = []
        self.is_fitted_   = False

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit all stateful encoders on the training partition.

        Must be called before ``transform``.  Only training data should be
        passed here — using val/test data will cause data leakage.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split produced by ``splitter.time_aware_split``.

        Returns
        -------
        self
        """
        logger.info("FeaturePipeline.fit() — deriving encoders from training set …")
        cfg = self.config

        # Behavioral amount encoders
        if cfg.run_behavioral:
            logger.info("  Fitting amount encoders …")
            self.encoders_["behavioral"] = fit_amount_encoders(
                train_df, entity_cols=cfg.behavioral_entity_cols
            )

        # Address consistency encoders
        if cfg.run_identity:
            logger.info("  Fitting address encoders …")
            self.encoders_["address"] = fit_address_encoders(
                train_df,
                entity_col=cfg.address_entity_col,
                addr_cols=list(cfg.address_cols),
            )

        self.is_fitted_ = True
        logger.info("FeaturePipeline.fit() complete.")
        return self

    # ------------------------------------------------------------------ #
    # Transform                                                            #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature modules to ``df`` using the fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame
            Any partition (train, val, or test).

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with all engineered features appended.
            Row order and original columns are preserved.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "FeaturePipeline.transform() called before fit(). "
                "Call pipeline.fit(train_df) first."
            )

        cfg  = self.config
        out  = df.copy()
        original_cols = set(df.columns)
        t0   = time.perf_counter()

        # ── 1. Velocity ───────────────────────────────────────────────────
        if cfg.run_velocity:
            logger.info("[1/3] Velocity features …")
            t1  = time.perf_counter()
            out = compute_velocity_features(
                out,
                entity_cols=cfg.velocity_entity_cols,
                windows=cfg.velocity_windows,
            )
            logger.info("  → Done in %.1f s", time.perf_counter() - t1)

        # ── 2. Behavioral ─────────────────────────────────────────────────
        if cfg.run_behavioral:
            logger.info("[2/3] Behavioral features …")
            t1  = time.perf_counter()
            out = compute_behavioral_features(
                out,
                encoders=self.encoders_.get("behavioral"),
                entity_cols=cfg.behavioral_entity_cols,
            )
            logger.info("  → Done in %.1f s", time.perf_counter() - t1)

        # ── 3. Identity / mismatch ────────────────────────────────────────
        if cfg.run_identity:
            logger.info("[3/3] Identity / mismatch features …")
            t1  = time.perf_counter()
            out = compute_identity_features(
                out,
                address_encoders=self.encoders_.get("address"),
                entity_col=cfg.address_entity_col,
                addr_cols=list(cfg.address_cols),
            )
            logger.info("  → Done in %.1f s", time.perf_counter() - t1)

        # ── Post-processing ───────────────────────────────────────────────
        new_cols = [c for c in out.columns if c not in original_cols]
        self.feature_names_ = new_cols  # update registry

        self._validate_output(out, new_cols)

        logger.info(
            "FeaturePipeline.transform() complete: added %d features in %.1f s total.",
            len(new_cols), time.perf_counter() - t0,
        )
        return out

    # ------------------------------------------------------------------ #
    # Convenience method: fit + transform all splits at once              #
    # ------------------------------------------------------------------ #

    def fit_transform_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit on train, then transform train / val / test in one call.

        This is the recommended workflow for notebooks and scripts.

        Parameters
        ----------
        train, val, test : pd.DataFrame

        Returns
        -------
        train_feat, val_feat, test_feat : pd.DataFrame
        """
        self.fit(train)
        logger.info("Transforming train split …")
        train_feat = self.transform(train)
        logger.info("Transforming val split …")
        val_feat   = self.transform(val)
        logger.info("Transforming test split …")
        test_feat  = self.transform(test)
        return train_feat, val_feat, test_feat

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _validate_output(self, df: pd.DataFrame, new_cols: List[str]) -> None:
        """Log warnings for any engineered column that is entirely NaN."""
        all_null = [c for c in new_cols if df[c].isnull().all()]
        if all_null:
            logger.warning(
                "The following engineered columns are entirely NaN — "
                "check upstream data: %s", all_null
            )

        high_null = [
            c for c in new_cols
            if 0 < df[c].isnull().mean() > 0.5
        ]
        if high_null:
            logger.debug(
                "Engineered columns with >50%% null (may be expected for "
                "sparse entities): %s", high_null
            )

    def feature_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of all engineered features.

        Useful for quick inspection and for building feature lists for
        model training.

        Returns
        -------
        pd.DataFrame
            Columns: feature_name, module.
        """
        rows = []
        for col in self.feature_names_:
            if col.startswith("vel_"):
                module = "velocity"
            elif col in (
                "hour_of_day", "day_of_week", "day_of_dataset",
                "is_night", "is_weekend",
                "log1p_amt", "amt_cents", "is_round_amount",
            ) or "ratio" in col or "zscore" in col:
                module = "behavioral"
            else:
                module = "identity_mismatch"
            rows.append({"feature_name": col, "module": module})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Functional convenience wrapper (for notebooks / quick scripts)
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    config: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: fit on ``train_df`` (or ``df`` if not provided),
    transform ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to transform.
    train_df : pd.DataFrame, optional
        Training partition used to fit encoders.  If None, ``df`` is used
        for fitting (correct only when ``df`` IS the training set).
    config : PipelineConfig, optional

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.

    Examples
    --------
    >>> from src.features.pipeline import build_features
    >>> # Training set — fit and transform in one step
    >>> train_feat = build_features(train, train_df=train)
    >>> # Validation — use training encoders
    >>> val_feat   = build_features(val, train_df=train)
    """
    pipeline = FeaturePipeline(config=config)
    pipeline.fit(train_df if train_df is not None else df)
    return pipeline.transform(df)


# ---------------------------------------------------------------------------
# CLI entry point — run the full pipeline and persist enriched splits
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.ingestion.splitter import load_splits, save_splits

    _PROJECT_ROOT  = Path(__file__).resolve().parents[2]
    _PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
    _FEATURE_DIR   = _PROJECT_ROOT / "data" / "processed" / "features"

    logger.info("Loading pre-split data …")
    train, val, test = load_splits(_PROCESSED_DIR)

    logger.info("Running FeaturePipeline …")
    pipeline = FeaturePipeline()
    train_f, val_f, test_f = pipeline.fit_transform_splits(train, val, test)

    logger.info("Persisting enriched splits to %s …", _FEATURE_DIR)
    save_splits(train_f, val_f, test_f, out_dir=_FEATURE_DIR)

    summary = pipeline.feature_summary()
    print("\n=== Feature inventory ===")
    print(summary.groupby("module").size().rename("count").to_string())
    print(f"\nTotal engineered features: {len(summary)}")
    print(f"Total columns in output  : {train_f.shape[1]}")
