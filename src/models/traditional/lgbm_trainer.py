"""
src/models/traditional/lgbm_trainer.py
=======================================
LightGBM fraud classifier with Optuna hyperparameter tuning.

Design philosophy
-----------------
Fraud detection is a **cost-asymmetric decision problem**, not a standard
classification task.  The cost of missing a fraud (false negative) is
typically 10–100× the cost of declining a good transaction (false positive).
This drives every modelling decision here:

1. **PR-AUC as the target metric.**  ROC-AUC is misleading at 3.5% fraud
   rate — a model that predicts "legitimate" for everything achieves ~0.96
   ROC-AUC.  PR-AUC (Average Precision) punishes poor recall at high
   precision directly, which is what a fraud operations team cares about.

2. **Custom LightGBM eval function on PR-AUC.**  Early stopping monitors
   PR-AUC on the validation set, not log-loss.  This aligns what the
   training loop optimises with what the Optuna objective maximises.

3. **`scale_pos_weight`** upweights fraud examples so the gradient boosting
   updates treat each fraud transaction as ≈28× more important than a legit
   one.  Optuna can also search around this baseline.

4. **Strict fit / predict separation.**  The trainer class never touches the
   test set.  ``predict_proba`` accepts any feature matrix.

5. **SHAP via TreeExplainer.**  LightGBM's native ``pred_contrib=True`` mode
   is fast but returns raw additive contributions.  The ``shap`` library's
   TreeExplainer wraps this cleanly and produces Explanation objects that
   work with all SHAP visualisation functions.

Module layout
-------------
``TrainerConfig``           — dataclass of all tunable knobs
``prepare_features``        — column selection & type coercion
``lgb_pr_auc_metric``       — custom LightGBM eval function
``LGBMFraudTrainer``        — stateful trainer class
    .tune()                 — Optuna study
    .train_final()          — single train run with best params
    .fit()                  — tune + train in one call
    .predict_proba()
    .get_feature_importance()
    .compute_shap_values()
    .save() / .load()
CLI                         — run as ``python -m src.models.traditional.lgbm_trainer``
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial INFO spam — we log our own summaries
optuna.logging.set_verbosity(optuna.logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR   = _PROJECT_ROOT / "models"

# Columns that are identifiers or targets — never fed to the model
_EXCLUDED_COLS: List[str] = ["TransactionID", "isFraud", "TransactionDT"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    All knobs for the LightGBM trainer.

    Attributes
    ----------
    n_trials : int
        Number of Optuna trials for hyperparameter search.
    n_estimators : int
        Maximum number of boosting rounds (early stopping will cut this short).
    early_stopping_rounds : int
        Stop if PR-AUC on val hasn't improved for this many rounds.
    log_eval_period : int
        Print training progress every N rounds (0 = silent).
    scale_pos_weight_base : float
        Starting point for imbalance weighting.  If None, computed from data
        as ``n_negative / n_positive``.  Optuna searches ±50% around this.
    n_jobs : int
        LightGBM threads.  -1 = all cores.
    random_seed : int
    model_name : str
        Used for file names when saving artefacts.
    """
    n_trials              : int            = 10
    n_estimators          : int            = 300
    early_stopping_rounds : int            = 15
    log_eval_period       : int            = 50
    scale_pos_weight_base : Optional[float]= None   # auto-computed if None
    n_jobs                : int            = -1
    random_seed           : int            = 42
    model_name            : str            = "lgbm_fraud"

    # Fixed LightGBM params that are NOT tuned by Optuna
    fixed_params: Dict = field(default_factory=lambda: {
        "objective"      : "binary",
        "metric"         : "custom",     # we supply our own eval fn
        "boosting_type"  : "gbdt",
        "verbosity"      : -1,
        "is_unbalance"   : False,        # we use scale_pos_weight instead
    })


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    label_col: str = "isFraud",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separate and coerce features for LightGBM.

    Steps
    -----
    1. Drop identifier / target / explicitly excluded columns.
    2. Object-dtype columns (e.g. ``p_email_tld``) are label-encoded to
       int so LightGBM can treat them as categoricals natively.
    3. Return the feature matrix ``X``, target ``y``, and the list of
       categorical feature column names.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched split produced by ``FeaturePipeline.transform()``.
    label_col : str
        Target column name.
    drop_cols : list of str, optional
        Additional columns to exclude.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    categorical_feature_names : list of str
        Subset of ``X.columns`` that LightGBM should treat as categorical.
    """
    exclude = set(_EXCLUDED_COLS + (drop_cols or []))

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    y = df[label_col].astype(np.int8)
    feature_cols = [c for c in df.columns if c not in exclude and c != label_col]
    X = df[feature_cols].copy()

    # ── Identify and coerce categorical columns ────────────────────────────
    cat_cols = []

    for col in X.columns:
        if X[col].dtype.name == "category":
            # Convert pandas category → integer codes.
            # LightGBM handles integer-coded categoricals natively when the
            # column is listed in ``categorical_feature``.
            X[col] = X[col].cat.codes.astype(np.int16)
            cat_cols.append(col)

        elif X[col].dtype == object:
            # Remaining string columns (e.g. email TLD strings from identity module)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str)).astype(np.int16)
            cat_cols.append(col)

    logger.info(
        "prepare_features: %d total features  |  %d categorical  |  fraud rate: %.4f%%",
        len(feature_cols), len(cat_cols), y.mean() * 100,
    )
    return X, y, cat_cols


# ---------------------------------------------------------------------------
# Custom LightGBM PR-AUC metric
# ---------------------------------------------------------------------------

def lgb_pr_auc_metric(
    y_pred: np.ndarray,
    dataset: lgb.Dataset,
) -> Tuple[str, float, bool]:
    """
    Custom evaluation metric for LightGBM: PR-AUC (Average Precision).

    Registered with LightGBM via ``feval=lgb_pr_auc_metric``.
    Early stopping monitors this value directly.

    Returns
    -------
    tuple
        ``(metric_name, value, is_higher_better)``
    """
    y_true = dataset.get_label()
    score  = average_precision_score(y_true, y_pred)
    return "pr_auc", score, True   # higher is better → early stopping maximises


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LGBMFraudTrainer:
    """
    Stateful LightGBM trainer for fraud detection.

    Workflow
    --------
    ::

        trainer = LGBMFraudTrainer(config)
        X_train, y_train, cat_cols = prepare_features(train_feat)
        X_val,   y_val,   _        = prepare_features(val_feat)

        # Option A: tune + train in one call
        trainer.fit(X_train, y_train, X_val, y_val, cat_cols)

        # Option B: tune first, inspect, then train
        study  = trainer.tune(X_train, y_train, X_val, y_val, cat_cols)
        model  = trainer.train_final(X_train, y_train, X_val, y_val,
                                     cat_cols, study.best_params)

        # Predict
        proba  = trainer.predict_proba(X_test)

        # Explain
        imp    = trainer.get_feature_importance()
        shap_v = trainer.compute_shap_values(X_val.sample(2000))

        # Persist
        trainer.save(_MODELS_DIR / "lgbm_v1")

    Attributes
    ----------
    config : TrainerConfig
    model_ : lgb.Booster or None
        Populated after ``train_final`` / ``fit``.
    study_ : optuna.Study or None
        Populated after ``tune`` / ``fit``.
    best_params_ : dict or None
        Best Optuna params merged with fixed params.
    feature_names_ : list[str]
        Column names of the training feature matrix.
    cat_feature_names_ : list[str]
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config              = config or TrainerConfig()
        self.model_              : Optional[lgb.Booster]  = None
        self.study_              : Optional[optuna.Study] = None
        self.best_params_        : Optional[Dict]         = None
        self.feature_names_      : List[str]              = []
        self.cat_feature_names_  : List[str]              = []
        self._scale_pos_weight   : float                  = 1.0
        self._label_encoders     : Dict                   = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        """Compute class imbalance ratio (n_negative / n_positive)."""
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        spw   = n_neg / max(n_pos, 1)
        logger.info(
            "scale_pos_weight: %.2f  (n_neg=%d, n_pos=%d)",
            spw, int(n_neg), int(n_pos),
        )
        return float(spw)

    def _build_datasets(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val  : pd.DataFrame,
        y_val  : pd.Series,
        cat_cols: List[str],
    ) -> Tuple[lgb.Dataset, lgb.Dataset]:
        """Construct LightGBM Dataset objects."""
        dtrain = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val, label=y_val,
            reference=dtrain,          # ensures same binning as train
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        return dtrain, dval

    def _merge_params(self, optuna_params: Dict) -> Dict:
        """Merge fixed config params with Optuna-suggested params."""
        merged = dict(self.config.fixed_params)
        merged.update(optuna_params)
        merged["seed"]    = self.config.random_seed
        merged["n_jobs"]  = self.config.n_jobs
        return merged

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning                                                #
    # ------------------------------------------------------------------ #

    def tune(
        self,
        X_train  : pd.DataFrame,
        y_train  : pd.Series,
        X_val    : pd.DataFrame,
        y_val    : pd.Series,
        cat_cols : List[str],
    ) -> optuna.Study:
        """
        Run an Optuna hyperparameter search and return the completed study.

        The objective function trains a LightGBM model for each trial with
        early stopping and returns the best PR-AUC on the validation set.

        Search space
        ------------
        - ``num_leaves``         : 31 – 512
        - ``max_depth``          : 4 – 12  (-1 = unlimited not searched)
        - ``learning_rate``      : 5e-3 – 0.3  (log scale)
        - ``feature_fraction``   : 0.4 – 1.0
        - ``bagging_fraction``   : 0.4 – 1.0
        - ``bagging_freq``       : 1 – 7
        - ``min_child_samples``  : 20 – 300
        - ``reg_alpha``          : 1e-8 – 10.0  (log scale)
        - ``reg_lambda``         : 1e-8 – 10.0  (log scale)
        - ``scale_pos_weight``   : [spw × 0.5, spw × 2.0]

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : validation data (used for early stopping + objective)
        cat_cols         : categorical feature names

        Returns
        -------
        optuna.Study
        """
        self._scale_pos_weight = (
            self.config.scale_pos_weight_base
            or self._compute_scale_pos_weight(y_train)
        )
        self.feature_names_     = list(X_train.columns)
        self.cat_feature_names_ = cat_cols

        dtrain, dval = self._build_datasets(X_train, y_train, X_val, y_val, cat_cols)

        spw_base = self._scale_pos_weight

        def objective(trial: optuna.Trial) -> float:
            params = self._merge_params({
                "num_leaves"       : trial.suggest_int("num_leaves", 31, 512),
                "max_depth"        : trial.suggest_int("max_depth", 4, 12),
                "learning_rate"    : trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                "feature_fraction" : trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq"     : trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight" : trial.suggest_float(
                    "scale_pos_weight", spw_base * 0.5, spw_base * 2.0
                ),
            })

            # Per-trial caps: kept deliberately low so each trial finishes
            # in seconds, not minutes.  The final model (train_final) uses
            # the full n_estimators budget with a relaxed early stopping.
            _TRIAL_MAX_ROUNDS   = 300
            _TRIAL_EARLY_STOP   = 15

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=_TRIAL_EARLY_STOP,
                    verbose=False,
                ),
                lgb.log_evaluation(period=-1),   # silent per-trial
            ]

            model = lgb.train(
                params,
                dtrain,
                num_boost_round  = _TRIAL_MAX_ROUNDS,
                valid_sets       = [dval],
                valid_names      = ["val"],
                feval            = lgb_pr_auc_metric,
                callbacks        = callbacks,
            )

            # best_score is keyed by [valid_name][metric_name]
            best_pr_auc = model.best_score["val"]["pr_auc"]

            # Prune unpromising trials early
            trial.report(best_pr_auc, step=model.best_iteration)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return best_pr_auc

        study = optuna.create_study(
            direction  = "maximize",
            study_name = self.config.model_name,
            pruner     = optuna.pruners.MedianPruner(n_warmup_steps=10),
            sampler    = optuna.samplers.TPESampler(seed=self.config.random_seed),
        )

        logger.info(
            "Optuna: starting %d trials  |  target = PR-AUC on val  …",
            self.config.n_trials,
        )
        t0 = time.perf_counter()

        study.optimize(
            objective,
            n_trials        = self.config.n_trials,
            show_progress_bar=True,
            callbacks        = [_optuna_logging_callback],
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Optuna: finished %d trials in %.1f s  |  best PR-AUC: %.6f",
            len(study.trials), elapsed, study.best_value,
        )
        logger.info("Best params: %s", json.dumps(study.best_params, indent=2))

        self.study_ = study
        return study

    # ------------------------------------------------------------------ #
    # Final model training                                                 #
    # ------------------------------------------------------------------ #

    def train_final(
        self,
        X_train  : pd.DataFrame,
        y_train  : pd.Series,
        X_val    : pd.DataFrame,
        y_val    : pd.Series,
        cat_cols : List[str],
        best_params: Optional[Dict] = None,
    ) -> lgb.Booster:
        """
        Train the final model with a given parameter set and early stopping.

        If ``best_params`` is None, uses ``self.study_.best_params`` (requires
        ``tune()`` to have been called first).

        Parameters
        ----------
        X_train, y_train, X_val, y_val : data splits
        cat_cols                        : categorical feature names
        best_params : dict, optional
            Optuna-suggested params.  Merged with ``TrainerConfig.fixed_params``.

        Returns
        -------
        lgb.Booster
        """
        if best_params is None:
            if self.study_ is None:
                raise RuntimeError(
                    "No best_params provided and tune() has not been called. "
                    "Either call tune() first or pass best_params explicitly."
                )
            best_params = self.study_.best_params

        self.feature_names_     = list(X_train.columns)
        self.cat_feature_names_ = cat_cols
        self._scale_pos_weight  = (
            self.config.scale_pos_weight_base
            or self._compute_scale_pos_weight(y_train)
        )

        params = self._merge_params(best_params)
        dtrain, dval = self._build_datasets(X_train, y_train, X_val, y_val, cat_cols)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.early_stopping_rounds,
                verbose=True,
            ),
            lgb.log_evaluation(period=self.config.log_eval_period),
        ]

        logger.info("Training final model with best params …")
        t0 = time.perf_counter()

        model = lgb.train(
            params,
            dtrain,
            num_boost_round = self.config.n_estimators,
            valid_sets      = [dval],
            valid_names     = ["val"],
            feval           = lgb_pr_auc_metric,
            callbacks       = callbacks,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Final model: %d trees  |  best iteration=%d  |  "
            "val PR-AUC=%.6f  |  %.1f s",
            model.num_trees(),
            model.best_iteration,
            model.best_score["val"]["pr_auc"],
            elapsed,
        )

        self.model_       = model
        self.best_params_ = params
        return model

    # ------------------------------------------------------------------ #
    # Fit: tune + train in one call                                        #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train  : pd.DataFrame,
        y_train  : pd.Series,
        X_val    : pd.DataFrame,
        y_val    : pd.Series,
        cat_cols : List[str],
    ) -> "LGBMFraudTrainer":
        """
        Full workflow: tune hyperparameters, then train the final model.

        Parameters
        ----------
        X_train, y_train, X_val, y_val : train / validation splits
        cat_cols : categorical feature name list

        Returns
        -------
        self
        """
        self.tune(X_train, y_train, X_val, y_val, cat_cols)
        self.train_final(X_train, y_train, X_val, y_val, cat_cols)
        return self

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return fraud probability scores ∈ [0, 1] for each row of ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns as training data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probability of fraud (positive class).
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained yet. Call fit() or train_final().")

        # Align columns to training order; fill any missing with 0
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.model_.predict(X_aligned, num_iteration=self.model_.best_iteration)

    # ------------------------------------------------------------------ #
    # Explainability                                                        #
    # ------------------------------------------------------------------ #

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of LightGBM native feature importances.

        Parameters
        ----------
        importance_type : str
            ``"gain"`` (default) — total information gain contributed by
            each feature across all splits.  More interpretable than ``"split"``.
            ``"split"`` — number of times a feature is used in a split.
        top_n : int
            Return only the top-N most important features.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``, ``rank``.
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        imp = pd.DataFrame({
            "feature"   : self.model_.feature_name(),
            "importance": self.model_.feature_importance(importance_type=importance_type),
        })
        imp = (
            imp.sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        imp["rank"] = imp.index + 1
        imp["importance_pct"] = (
            imp["importance"] / imp["importance"].sum() * 100
        ).round(2)

        logger.info(
            "Top 5 features by %s: %s",
            importance_type,
            imp["feature"].head(5).tolist(),
        )
        return imp

    def compute_shap_values(
        self,
        X       : pd.DataFrame,
        n_sample: int = 5_000,
        seed    : int = 42,
    ):
        """
        Compute SHAP values using the ``shap`` library's TreeExplainer.

        SHAP (SHapley Additive exPlanations) values measure each feature's
        additive contribution to the model output for each individual
        prediction.  This is the gold standard for post-hoc explainability
        in fraud review workflows.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (val or test split).
        n_sample : int
            Number of rows to compute SHAP values for.  Full dataset can be
            slow — 5k rows is enough for a representative analysis.
        seed : int

        Returns
        -------
        shap.Explanation
            Contains ``.values``, ``.base_values``, ``.data``.
            Pass directly to ``shap.summary_plot()``, ``shap.waterfall_plot()``, etc.

        Examples
        --------
        >>> import shap
        >>> sv = trainer.compute_shap_values(X_val)
        >>> shap.summary_plot(sv, max_display=20)
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "The 'shap' package is required for SHAP analysis. "
                "Install it with: pip install shap"
            )

        if self.model_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        if len(X_aligned) > n_sample:
            X_aligned = X_aligned.sample(n=n_sample, random_state=seed)
            logger.info("SHAP: sampled %d rows from %d total.", n_sample, len(X))

        logger.info("SHAP: building TreeExplainer and computing values …")
        t0 = time.perf_counter()

        explainer  = shap.TreeExplainer(self.model_)
        shap_vals  = explainer(X_aligned)

        logger.info("SHAP: done in %.1f s.", time.perf_counter() - t0)
        return shap_vals

    # ------------------------------------------------------------------ #
    # Evaluation summary                                                   #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        X    : pd.DataFrame,
        y    : pd.Series,
        split: str = "eval",
    ) -> Dict[str, float]:
        """
        Compute key fraud detection metrics on a labelled split.

        Metrics
        -------
        - PR-AUC (Average Precision) — primary metric
        - ROC-AUC
        - Precision / Recall / F1 at the 0.5 decision threshold
          (threshold tuning lives in ``src/evaluation/threshold.py``)

        Parameters
        ----------
        X     : feature matrix
        y     : true labels
        split : label for logging

        Returns
        -------
        dict
        """
        from sklearn.metrics import (
            f1_score, precision_score, recall_score,
        )

        proba = self.predict_proba(X)
        pred  = (proba >= 0.5).astype(int)

        metrics = {
            "pr_auc" : round(average_precision_score(y, proba), 6),
            "roc_auc": round(roc_auc_score(y, proba),            6),
            "precision_at_50": round(precision_score(y, pred, zero_division=0), 4),
            "recall_at_50"   : round(recall_score(y, pred,    zero_division=0), 4),
            "f1_at_50"       : round(f1_score(y, pred,        zero_division=0), 4),
        }

        logger.info(
            "[%s]  PR-AUC=%.4f  ROC-AUC=%.4f  Prec@0.5=%.4f  "
            "Recall@0.5=%.4f  F1@0.5=%.4f",
            split,
            metrics["pr_auc"], metrics["roc_auc"],
            metrics["precision_at_50"], metrics["recall_at_50"],
            metrics["f1_at_50"],
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, dir_path: Path) -> None:
        """
        Persist all artefacts to ``dir_path``.

        Saved files
        -----------
        ``model.txt``           — LightGBM native text format (human-readable,
                                  version-stable, can be loaded by the C++ lib)
        ``trainer_meta.pkl``    — Python objects: config, feature names,
                                  encoders, scale_pos_weight
        ``optuna_study.pkl``    — Optuna Study object (contains all trial history)
        ``best_params.json``    — Best params as JSON for quick inspection
        ``feature_importance_gain.csv``  — Ranked importance table
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.model_ is None:
            raise RuntimeError("Nothing to save — model has not been trained.")

        # ── LightGBM model ────────────────────────────────────────────────
        model_path = dir_path / "model.txt"
        self.model_.save_model(str(model_path))
        logger.info("Saved LightGBM model → %s", model_path)

        # ── Trainer metadata (JSON, not pickle) ───────────────────────────
        # Storing as JSON prevents the __main__.TrainerConfig pickle trap:
        # if the trainer was run as __main__, pickle records the class as
        # __main__.TrainerConfig, which becomes unresolvable when loaded
        # from a different entry point (e.g. the evaluator).
        # Plain JSON has no class-path dependency.
        meta = {
            "config_dict"        : vars(self.config),
            "feature_names_"     : self.feature_names_,
            "cat_feature_names_" : self.cat_feature_names_,
            "_scale_pos_weight"  : self._scale_pos_weight,
        }
        meta_path = dir_path / "trainer_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Saved trainer metadata → %s", meta_path)

        # Remove any stale .pkl written by a previous save() so load()
        # doesn't accidentally fall back to it.
        stale_pkl = dir_path / "trainer_meta.pkl"
        if stale_pkl.exists():
            stale_pkl.unlink()
            logger.info("Removed stale trainer_meta.pkl")

        # ── Optuna study ──────────────────────────────────────────────────
        if self.study_ is not None:
            study_path = dir_path / "optuna_study.pkl"
            with open(study_path, "wb") as f:
                pickle.dump(self.study_, f)
            logger.info("Saved Optuna study → %s", study_path)

        # ── Best params as JSON ───────────────────────────────────────────
        if self.best_params_ is not None:
            params_path = dir_path / "best_params.json"
            with open(params_path, "w") as f:
                json.dump(self.best_params_, f, indent=2, default=str)
            logger.info("Saved best params → %s", params_path)

        # ── Feature importance ────────────────────────────────────────────
        imp_df   = self.get_feature_importance(importance_type="gain", top_n=200)
        imp_path = dir_path / "feature_importance_gain.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info("Saved feature importance → %s", imp_path)

        logger.info("All artefacts saved to %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "LGBMFraudTrainer":
        """
        Re-hydrate a trainer from a previously saved directory.

        Parameters
        ----------
        dir_path : Path
            Directory produced by ``save()``.

        Returns
        -------
        LGBMFraudTrainer
            Fully restored trainer ready for ``predict_proba`` and
            ``compute_shap_values``.
        """
        dir_path   = Path(dir_path)
        model_path = dir_path / "model.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"model.txt not found in {dir_path}")

        # Prefer the JSON metadata (no pickle class-path dependency).
        # Fall back to legacy .pkl for backwards compatibility.
        json_meta_path = dir_path / "trainer_meta.json"
        pkl_meta_path  = dir_path / "trainer_meta.pkl"

        if json_meta_path.exists():
            with open(json_meta_path) as f:
                meta = json.load(f)
            # Reconstruct TrainerConfig from the stored plain dict.
            # Only pass fields that the current dataclass actually declares
            # (guards against version drift between save and load).
            valid_fields = set(TrainerConfig.__dataclass_fields__.keys())
            config_kwargs = {
                k: v for k, v in meta["config_dict"].items()
                if k in valid_fields
            }
            config = TrainerConfig(**config_kwargs)
        elif pkl_meta_path.exists():
            logger.warning(
                "Found legacy trainer_meta.pkl — attempting to load it. "
                "Re-save the model to upgrade to the JSON format."
            )
            try:
                with open(pkl_meta_path, "rb") as f:
                    meta = pickle.load(f)
                config = meta["config"]
            except (AttributeError, ImportError) as exc:
                # Stale pickle was saved from a __main__ context, so the
                # class is recorded as __main__.TrainerConfig and can't be
                # found from any other entry point.  Recover gracefully:
                # reconstruct config from defaults, pull feature names from
                # the LightGBM model file (which stores them natively).
                logger.warning(
                    "Could not deserialize TrainerConfig from pkl (%s). "
                    "Reconstructing from defaults + model file. "
                    "Delete trainer_meta.pkl and re-run training to make "
                    "this warning go away permanently.", exc,
                )
                config  = TrainerConfig()
                _booster = lgb.Booster(model_file=str(model_path))
                meta = {
                    "feature_names_"    : _booster.feature_name(),
                    "cat_feature_names_": [],
                    "_scale_pos_weight" : 1.0,
                }
        else:
            raise FileNotFoundError(
                f"Neither trainer_meta.json nor trainer_meta.pkl found in {dir_path}"
            )

        trainer                    = cls(config=config)
        trainer.feature_names_     = meta["feature_names_"]
        trainer.cat_feature_names_ = meta["cat_feature_names_"]
        trainer._scale_pos_weight  = meta["_scale_pos_weight"]
        trainer.model_             = lgb.Booster(model_file=str(model_path))

        study_path = dir_path / "optuna_study.pkl"
        if study_path.exists():
            with open(study_path, "rb") as f:
                trainer.study_ = pickle.load(f)

        params_path = dir_path / "best_params.json"
        if params_path.exists():
            with open(params_path) as f:
                trainer.best_params_ = json.load(f)

        logger.info("Trainer loaded from %s", dir_path)
        return trainer


# ---------------------------------------------------------------------------
# Optuna callback — logs best value every 10 trials
# ---------------------------------------------------------------------------

def _optuna_logging_callback(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """Log a progress line every 10 completed trials."""
    n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n % 10 == 0 and n > 0:
        logger.info(
            "Optuna progress: %d trials complete  |  best PR-AUC so far: %.6f",
            n, study.best_value,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.ingestion.splitter import load_splits
    from src.features.pipeline import FeaturePipeline

    _PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
    _FEATURE_DIR   = _PROCESSED_DIR / "features"
    _OUTPUT_DIR    = _MODELS_DIR / "lgbm_v1"

    # ── Load data ─────────────────────────────────────────────────────────
    # Try pre-built feature splits first; fall back to raw splits + pipeline
    if (_FEATURE_DIR / "train.parquet").exists():
        logger.info("Loading pre-built feature splits …")
        train_f, val_f, test_f = load_splits(_FEATURE_DIR)
    else:
        logger.info("Feature splits not found — running pipeline …")
        train_r, val_r, test_r = load_splits(_PROCESSED_DIR)
        pipe = FeaturePipeline()
        train_f, val_f, test_f = pipe.fit_transform_splits(train_r, val_r, test_r)

    # ── Prepare feature matrices ──────────────────────────────────────────
    X_train, y_train, cat_cols = prepare_features(train_f)
    X_val,   y_val,   _        = prepare_features(val_f)
    X_test,  y_test,  _        = prepare_features(test_f)

    logger.info(
        "Feature matrix: train=%s  val=%s  test=%s  cat_features=%d",
        X_train.shape, X_val.shape, X_test.shape, len(cat_cols),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    config  = TrainerConfig(n_trials=10, n_estimators=300)
    trainer = LGBMFraudTrainer(config)
    trainer.fit(X_train, y_train, X_val, y_val, cat_cols)

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_metrics  = trainer.evaluate(X_val,  y_val,  split="val")
    test_metrics = trainer.evaluate(X_test, y_test, split="test")

    print("\n=== Validation metrics ===")
    for k, v in val_metrics.items():
        print(f"  {k:<25} {v}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k:<25} {v}")

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save(_OUTPUT_DIR)
    print(f"\nAll artefacts saved to: {_OUTPUT_DIR}")
