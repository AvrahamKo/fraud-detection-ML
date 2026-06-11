"""
src/evaluation/evaluator.py
============================
Head-to-head evaluation of the LightGBM and Bayesian Logistic Regression
fraud models on the held-out test partition.

Evaluation philosophy
---------------------
A fraud operations team cares about three things, in priority order:

1. **Catch rate** — what fraction of fraud do we stop before it clears?
   (Recall at the operating threshold.)

2. **False positive rate** — what fraction of good customers do we
   incorrectly decline?  Each false positive is a declined sale, a
   support ticket, and a potential churn event.  Measured by precision
   at the operating threshold and the PR curve shape.

3. **Confidence in the score** — can we trust the model's probability
   output for downstream decisioning?  Calibration + uncertainty.

This module measures all three for both models and produces:

Figures (saved to ``reports/figures/``)
----------------------------------------
* ``fig_01_pr_roc_curves.png``          — PR + ROC side-by-side
* ``fig_02_uncertainty_distribution.png`` — Bayesian uncertainty by class
* ``fig_03_lgbm_vs_bayes_scatter.png``  — Model agreement / disagreement
* ``fig_04_fn_recovery_curve.png``      — FN recovery vs. review queue size
* ``fig_05_calibration.png``            — Reliability diagrams

Artefacts (saved to ``reports/``)
-----------------------------------
* ``model_comparison_metrics.json``     — All metrics + analysis tables

Key analytical insight — the Forter differentiator
---------------------------------------------------
LightGBM scores every transaction with a point estimate and no notion
of confidence.  A score of 0.35 could mean "definitely borderline" or
"I've never seen a pattern like this".  These two situations call for
different responses:
  * Borderline but familiar → apply business rule / score band
  * High uncertainty         → route to human review regardless of score

``analyze_fn_recovery()`` quantifies this: among all fraud transactions
that LightGBM missed (false negatives), what fraction also had high
Bayesian epistemic uncertainty?  If the answer is "60% of LightGBM's
missed fraud transactions had uncertainty in the top quartile", that is a
compelling, concrete argument for a hybrid decisioning system at Forter.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_FIGURES_DIR   = _PROJECT_ROOT / "reports" / "figures"
_REPORTS_DIR   = _PROJECT_ROOT / "reports"

# ── Colour palette (consistent across all figures) ────────────────────────
_C_LGBM  = "#1565C0"   # dark blue  — LightGBM
_C_BAYES = "#E65100"   # deep orange — Bayesian
_C_BASE  = "#757575"   # grey        — baseline / random


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    """
    Paths and thresholds for the evaluation run.

    Attributes
    ----------
    lgbm_model_dir : Path
        Directory produced by ``LGBMFraudTrainer.save()``.
    bayes_model_dir : Path
        Directory produced by ``BayesianFraudModel.save()``.
    feature_splits_dir : Path
        Directory containing ``train/val/test.parquet`` enriched by the
        feature pipeline.  Falls back to raw splits + pipeline if absent.
    raw_splits_dir : Path
        Raw splits (pre-feature-engineering) used as fallback.
    figures_dir : Path
    reports_dir : Path
    label_col : str
    decision_threshold : float
        Operating threshold for computing precision/recall/F1 statistics.
        0.5 is reported, but ``analyze_fn_recovery`` sweeps over many.
    uncertainty_percentiles : list of float
        Percentiles of the HDI-width distribution used to define
        "high uncertainty" review queues (e.g. 0.75 = top quartile).
    disagreement_threshold : float
        |lgbm_score − bayes_mean| above which a transaction is called
        "disagreement" and analysed separately.
    dpi : int
        Figure DPI (150 = good quality without huge file sizes).
    """
    lgbm_model_dir       : Path  = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "lgbm_v1"
    )
    bayes_model_dir      : Path  = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "bayes_lr_v1"
    )
    feature_splits_dir   : Path  = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "features"
    )
    raw_splits_dir       : Path  = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed"
    )
    figures_dir          : Path  = field(default_factory=lambda: _FIGURES_DIR)
    reports_dir          : Path  = field(default_factory=lambda: _REPORTS_DIR)
    label_col            : str   = "isFraud"
    decision_threshold   : float = 0.5
    uncertainty_percentiles: List[float] = field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    )
    disagreement_threshold : float = 0.20
    dpi                  : int   = 150


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def _optimal_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Return (threshold, f1) at the point maximising F1 on the PR curve."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve appends a final point with no threshold
    f1_scores = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
    best_idx  = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def _ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between the
    cumulative score distributions of fraud and legitimate transactions.

    KS is widely used by payments risk teams alongside Gini coefficient
    (= 2×ROC-AUC − 1) as a model rank-ordering quality metric.
    """
    fraud_scores = np.sort(y_proba[y_true == 1])
    legit_scores = np.sort(y_proba[y_true == 0])
    # Evaluate CDFs on a common grid
    grid = np.linspace(0, 1, 1000)
    cdf_fraud = np.searchsorted(fraud_scores, grid, side="right") / len(fraud_scores)
    cdf_legit = np.searchsorted(legit_scores, grid, side="right") / len(legit_scores)
    return float(np.max(np.abs(cdf_fraud - cdf_legit)))


def _brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probability and true label."""
    return float(np.mean((y_proba - y_true) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE): weighted mean absolute difference
    between predicted probability and actual fraction of positives in each
    probability bin.

    A perfectly calibrated model has ECE = 0.  Values > 0.05 indicate
    meaningful over/under-confidence.
    """
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    n           = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask   = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc    = y_true[mask].mean()         # actual fraud rate in bin
        conf   = y_proba[mask].mean()        # mean predicted probability
        weight = mask.sum() / n
        ece   += weight * abs(acc - conf)

    return float(ece)


def compute_all_metrics(
    y_true : np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> Dict:
    """
    Compute the full metrics suite for one model.

    Returns a flat dict suitable for JSON serialisation.
    """
    opt_thresh, opt_f1 = _optimal_threshold_f1(y_true, y_proba)
    y_pred_50  = (y_proba >= 0.5).astype(int)
    y_pred_opt = (y_proba >= opt_thresh).astype(int)

    return {
        "model"                : model_name,
        # ── Discrimination ────────────────────────────────────────────────
        "pr_auc"               : round(average_precision_score(y_true, y_proba), 6),
        "roc_auc"              : round(roc_auc_score(y_true, y_proba), 6),
        "gini"                 : round(2 * roc_auc_score(y_true, y_proba) - 1, 6),
        "ks_statistic"         : round(_ks_statistic(y_true, y_proba), 6),
        # ── Calibration ───────────────────────────────────────────────────
        "brier_score"          : round(_brier_score(y_true, y_proba), 6),
        "ece"                  : round(_expected_calibration_error(y_true, y_proba), 6),
        # ── Threshold @ 0.5 ──────────────────────────────────────────────
        "precision_at_50"      : round(float(np.nan_to_num(
            y_pred_50[y_pred_50 == 1].size / max(y_pred_50.sum(), 1) *
            (y_true[y_pred_50 == 1].sum() / max(y_pred_50.sum(), 1))
        )), 4),
        "recall_at_50"         : round(float(y_true[y_pred_50 == 1].sum() / max(y_true.sum(), 1)), 4),
        "f1_at_50"             : round(f1_score(y_true, y_pred_50, zero_division=0), 4),
        # ── Threshold @ optimal F1 ────────────────────────────────────────
        "optimal_threshold"    : round(opt_thresh, 4),
        "f1_at_optimal"        : round(opt_f1, 4),
        "precision_at_optimal" : round(float(
            np.nan_to_num(
                y_true[y_pred_opt == 1].sum() / max(y_pred_opt.sum(), 1)
            )
        ), 4),
        "recall_at_optimal"    : round(float(y_true[y_pred_opt == 1].sum() / max(y_true.sum(), 1)), 4),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FraudEvaluator:
    """
    Loads both models, runs predictions on the test set, and produces
    a full comparative evaluation with figures and a JSON report.

    Attributes
    ----------
    config : EvaluatorConfig
    lgbm_trainer : LGBMFraudTrainer (loaded lazily)
    bayes_model  : BayesianFraudModel (loaded lazily)
    test_df      : pd.DataFrame — enriched test split
    results_     : dict — populated by ``run()``
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config        = config or EvaluatorConfig()
        self.lgbm_trainer  = None
        self.bayes_model   = None
        self.test_df       : Optional[pd.DataFrame] = None
        self.results_      : Dict = {}

        # Prediction arrays — populated by _run_predictions()
        self._y_true        : Optional[np.ndarray] = None
        self._lgbm_proba    : Optional[np.ndarray] = None
        self._bayes_mean    : Optional[np.ndarray] = None
        self._bayes_std     : Optional[np.ndarray] = None
        self._bayes_hdi_lo  : Optional[np.ndarray] = None
        self._bayes_hdi_hi  : Optional[np.ndarray] = None

        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data + model loading                                                 #
    # ------------------------------------------------------------------ #

    def _load_test_data(self) -> pd.DataFrame:
        """Load the enriched test split (feature pipeline output)."""
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.ingestion.splitter import load_splits

        feat_dir = self.config.feature_splits_dir
        raw_dir  = self.config.raw_splits_dir

        if (feat_dir / "test.parquet").exists():
            logger.info("Loading pre-built feature test split …")
            _, _, test_df = load_splits(feat_dir)
        elif (raw_dir / "test.parquet").exists():
            logger.info(
                "Feature splits not found — running pipeline on raw splits …"
            )
            from src.features.pipeline import FeaturePipeline
            train_r, val_r, test_r = load_splits(raw_dir)
            pipe = FeaturePipeline()
            pipe.fit(train_r)
            test_df = pipe.transform(test_r)
        else:
            raise FileNotFoundError(
                f"No test split found in:\n  {feat_dir}\n  {raw_dir}\n"
                "Run the ingestion + feature pipeline first."
            )

        logger.info("Test split loaded: shape=%s", test_df.shape)
        return test_df

    def _load_models(self) -> None:
        """Load LightGBM and Bayesian models from disk."""
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.models.traditional.lgbm_trainer import LGBMFraudTrainer
        from src.models.bayesian.bayes_lr import BayesianFraudModel

        logger.info("Loading LightGBM model from %s …", self.config.lgbm_model_dir)
        self.lgbm_trainer = LGBMFraudTrainer.load(self.config.lgbm_model_dir)

        logger.info("Loading Bayesian model from %s …", self.config.bayes_model_dir)
        self.bayes_model = BayesianFraudModel.load(self.config.bayes_model_dir)

    def _run_predictions(self) -> None:
        """Generate predictions from both models on the test set."""
        from src.models.traditional.lgbm_trainer import prepare_features

        cfg = self.config
        df  = self.test_df

        # ── LightGBM ──────────────────────────────────────────────────────
        logger.info("Running LightGBM predictions on %d test rows …", len(df))
        X_test, y_test, _ = prepare_features(df, label_col=cfg.label_col)
        self._y_true     = y_test.values.astype(np.int8)
        t0 = time.perf_counter()
        self._lgbm_proba = self.lgbm_trainer.predict_proba(X_test)
        logger.info("LightGBM prediction: %.2f s", time.perf_counter() - t0)

        # ── Bayesian ──────────────────────────────────────────────────────
        logger.info("Running Bayesian predictions …")
        t0 = time.perf_counter()
        bayes_preds = self.bayes_model.predict(df, label_col=cfg.label_col)
        logger.info("Bayesian prediction: %.2f s", time.perf_counter() - t0)

        self._bayes_mean   = bayes_preds["mean_proba"].values.astype(np.float32)
        self._bayes_std    = bayes_preds["uncertainty"].values.astype(np.float32)
        self._bayes_hdi_lo = bayes_preds["hdi_lower"].values.astype(np.float32)
        self._bayes_hdi_hi = bayes_preds["hdi_upper"].values.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def compute_metrics_matrix(self) -> Dict:
        """
        Compute and log the full metrics suite for both models.

        Returns
        -------
        dict with keys "lgbm" and "bayes", each containing the metrics dict.
        """
        logger.info("Computing metrics matrix …")
        lgbm_metrics  = compute_all_metrics(self._y_true, self._lgbm_proba,  "LightGBM")
        bayes_metrics = compute_all_metrics(self._y_true, self._bayes_mean, "Bayesian LR")

        # Pretty-print comparison table
        keys = [k for k in lgbm_metrics if k != "model"]
        logger.info("")
        logger.info("=" * 68)
        logger.info("%-30s  %-16s  %-16s", "Metric", "LightGBM", "Bayesian LR")
        logger.info("=" * 68)
        for k in keys:
            logger.info(
                "%-30s  %-16s  %-16s",
                k, str(lgbm_metrics[k]), str(bayes_metrics[k]),
            )
        logger.info("=" * 68)

        return {"lgbm": lgbm_metrics, "bayes": bayes_metrics}

    # ------------------------------------------------------------------ #
    # Figures                                                              #
    # ------------------------------------------------------------------ #

    def plot_pr_roc_curves(self) -> None:
        """
        Figure 1: Precision-Recall and ROC curves for both models on one canvas.

        The PR curve is the primary evaluation tool for imbalanced fraud
        detection.  The ROC curve is included for completeness and
        comparability with external benchmarks.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Model Comparison — PR and ROC Curves (Test Set)",
            fontsize=14, fontweight="bold", y=1.01,
        )

        # ── Precision-Recall ──────────────────────────────────────────────
        ax = axes[0]
        for name, proba, color in [
            ("LightGBM",    self._lgbm_proba, _C_LGBM),
            ("Bayesian LR", self._bayes_mean, _C_BAYES),
        ]:
            prec, rec, _ = precision_recall_curve(self._y_true, proba)
            auc          = average_precision_score(self._y_true, proba)
            ax.plot(rec, prec, color=color, linewidth=2.2,
                    label=f"{name}  (AP = {auc:.4f})")

        # Baseline: random classifier = fraud prevalence
        baseline = self._y_true.mean()
        ax.axhline(baseline, color=_C_BASE, linestyle="--", linewidth=1.2,
                   label=f"Random  (AP = {baseline:.4f})")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        # ── ROC ───────────────────────────────────────────────────────────
        ax = axes[1]
        for name, proba, color in [
            ("LightGBM",    self._lgbm_proba, _C_LGBM),
            ("Bayesian LR", self._bayes_mean, _C_BAYES),
        ]:
            fpr, tpr, _ = roc_curve(self._y_true, proba)
            auc         = roc_auc_score(self._y_true, proba)
            ax.plot(fpr, tpr, color=color, linewidth=2.2,
                    label=f"{name}  (AUC = {auc:.4f})")

        ax.plot([0, 1], [0, 1], color=_C_BASE, linestyle="--",
                linewidth=1.2, label="Random (AUC = 0.5000)")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.config.figures_dir / "fig_01_pr_roc_curves.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        plt.close(fig)

    def plot_uncertainty_distribution(self) -> None:
        """
        Figure 2: Bayesian epistemic uncertainty distribution split by class.

        The key question: does the Bayesian model express higher uncertainty
        on fraud transactions than on legitimate ones?  If yes, uncertainty
        is an independent fraud signal beyond the mean score.
        """
        hdi_width = self._bayes_hdi_hi - self._bayes_hdi_lo

        fraud_width = hdi_width[self._y_true == 1]
        legit_width = hdi_width[self._y_true == 0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Bayesian Epistemic Uncertainty: HDI Width by Transaction Class",
            fontsize=13, fontweight="bold", y=1.01,
        )

        # ── Histogram ─────────────────────────────────────────────────────
        ax = axes[0]
        bins = np.linspace(0, hdi_width.max(), 60)
        ax.hist(legit_width, bins=bins, color=_C_LGBM, alpha=0.5,
                density=True, label=f"Legit (n={len(legit_width):,})")
        ax.hist(fraud_width, bins=bins, color=_C_BAYES, alpha=0.6,
                density=True, label=f"Fraud (n={len(fraud_width):,})")
        ax.axvline(np.percentile(hdi_width, 75), color="black",
                   linestyle="--", linewidth=1.2, label="75th pct (all)")
        ax.set_xlabel("HDI Width (hdi_upper − hdi_lower)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Distribution of Uncertainty by Class", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate means
        ax.axvline(fraud_width.mean(), color=_C_BAYES, linestyle=":",
                   linewidth=1.5, alpha=0.8)
        ax.axvline(legit_width.mean(), color=_C_LGBM, linestyle=":",
                   linewidth=1.5, alpha=0.8)

        # ── CDF ───────────────────────────────────────────────────────────
        ax = axes[1]
        for label, vals, color in [
            ("Legit",  legit_width, _C_LGBM),
            ("Fraud",  fraud_width, _C_BAYES),
        ]:
            sorted_v = np.sort(vals)
            cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.plot(sorted_v, cdf, color=color, linewidth=2, label=label)

        ax.set_xlabel("HDI Width", fontsize=11)
        ax.set_ylabel("Cumulative Fraction", fontsize=11)
        ax.set_title("Uncertainty CDF by Class", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Summary stats annotation
        stats_text = (
            f"Fraud  mean uncertainty: {fraud_width.mean():.4f}\n"
            f"Legit  mean uncertainty: {legit_width.mean():.4f}\n"
            f"Ratio (fraud / legit):   {fraud_width.mean() / max(legit_width.mean(), 1e-9):.2f}×"
        )
        ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        path = self.config.figures_dir / "fig_02_uncertainty_distribution.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        plt.close(fig)

    def plot_model_agreement(self) -> None:
        """
        Figure 3: LightGBM score vs. Bayesian mean score scatter,
        coloured by epistemic uncertainty.

        Quadrants:
          • Both high    → high confidence fraud → auto-decline
          • Both low     → high confidence legit  → auto-approve
          • Disagreement → high uncertainty       → human review queue

        Colour encodes HDI width: darker = more uncertain.
        """
        hdi_width = self._bayes_hdi_hi - self._bayes_hdi_lo
        n_plot    = min(20_000, len(self._lgbm_proba))

        rng = np.random.default_rng(42)
        idx = rng.choice(len(self._lgbm_proba), size=n_plot, replace=False)

        lg   = self._lgbm_proba[idx]
        bm   = self._bayes_mean[idx]
        unc  = hdi_width[idx]
        fig, ax = plt.subplots(figsize=(9, 8))

        sc = ax.scatter(
            lg, bm,
            c=unc, cmap="YlOrRd",
            s=6, alpha=0.5,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Bayesian HDI Width (Epistemic Uncertainty)", fontsize=10)

        # Identity line (perfect agreement)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
                linewidth=1.2, alpha=0.7, label="Perfect agreement")

        # Disagreement threshold lines
        t = self.config.disagreement_threshold
        ax.axhspan(0, 0.5 - t, xmin=0.5 + t, xmax=1.0,
                   alpha=0.04, color=_C_LGBM,
                   label=f"LightGBM high, Bayes low (|Δ|>{t})")
        ax.axhspan(0.5 + t, 1.0, xmin=0.0, xmax=0.5 - t,
                   alpha=0.04, color=_C_BAYES,
                   label=f"LightGBM low, Bayes high (|Δ|>{t})")

        ax.set_xlabel("LightGBM Fraud Score", fontsize=12)
        ax.set_ylabel("Bayesian Mean P(Fraud)", fontsize=12)
        ax.set_title(
            f"Model Agreement Scatter (n={n_plot:,} test transactions)\n"
            "Colour = Bayesian epistemic uncertainty",
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        path = self.config.figures_dir / "fig_03_lgbm_vs_bayes_scatter.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        plt.close(fig)

    def plot_fn_recovery_curve(self, fn_recovery_table: List[Dict]) -> None:
        """
        Figure 4: False-Negative Recovery vs. Review Queue Size.

        X-axis: fraction of all transactions routed to human review
                (= queue size / total transactions).
        Y-axis: fraction of LightGBM's missed fraud (FN) caught by the queue.

        The steeper the early part of the curve, the more efficiently the
        Bayesian uncertainty identifies which missed frauds to surface.
        A random review queue is the diagonal baseline.
        """
        rows         = pd.DataFrame(fn_recovery_table)
        queue_frac   = rows["review_queue_pct_all"].values / 100
        fn_recovery  = rows["fn_recovery_pct"].values / 100

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(queue_frac, fn_recovery, color=_C_BAYES, linewidth=2.5,
                marker="o", markersize=6, label="Uncertainty-guided queue")
        ax.plot([0, 1], [0, 1], color=_C_BASE, linestyle="--",
                linewidth=1.5, label="Random review queue (baseline)")
        ax.fill_between(queue_frac, queue_frac, fn_recovery,
                        alpha=0.12, color=_C_BAYES, label="Gain over random")

        # Annotate each operating point
        for _, row in rows.iterrows():
            ax.annotate(
                f"p{int(row['uncertainty_percentile']*100)}\n"
                f"({row['review_queue_pct_all']:.1f}% queue)",
                xy=(row["review_queue_pct_all"] / 100, row["fn_recovery_pct"] / 100),
                xytext=(8, -12), textcoords="offset points",
                fontsize=7.5, color=_C_BAYES,
            )

        ax.set_xlabel("Review Queue Size (% of all transactions)", fontsize=12)
        ax.set_ylabel("Fraction of LightGBM False Negatives Caught", fontsize=12)
        ax.set_title(
            "False-Negative Recovery via Bayesian Uncertainty Routing\n"
            "Operating points: uncertainty percentile thresholds",
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlim([0, max(queue_frac) * 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.config.figures_dir / "fig_04_fn_recovery_curve.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        plt.close(fig)

    def plot_calibration(self) -> None:
        """
        Figure 5: Reliability diagrams for both models.

        A well-calibrated model has its reliability curve close to the
        diagonal.  Calibration matters when the model score is used as a
        direct probability input to a risk-pricing engine or in Bayesian
        updating of priors.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Probability Calibration — Reliability Diagrams (Test Set)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, (name, proba, color) in zip(axes, [
            ("LightGBM",    self._lgbm_proba, _C_LGBM),
            ("Bayesian LR", self._bayes_mean, _C_BAYES),
        ]):
            frac_pos, mean_pred = calibration_curve(
                self._y_true, proba, n_bins=10, strategy="uniform"
            )
            ece = _expected_calibration_error(self._y_true, proba)

            ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, color=color, linewidth=2.2,
                    marker="o", markersize=6, label=f"{name}")
            ax.fill_between(mean_pred, mean_pred, frac_pos,
                            alpha=0.1, color=color)
            ax.set_xlabel("Mean Predicted Probability", fontsize=11)
            ax.set_ylabel("Actual Fraud Fraction in Bin", fontsize=11)
            ax.set_title(
                f"{name} Reliability Diagram\nECE = {ece:.4f}",
                fontweight="bold",
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

        plt.tight_layout()
        path = self.config.figures_dir / "fig_05_calibration.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Uncertainty analysis                                                 #
    # ------------------------------------------------------------------ #

    def analyze_fn_recovery(self) -> Tuple[Dict, List[Dict]]:
        """
        Quantify how many of LightGBM's missed frauds (false negatives)
        are surfaced by routing high-uncertainty transactions to human review.

        Algorithm
        ---------
        For each uncertainty percentile p in ``config.uncertainty_percentiles``:
          1. Define the review queue as all transactions with
             HDI width ≥ percentile(p) of the test distribution.
          2. Count LightGBM FNs: fraud transactions scored < optimal threshold.
          3. Count FNs inside the review queue.
          4. FN recovery rate = FNs in queue / total FNs.
          5. Review queue precision = fraud in queue / all in queue.

        Returns
        -------
        summary : dict
            Top-level summary metrics.
        table : list of dicts
            Per-operating-point breakdown (suitable for JSON / DataFrame).
        """
        hdi_width    = self._bayes_hdi_hi - self._bayes_hdi_lo
        opt_thresh, _= _optimal_threshold_f1(self._y_true, self._lgbm_proba)
        lgbm_pred    = (self._lgbm_proba >= opt_thresh).astype(int)

        # LightGBM false negatives: fraud that LightGBM missed
        is_fn = (self._y_true == 1) & (lgbm_pred == 0)
        n_fn  = int(is_fn.sum())
        n_tot = len(self._y_true)

        logger.info(
            "FN analysis: LightGBM threshold=%.3f  |  FNs=%d / %d fraud  "
            "(%.1f%% miss rate)",
            opt_thresh, n_fn, int(self._y_true.sum()),
            100 * n_fn / max(self._y_true.sum(), 1),
        )

        table = []
        for pct in self.config.uncertainty_percentiles:
            threshold     = float(np.percentile(hdi_width, pct * 100))
            in_queue      = hdi_width >= threshold

            fn_in_queue   = is_fn & in_queue
            n_in_queue    = int(in_queue.sum())
            n_fn_caught   = int(fn_in_queue.sum())
            n_fraud_queue = int((self._y_true[in_queue] == 1).sum())

            fn_recovery   = n_fn_caught / max(n_fn, 1)
            queue_pct_all = n_in_queue / n_tot
            precision_q   = n_fraud_queue / max(n_in_queue, 1)

            entry = {
                "uncertainty_percentile"  : round(pct, 2),
                "hdi_width_threshold"     : round(threshold, 6),
                "review_queue_size"       : n_in_queue,
                "review_queue_pct_all"    : round(100 * queue_pct_all, 2),
                "fn_caught_in_queue"      : n_fn_caught,
                "fn_recovery_pct"         : round(100 * fn_recovery, 2),
                "fraud_in_queue"          : n_fraud_queue,
                "queue_precision_pct"     : round(100 * precision_q, 2),
            }
            table.append(entry)
            logger.info(
                "  [p%d] queue=%.1f%%  FN-recovery=%.1f%%  "
                "queue-precision=%.2f%%",
                int(pct * 100),
                100 * queue_pct_all, 100 * fn_recovery, 100 * precision_q,
            )

        # Summary: use the p75 operating point as the headline number
        p75_row = next((r for r in table if r["uncertainty_percentile"] == 0.75), table[-1])
        summary = {
            "lgbm_optimal_threshold"     : round(opt_thresh, 4),
            "total_lgbm_false_negatives" : n_fn,
            "fn_miss_rate_pct"           : round(100 * n_fn / max(self._y_true.sum(), 1), 2),
            "headline_operating_point"   : "p75 uncertainty",
            "headline_queue_size_pct"    : p75_row["review_queue_pct_all"],
            "headline_fn_recovery_pct"   : p75_row["fn_recovery_pct"],
            "headline_queue_precision_pct": p75_row["queue_precision_pct"],
        }
        return summary, table

    def analyze_disagreement(self) -> Dict:
        """
        Identify transactions where LightGBM and the Bayesian model disagree
        significantly and measure which model is more accurate in those cases.

        Disagreement is defined as ``|lgbm_score - bayes_mean| > threshold``.
        """
        diff          = np.abs(self._lgbm_proba - self._bayes_mean)
        t             = self.config.disagreement_threshold
        disagree_mask = diff > t
        n_disagree    = int(disagree_mask.sum())

        if n_disagree == 0:
            logger.warning("No disagreements found at threshold %.2f.", t)
            return {"n_disagreements": 0}

        y_dis    = self._y_true[disagree_mask]
        lg_dis   = (self._lgbm_proba[disagree_mask] >= 0.5).astype(int)
        bay_dis  = (self._bayes_mean[disagree_mask]  >= 0.5).astype(int)

        lgbm_acc  = float((lg_dis  == y_dis).mean())
        bayes_acc = float((bay_dis == y_dis).mean())

        # Separate: cases where LGBM is high, Bayes is low
        lgbm_hi_bayes_lo = (self._lgbm_proba > 0.5 + t/2) & (self._bayes_mean < 0.5 - t/2)
        bayes_hi_lgbm_lo = (self._bayes_mean > 0.5 + t/2) & (self._lgbm_proba < 0.5 - t/2)

        result = {
            "disagreement_threshold"      : t,
            "n_disagreements"             : n_disagree,
            "disagreement_rate_pct"       : round(100 * n_disagree / len(self._y_true), 2),
            "lgbm_accuracy_in_disagree"   : round(lgbm_acc, 4),
            "bayes_accuracy_in_disagree"  : round(bayes_acc, 4),
            "winner_in_disagreements"     : (
                "LightGBM" if lgbm_acc > bayes_acc else
                "Bayesian"  if bayes_acc > lgbm_acc else "Tie"
            ),
            "lgbm_high_bayes_low_count"   : int(lgbm_hi_bayes_lo.sum()),
            "bayes_high_lgbm_low_count"   : int(bayes_hi_lgbm_lo.sum()),
            "fraud_rate_lgbm_hi_bayes_lo" : round(float(
                self._y_true[lgbm_hi_bayes_lo].mean()
                if lgbm_hi_bayes_lo.sum() > 0 else 0.0
            ), 4),
            "fraud_rate_bayes_hi_lgbm_lo" : round(float(
                self._y_true[bayes_hi_lgbm_lo].mean()
                if bayes_hi_lgbm_lo.sum() > 0 else 0.0
            ), 4),
        }
        logger.info(
            "Disagreement analysis: %d transactions (%.1f%%)  |  "
            "LGBM acc=%.3f  Bayes acc=%.3f",
            n_disagree, result["disagreement_rate_pct"],
            lgbm_acc, bayes_acc,
        )
        return result

    def analyze_uncertainty_summary(self) -> Dict:
        """
        Compute top-level uncertainty statistics for the report.
        """
        hdi_width     = self._bayes_hdi_hi - self._bayes_hdi_lo
        fraud_width   = hdi_width[self._y_true == 1]
        legit_width   = hdi_width[self._y_true == 0]

        return {
            "mean_uncertainty_all"   : round(float(hdi_width.mean()), 6),
            "mean_uncertainty_fraud" : round(float(fraud_width.mean()), 6),
            "mean_uncertainty_legit" : round(float(legit_width.mean()), 6),
            "uncertainty_ratio"      : round(
                float(fraud_width.mean() / max(legit_width.mean(), 1e-9)), 4
            ),
            "pct_high_uncertainty_q75": round(
                100 * float((hdi_width >= np.percentile(hdi_width, 75)).mean()), 2
            ),
            "bayes_std_mean_fraud"   : round(float(self._bayes_std[self._y_true == 1].mean()), 6),
            "bayes_std_mean_legit"   : round(float(self._bayes_std[self._y_true == 0].mean()), 6),
        }

    # ------------------------------------------------------------------ #
    # Master run                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict:
        """
        Execute the full evaluation pipeline.

        Steps
        -----
        1. Load data
        2. Load models
        3. Run predictions
        4. Compute metrics matrix
        5. Compute uncertainty + FN recovery analysis
        6. Produce all figures
        7. Save JSON report

        Returns
        -------
        dict
            Complete results dictionary (also stored as ``self.results_``).
        """
        t_total = time.perf_counter()
        logger.info("=" * 65)
        logger.info("FraudEvaluator — full evaluation run")
        logger.info("=" * 65)

        # ── 1. Data ──────────────────────────────────────────────────────
        self.test_df = self._load_test_data()

        # ── 2. Models ─────────────────────────────────────────────────────
        self._load_models()

        # ── 3. Predictions ────────────────────────────────────────────────
        self._run_predictions()

        # ── 4. Metrics ────────────────────────────────────────────────────
        metrics = self.compute_metrics_matrix()

        # ── 5. Uncertainty analysis ───────────────────────────────────────
        logger.info("Running uncertainty analysis …")
        unc_summary        = self.analyze_uncertainty_summary()
        fn_summary, fn_table = self.analyze_fn_recovery()
        disagree_analysis  = self.analyze_disagreement()

        # ── 6. Figures ────────────────────────────────────────────────────
        logger.info("Generating figures …")
        self.plot_pr_roc_curves()
        self.plot_uncertainty_distribution()
        self.plot_model_agreement()
        self.plot_fn_recovery_curve(fn_table)
        self.plot_calibration()

        # ── 7. Assemble + save report ─────────────────────────────────────
        self.results_ = {
            "evaluation_metadata": {
                "test_rows"   : len(self._y_true),
                "test_fraud_n": int(self._y_true.sum()),
                "test_fraud_rate_pct": round(100 * self._y_true.mean(), 4),
                "elapsed_s"   : round(time.perf_counter() - t_total, 1),
            },
            "models"             : metrics,
            "uncertainty_analysis": unc_summary,
            "fn_recovery"        : {
                "summary": fn_summary,
                "table"  : fn_table,
            },
            "disagreement_analysis": disagree_analysis,
        }

        self._save_report()

        logger.info(
            "Evaluation complete in %.1f s.  Results saved to %s",
            time.perf_counter() - t_total, self.config.reports_dir,
        )
        return self.results_

    def _save_report(self) -> None:
        """Persist the JSON metrics report."""
        path = self.config.reports_dir / "model_comparison_metrics.json"
        with open(path, "w") as f:
            json.dump(self.results_, f, indent=2, default=str)
        logger.info("Saved metrics report → %s", path)

    def print_executive_summary(self) -> None:
        """
        Print a concise executive summary suitable for pasting into a README
        or presenting to a fraud review team.
        """
        if not self.results_:
            raise RuntimeError("Run evaluate.run() first.")

        lg  = self.results_["models"]["lgbm"]
        bay = self.results_["models"]["bayes"]
        fn  = self.results_["fn_recovery"]["summary"]
        unc = self.results_["uncertainty_analysis"]
        dis = self.results_["disagreement_analysis"]
        meta= self.results_["evaluation_metadata"]

        print("\n" + "═" * 65)
        print("  FRAUD DETECTION ENGINE — EVALUATION SUMMARY (TEST SET)")
        print("═" * 65)
        print(f"  Test set: {meta['test_rows']:,} transactions  |  "
              f"Fraud: {meta['test_fraud_n']:,} ({meta['test_fraud_rate_pct']:.2f}%)")
        print()
        print(f"  {'Metric':<28}  {'LightGBM':>12}  {'Bayesian LR':>12}")
        print(f"  {'-'*28}  {'-'*12}  {'-'*12}")
        for key, label in [
            ("pr_auc",      "PR-AUC (primary)"),
            ("roc_auc",     "ROC-AUC"),
            ("gini",        "Gini coefficient"),
            ("ks_statistic","KS Statistic"),
            ("brier_score", "Brier Score"),
            ("ece",         "Calibration (ECE)"),
            ("f1_at_optimal","F1 @ optimal thresh"),
        ]:
            print(f"  {label:<28}  {str(lg[key]):>12}  {str(bay[key]):>12}")

        print()
        print("  BAYESIAN UNCERTAINTY ANALYSIS")
        print(f"  {'-'*50}")
        print(f"  Mean uncertainty — fraud :  {unc['mean_uncertainty_fraud']:.4f}")
        print(f"  Mean uncertainty — legit :  {unc['mean_uncertainty_legit']:.4f}")
        print(f"  Uncertainty ratio         :  {unc['uncertainty_ratio']:.2f}× higher on fraud")
        print()
        print("  FALSE-NEGATIVE RECOVERY (Bayesian uncertainty routing)")
        print(f"  {'-'*50}")
        print(f"  LightGBM missed fraud    :  {fn['total_lgbm_false_negatives']:,} "
              f"({fn['fn_miss_rate_pct']:.1f}% miss rate)")
        print(f"  Operating point          :  {fn['headline_operating_point']}")
        print(f"  Review queue size        :  {fn['headline_queue_size_pct']:.1f}% of all transactions")
        print(f"  Missed fraud recovered   :  {fn['headline_fn_recovery_pct']:.1f}% of LightGBM FNs")
        print(f"  Queue precision          :  {fn['headline_queue_precision_pct']:.2f}%")
        print()
        print("  MODEL DISAGREEMENT ANALYSIS")
        print(f"  {'-'*50}")
        print(f"  Disagreement rate        :  {dis.get('disagreement_rate_pct', 'N/A')}% of transactions")
        print(f"  Winner in disagreements  :  {dis.get('winner_in_disagreements', 'N/A')}")
        print("═" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config    = EvaluatorConfig()
    evaluator = FraudEvaluator(config)

    results = evaluator.run()
    evaluator.print_executive_summary()

    print(f"\n5 figures saved to : {config.figures_dir}")
    print(f"JSON report saved to: {config.reports_dir / 'model_comparison_metrics.json'}")
