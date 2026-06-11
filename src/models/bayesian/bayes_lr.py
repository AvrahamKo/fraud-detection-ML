"""
src/models/bayesian/bayes_lr.py
================================
Bayesian Logistic Regression for fraud detection using PyMC + ADVI.

Why Bayesian? The core differentiator.
---------------------------------------
LightGBM returns a point estimate — a single number p(fraud | x).  A
Bayesian model returns a *distribution* over that probability.  That
distribution encodes two distinct types of uncertainty:

* **Epistemic uncertainty** (model uncertainty): the model doesn't have
  enough training signal to be confident about this transaction.  High
  when the transaction pattern is rare or novel — exactly the case for
  adaptive adversarial fraud that the model hasn't seen before.
  Manifests as a wide posterior: ``std(p(fraud | x, θ) over θ) is large``.

* **Aleatoric uncertainty** (data uncertainty): even with infinite data
  the outcome is inherently stochastic.  Irreducible.  Manifests as
  ``p̂`` near 0.5 regardless of how many posterior samples you draw.

In a production fraud system, the epistemic uncertainty column is directly
actionable:
  - Low uncertainty, high p(fraud)   → auto-decline
  - Low uncertainty, low p(fraud)    → auto-approve
  - High uncertainty, any p(fraud)   → route to human review queue

This is the core analytical differentiator versus the LightGBM model.

Scalability strategy
---------------------
Standard MCMC (NUTS) on 350k rows at ~60 features would take days.
We use two complementary techniques:

1. **Feature reduction** to a canonical set of 15 interpretable,
   high-signal features (velocity, spend spikes, identity mismatch).
   This keeps the posterior geometry manageable.

2. **ADVI** (Automatic Differentiation Variational Inference) via
   ``pm.fit(method="advi")``.  ADVI approximates the posterior with a
   mean-field Gaussian family.  It is orders of magnitude faster than MCMC
   and converges in minutes.  The trade-off: it underestimates posterior
   variance (mean-field assumption).  For a portfolio comparison this
   is explicitly documented; in production you would use full-rank VI or
   NUTS on a GPU cluster.

3. **Stratified subsampling** — we train on at most ``n_sample_max`` rows,
   preserving the true fraud rate via stratified sampling.  This gives ADVI
   enough signal (thousands of fraud examples) while keeping wall-clock time
   under 5 minutes on a laptop.

Module layout
-------------
``BAYESIAN_FEATURES``        — canonical 15-feature list
``BayesianConfig``           — dataclass of all knobs
``prepare_bayesian_features``— select → subsample → scale
``BayesianFraudModel``       — stateful model class
    .fit()                   — build PyMC model + run ADVI + sample trace
    .predict()               — mean + std + HDI per transaction
    .get_coefficient_summary()— posterior means and 94% HDI for each β
    .plot_elbo()             — ELBO convergence curve
    .save() / .load()        — artefact persistence via ArviZ + pickle
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR   = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Canonical feature set
# ---------------------------------------------------------------------------

BAYESIAN_FEATURES: List[str] = [
    # ── Velocity: card-level ─────────────────────────────────────────────
    # Fraudsters burst-fire transactions on a compromised card.  The 1-hour
    # count is the sharpest signal; 24h and 7d capture sustained campaigns.
    "vel_card1_1h_count",
    "vel_card1_24h_count",
    "vel_card1_7d_count",
    "vel_card1_1h_sum",
    # ── Velocity: region-level ────────────────────────────────────────────
    # Fraud rings often share a billing region (drop-ship warehouses, etc.)
    "vel_addr1_24h_count",
    # ── Behavioral: spend spikes ──────────────────────────────────────────
    # A legitimate cardholder's spend follows a historical distribution.
    # Deviations — especially upward — are a strong fraud signal.
    "log1p_amt",
    "amt_to_card1_mean_ratio",
    "amt_zscore_card1",
    # ── Temporal ─────────────────────────────────────────────────────────
    "hour_of_day",
    "is_night",
    "is_weekend",
    # ── Identity mismatch ─────────────────────────────────────────────────
    "email_match",
    "has_identity",
    "card1_addr1_is_mode",
    # ── Identity completeness ─────────────────────────────────────────────
    "identity_completeness",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BayesianConfig:
    """
    All knobs for the Bayesian Logistic Regression trainer.

    Attributes
    ----------
    features : list of str
        Feature columns to include.  Defaults to ``BAYESIAN_FEATURES``.
    n_sample_max : int
        Maximum training rows fed to ADVI.  A stratified subsample is
        taken if the dataset is larger.
    min_fraud_in_sample : int
        Guarantee at least this many fraud rows in the subsample.
        Prevents the posterior from collapsing on sparse signal.
    n_advi_iterations : int
        Number of ADVI gradient steps.  30k is sufficient for convergence
        on ~50k rows with 15 features; reduce to 10k for rapid prototyping.
    n_posterior_samples : int
        Number of samples to draw from the ADVI posterior approximation.
        These are stored as the ``trace_`` InferenceData.
    n_prediction_samples : int
        Posterior samples used per test batch for uncertainty estimation.
        500 gives stable std estimates; 200 is enough for ranking.
    prior_beta_sigma : float
        Standard deviation of the Normal(0, σ) prior on coefficients.
        1.0 is weakly informative on standardised features.
    prior_alpha_sigma : float
        Std of the Normal(0, σ) prior on the intercept.
        2.0 allows the baseline log-odds to float freely.
    random_seed : int
    model_name : str
    """
    features           : List[str] = field(
        default_factory=lambda: list(BAYESIAN_FEATURES)
    )
    n_sample_max       : int   = 50_000
    min_fraud_in_sample: int   = 2_000
    n_advi_iterations  : int   = 30_000
    n_posterior_samples: int   = 2_000
    n_prediction_samples: int  = 500
    prior_beta_sigma   : float = 1.0
    prior_alpha_sigma  : float = 2.0
    random_seed        : int   = 42
    model_name         : str   = "bayes_lr_fraud"


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_bayesian_features(
    df          : pd.DataFrame,
    config      : BayesianConfig,
    label_col   : str = "isFraud",
    scaler      : Optional[StandardScaler] = None,
    is_train    : bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler, List[str]]:
    """
    Select, subsample (train only), and standardise features for the
    Bayesian model.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched split from ``FeaturePipeline.transform()``.
    config : BayesianConfig
    label_col : str
    scaler : StandardScaler, optional
        Pre-fitted scaler.  Must be passed for val/test to avoid leakage.
        If ``None`` and ``is_train=True``, a new scaler is fitted here.
    is_train : bool
        If True, applies stratified subsampling and fits the scaler.
        If False, uses the full ``df`` with the passed scaler.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_rows, n_features)
    y        : np.ndarray or None
        Integer labels; None if ``label_col`` is absent (inference mode).
    scaler   : StandardScaler
        Fitted scaler (same object if passed in; new if fitted here).
    available_features : list of str
        Subset of ``config.features`` actually present in ``df``.
    """
    # ── Resolve available features ────────────────────────────────────────
    available = [f for f in config.features if f in df.columns]
    missing   = [f for f in config.features if f not in df.columns]
    if missing:
        logger.warning(
            "Bayesian prep: %d requested features absent — skipping: %s",
            len(missing), missing,
        )
    if not available:
        raise ValueError(
            "None of the requested Bayesian features are present in the "
            "DataFrame.  Run the FeaturePipeline first."
        )
    logger.info("Bayesian prep: using %d features.", len(available))

    # ── Target ────────────────────────────────────────────────────────────
    if label_col in df.columns:
        y_series = df[label_col]
    else:
        y_series = None

    # ── Stratified subsample (training only) ─────────────────────────────
    if is_train and y_series is not None and len(df) > config.n_sample_max:
        logger.info(
            "Bayesian prep: stratified subsample %d → %d rows …",
            len(df), config.n_sample_max,
        )
        rng        = np.random.default_rng(config.random_seed)
        fraud_idx  = df.index[y_series == 1].tolist()
        legit_idx  = df.index[y_series == 0].tolist()

        # Guarantee minimum fraud rows
        n_fraud = max(config.min_fraud_in_sample, int(config.n_sample_max * y_series.mean()))
        n_fraud = min(n_fraud, len(fraud_idx))
        n_legit = min(config.n_sample_max - n_fraud, len(legit_idx))

        chosen_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        chosen_legit = rng.choice(legit_idx, size=n_legit, replace=False)
        chosen       = np.concatenate([chosen_fraud, chosen_legit])
        rng.shuffle(chosen)

        df       = df.loc[chosen]
        y_series = y_series.loc[chosen]
        logger.info(
            "Subsample: %d rows  |  fraud=%d (%.2f%%)",
            len(df), int(y_series.sum()), y_series.mean() * 100,
        )

    # ── Extract numeric matrix; coerce categoricals ───────────────────────
    X_raw = df[available].copy()
    for col in X_raw.columns:
        if X_raw[col].dtype.name == "category":
            X_raw[col] = X_raw[col].cat.codes
        elif X_raw[col].dtype == object:
            X_raw[col] = pd.factorize(X_raw[col])[0]
    X_raw = X_raw.astype(np.float32)

    # Fill remaining NaNs with 0 (entity never seen → zero velocity is
    # the correct imputation; other features have been cleaned upstream)
    X_raw = X_raw.fillna(0.0)

    # ── Scale ─────────────────────────────────────────────────────────────
    if scaler is None:
        if not is_train:
            raise ValueError(
                "scaler=None is only valid when is_train=True. "
                "Pass the training scaler when preparing val/test data."
            )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
        logger.info("StandardScaler fitted on %d training rows.", len(X_raw))
    else:
        X_scaled = scaler.transform(X_raw).astype(np.float32)

    y_arr = y_series.astype(np.int8).values if y_series is not None else None
    return X_scaled, y_arr, scaler, available


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BayesianFraudModel:
    """
    Bayesian Logistic Regression fraud classifier.

    The model learns a posterior distribution over the weight vector β
    rather than a single point estimate.  At prediction time, we draw
    samples from this posterior and propagate them through the sigmoid
    function to obtain a *distribution* over fraud probability for each
    transaction.

    Workflow
    --------
    ::

        from src.models.bayesian.bayes_lr import BayesianFraudModel, BayesianConfig

        model = BayesianFraudModel(BayesianConfig())
        model.fit(train_feat, label_col="isFraud")

        results = model.predict(val_feat)
        # results["mean_proba"]    — point estimate of p(fraud)
        # results["uncertainty"]   — epistemic uncertainty (posterior std)
        # results["hdi_lower"]     — 94% HDI lower bound
        # results["hdi_upper"]     — 94% HDI upper bound

        coefs = model.get_coefficient_summary()
        model.save(Path("models/bayes_v1"))

    Attributes
    ----------
    config : BayesianConfig
    trace_ : arviz.InferenceData
        Posterior samples from ADVI.  Available after ``fit()``.
    scaler_ : StandardScaler
        Fitted on training features.
    feature_names_ : list of str
        Features actually used (subset of config.features present in data).
    elbo_history_ : np.ndarray
        ELBO per ADVI iteration.  Plot with ``plot_elbo()``.
    _alpha_draws : np.ndarray, shape (n_posterior_samples,)
        Cached intercept samples for fast prediction.
    _beta_draws  : np.ndarray, shape (n_posterior_samples, n_features)
        Cached coefficient samples for fast prediction.
    """

    def __init__(self, config: Optional[BayesianConfig] = None) -> None:
        self.config          = config or BayesianConfig()
        self.trace_          : Optional[az.InferenceData] = None
        self.scaler_         : Optional[StandardScaler]   = None
        self.feature_names_  : List[str]                  = []
        self.elbo_history_   : Optional[np.ndarray]       = None
        self._alpha_draws    : Optional[np.ndarray]       = None
        self._beta_draws     : Optional[np.ndarray]       = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_df  : pd.DataFrame,
        label_col : str = "isFraud",
    ) -> "BayesianFraudModel":
        """
        Prepare data, build the PyMC model, run ADVI, and sample the trace.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split (enriched by FeaturePipeline).
        label_col : str
            Target column name.

        Returns
        -------
        self
        """
        t0 = time.perf_counter()
        cfg = self.config

        # ── Prepare ───────────────────────────────────────────────────────
        X, y, scaler, feat_names = prepare_bayesian_features(
            train_df, cfg, label_col=label_col,
            scaler=None, is_train=True,
        )
        self.scaler_        = scaler
        self.feature_names_ = feat_names
        n_features          = X.shape[1]

        logger.info(
            "Bayesian model: %d training rows  |  %d features  |  "
            "fraud=%.3f%%  |  seed=%d",
            len(X), n_features, y.mean() * 100, cfg.random_seed,
        )

        # ── Build PyMC model and run ADVI ─────────────────────────────────
        self._build_and_fit(X, y, n_features)

        logger.info(
            "BayesianFraudModel.fit() complete in %.1f s.",
            time.perf_counter() - t0,
        )
        return self

    def _build_and_fit(
        self,
        X         : np.ndarray,
        y         : np.ndarray,
        n_features: int,
    ) -> None:
        """
        Internal: construct the PyMC graph, run ADVI, sample the trace.

        Model specification
        -------------------
        * Intercept α ~ Normal(0, σ_α)
          Weakly informative — allows the overall log-odds of fraud to
          shift freely.

        * Coefficients β_j ~ Normal(0, σ_β)  for j = 1 … p
          Independent Normal priors on standardised features.  σ_β = 1.0
          puts 95% of prior mass on coefficients that change the log-odds
          by at most ±2, which is appropriate for fraud signals.

        * Likelihood: y_i ~ Bernoulli(logit_p = α + X_i · β)
          We use ``logit_p`` directly (numerically more stable than ``p``).

        ADVI approximation
        ------------------
        Mean-field ADVI fits a Gaussian q(θ) ≈ p(θ | data) by minimising
        the KL divergence via stochastic gradient descent on the ELBO.
        The ELBO history is saved for convergence diagnostics.
        """
        cfg = self.config

        logger.info("Building PyMC model …")
        with pm.Model():
            # ── Priors ────────────────────────────────────────────────────
            alpha = pm.Normal(
                "alpha",
                mu=0.0, sigma=cfg.prior_alpha_sigma,
            )
            beta = pm.Normal(
                "beta",
                mu=0.0, sigma=cfg.prior_beta_sigma,
                shape=n_features,
            )

            # ── Linear predictor (log-odds of fraud) ──────────────────────
            # pm.math.dot handles PyTensor / Theano backend transparently
            logit_p = alpha + pm.math.dot(X, beta)

            # ── Likelihood ────────────────────────────────────────────────
            # Using logit_p= is numerically stable and avoids computing
            # sigmoid inside the graph (saves a graph node)
            pm.Bernoulli("obs", logit_p=logit_p, observed=y)

            # ── ADVI ──────────────────────────────────────────────────────
            logger.info(
                "Running ADVI (%d iterations) — this may take a few minutes …",
                cfg.n_advi_iterations,
            )
            t_advi = time.perf_counter()

            with warnings.catch_warnings():
                # Suppress PyMC's "UserWarning: gradient contains NaN" during
                # early ADVI iterations before the optimiser stabilises
                warnings.simplefilter("ignore", UserWarning)
                approx = pm.fit(
                    n            = cfg.n_advi_iterations,
                    method       = "advi",
                    random_seed  = cfg.random_seed,
                    progressbar  = True,
                    callbacks    = [
                        pm.callbacks.CheckParametersConvergence(
                            diff      = "absolute",
                            tolerance = 0.01,
                            every     = 500,
                        )
                    ],
                )

            elbo_elapsed = time.perf_counter() - t_advi
            logger.info("ADVI finished in %.1f s.", elbo_elapsed)

            # Save ELBO history for convergence diagnostic plot
            self.elbo_history_ = np.array(approx.hist)

            # Final ELBO (lower = worse, higher = better for ELBO)
            logger.info(
                "Final ELBO: %.2f  (initial: %.2f)",
                self.elbo_history_[-1], self.elbo_history_[0],
            )

            # ── Sample from the ADVI posterior ────────────────────────────
            logger.info(
                "Sampling %d draws from ADVI posterior …",
                cfg.n_posterior_samples,
            )
            trace = approx.sample(
                draws       = cfg.n_posterior_samples,
                random_seed = cfg.random_seed,
            )

        self.trace_ = trace

        # Cache flattened draws for fast numpy prediction
        # trace.posterior has shape (chain, draw, [shape])
        self._alpha_draws = (
            trace.posterior["alpha"]
            .values
            .flatten()                       # (n_chains * n_draws,)
            .astype(np.float32)
        )
        self._beta_draws = (
            trace.posterior["beta"]
            .values
            .reshape(-1, n_features)         # (n_chains * n_draws, n_features)
            .astype(np.float32)
        )
        logger.info(
            "Posterior cached: alpha_draws=%s  beta_draws=%s",
            self._alpha_draws.shape, self._beta_draws.shape,
        )

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        df        : pd.DataFrame,
        label_col : str = "isFraud",
    ) -> pd.DataFrame:
        """
        Compute posterior predictive distribution per transaction.

        For each row we draw ``n_prediction_samples`` weight vectors from
        the stored posterior and evaluate the sigmoid.  The resulting
        distribution over p(fraud | x, θ) gives us:

        * ``mean_proba``  — point estimate (equivalent to LightGBM's output)
        * ``uncertainty`` — epistemic uncertainty (std of the posterior)
        * ``hdi_lower``   — lower bound of 94% Highest Density Interval
        * ``hdi_upper``   — upper bound of 94% HDI

        **High uncertainty + high mean_proba → auto-decline.**
        **High uncertainty + mid mean_proba  → human review queue.**
        **Low uncertainty  + low mean_proba  → auto-approve.**

        Parameters
        ----------
        df : pd.DataFrame
            Any enriched split.  Must contain the same feature columns as
            the training data.
        label_col : str
            If present, a ``true_label`` column is included in the output
            for convenience.

        Returns
        -------
        pd.DataFrame
            One row per input row.  Columns:
            ``mean_proba``, ``uncertainty``, ``hdi_lower``, ``hdi_upper``,
            and optionally ``true_label``.
        """
        if self._alpha_draws is None or self._beta_draws is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled, y_arr, _, _ = prepare_bayesian_features(
            df, self.config,
            label_col=label_col,
            scaler=self.scaler_,
            is_train=False,
        )

        mean_p, std_p, hdi_lo, hdi_hi = self._predict_raw(X_scaled)

        result = pd.DataFrame({
            "mean_proba" : mean_p,
            "uncertainty": std_p,
            "hdi_lower"  : hdi_lo,
            "hdi_upper"  : hdi_hi,
        }, index=df.index[:len(mean_p)])

        if y_arr is not None:
            result["true_label"] = y_arr

        return result

    def _predict_raw(
        self,
        X_scaled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Core posterior predictive computation (pure numpy, no PyMC overhead).

        Steps
        -----
        1. Subsample ``n_prediction_samples`` posterior draws (for speed).
        2. Compute logits: ``logit_p[s, i] = α[s] + X[i] @ β[s]``
        3. Apply sigmoid: ``p_matrix[s, i] = σ(logit_p[s, i])``
        4. Aggregate across posterior samples: mean, std, 3% / 97% quantiles
           (approximating the 94% HDI under near-Gaussian posteriors).

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_test, n_features)

        Returns
        -------
        mean_proba, uncertainty, hdi_lower, hdi_upper : np.ndarray
            All shape (n_test,).
        """
        cfg      = self.config
        n_draws  = len(self._alpha_draws)
        n_sample = min(cfg.n_prediction_samples, n_draws)

        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(n_draws, size=n_sample, replace=False)

        alpha_s = self._alpha_draws[idx]            # (n_sample,)
        beta_s  = self._beta_draws[idx]             # (n_sample, n_features)

        # Broadcasting: (n_sample, n_test)
        # alpha_s[:, None]  → (n_sample, 1)
        # (beta_s @ X_scaled.T) → (n_sample, n_test)
        logits   = alpha_s[:, None] + (beta_s @ X_scaled.T)  # (n_sample, n_test)
        p_matrix = sigmoid(logits).astype(np.float32)        # (n_sample, n_test)

        mean_p = p_matrix.mean(axis=0)
        std_p  = p_matrix.std(axis=0)

        # 94% HDI approximated as 3rd–97th percentile
        # (exact HDI via ArviZ is available but slow for large n_test)
        hdi_lo = np.percentile(p_matrix, 3,  axis=0)
        hdi_hi = np.percentile(p_matrix, 97, axis=0)

        return mean_p, std_p, hdi_lo, hdi_hi

    # ------------------------------------------------------------------ #
    # Interpretability                                                      #
    # ------------------------------------------------------------------ #

    def get_coefficient_summary(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of posterior statistics for each coefficient.

        Columns
        -------
        ``feature``       : feature name (or "intercept")
        ``mean``          : posterior mean of β_j
        ``sd``            : posterior standard deviation
        ``hdi_3%``        : lower bound of 94% HDI
        ``hdi_97%``       : upper bound of 94% HDI
        ``r_hat``         : Gelman-Rubin convergence diagnostic
                            (always 1.0 for single-chain ADVI — shown for
                            compatibility with MCMC traces)
        ``direction``     : "increases fraud risk" / "decreases fraud risk"
        ``significant``   : True if 94% HDI does not straddle zero

        Returns
        -------
        pd.DataFrame sorted by |mean| descending.
        """
        if self.trace_ is None:
            raise RuntimeError("Model not fitted.")

        summary = az.summary(
            self.trace_,
            var_names=["alpha", "beta"],
            hdi_prob=0.94,
            round_to=6,
        )

        # Rename index rows to meaningful feature names
        n_feat  = len(self.feature_names_)
        new_idx = ["intercept"] + self.feature_names_

        # az.summary returns rows: alpha (1 row) + beta[0..n-1] (n rows)
        if len(summary) == 1 + n_feat:
            summary.index = new_idx
        else:
            logger.warning(
                "Summary row count (%d) doesn't match 1 + n_features (%d); "
                "index not renamed.", len(summary), 1 + n_feat
            )

        summary = summary.reset_index().rename(columns={"index": "feature"})
        summary["significant"] = (
            (summary["hdi_3%"] > 0) | (summary["hdi_97%"] < 0)
        )
        summary["direction"] = summary["mean"].apply(
            lambda m: "increases fraud risk" if m > 0 else "decreases fraud risk"
        )

        return summary.sort_values("mean", key=abs, ascending=False)

    def plot_elbo(self, ax=None, skip_first_n: int = 500):
        """
        Plot the ELBO convergence curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, a new figure is created.
        skip_first_n : int
            Skip the first N iterations where ELBO is very negative
            (scale distortion during early optimisation).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.elbo_history_ is None:
            raise RuntimeError("No ELBO history — call fit() first.")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        history = self.elbo_history_[skip_first_n:]
        iters   = np.arange(skip_first_n, skip_first_n + len(history))

        ax.plot(iters, history, color="#2196F3", linewidth=0.8, alpha=0.8)
        ax.set_xlabel("ADVI Iteration")
        ax.set_ylabel("Negative ELBO (lower = better fit)")
        ax.set_title("ADVI Convergence — ELBO History", fontweight="bold")
        ax.grid(True, alpha=0.3)

        return ax

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, dir_path: Path) -> None:
        """
        Persist all artefacts to ``dir_path``.

        Saved files
        -----------
        ``trace.nc``            — ArviZ InferenceData (NetCDF4 format).
                                  Human-readable, version-stable.
        ``scaler.pkl``          — Fitted StandardScaler.
        ``elbo_history.npy``    — ADVI ELBO per iteration.
        ``alpha_draws.npy``     — Cached intercept posterior samples.
        ``beta_draws.npy``      — Cached coefficient posterior samples.
        ``meta.json``           — Config, feature names, model metadata.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.trace_ is None:
            raise RuntimeError("Nothing to save — call fit() first.")

        # InferenceData → NetCDF4
        trace_path = dir_path / "trace.nc"
        az.to_netcdf(self.trace_, str(trace_path))
        logger.info("Saved InferenceData → %s", trace_path)

        # Scaler
        scaler_path = dir_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler_, f)
        logger.info("Saved scaler → %s", scaler_path)

        # ELBO history
        if self.elbo_history_ is not None:
            elbo_path = dir_path / "elbo_history.npy"
            np.save(str(elbo_path), self.elbo_history_)
            logger.info("Saved ELBO history → %s", elbo_path)

        # Cached draws (for fast loading — avoids re-extracting from trace)
        np.save(str(dir_path / "alpha_draws.npy"), self._alpha_draws)
        np.save(str(dir_path / "beta_draws.npy"),  self._beta_draws)

        # Metadata
        meta = {
            "model_name"     : self.config.model_name,
            "feature_names"  : self.feature_names_,
            "config"         : {
                k: v for k, v in vars(self.config).items()
                if not isinstance(v, list)        # lists saved separately
            },
            "config_features": self.config.features,
            "n_training_features": len(self.feature_names_),
        }
        meta_path = dir_path / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Saved metadata → %s", meta_path)

        # Coefficient summary (for quick inspection / report)
        try:
            coef_df   = self.get_coefficient_summary()
            coef_path = dir_path / "coefficient_summary.csv"
            coef_df.to_csv(coef_path, index=False)
            logger.info("Saved coefficient summary → %s", coef_path)
        except Exception as exc:
            logger.warning("Could not save coefficient summary: %s", exc)

        logger.info("All Bayesian artefacts saved to %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "BayesianFraudModel":
        """
        Re-hydrate a BayesianFraudModel from a saved directory.

        The model is restored to a fully predict()-ready state without
        rebuilding the PyMC graph (we load the cached numpy draws directly).

        Parameters
        ----------
        dir_path : Path

        Returns
        -------
        BayesianFraudModel
        """
        dir_path = Path(dir_path)

        # Metadata
        meta_path = dir_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {dir_path}")
        with open(meta_path) as f:
            meta = json.load(f)

        # Reconstruct config
        cfg = BayesianConfig()
        for k, v in meta.get("config", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.features = meta.get("config_features", list(BAYESIAN_FEATURES))

        model = cls(config=cfg)
        model.feature_names_ = meta["feature_names"]

        # Scaler
        scaler_path = dir_path / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            model.scaler_ = pickle.load(f)

        # InferenceData (ArviZ NetCDF)
        trace_path = dir_path / "trace.nc"
        if trace_path.exists():
            model.trace_ = az.from_netcdf(str(trace_path))
            logger.info("Loaded InferenceData from %s", trace_path)

        # Cached draws
        alpha_path = dir_path / "alpha_draws.npy"
        beta_path  = dir_path / "beta_draws.npy"
        if alpha_path.exists() and beta_path.exists():
            model._alpha_draws = np.load(str(alpha_path))
            model._beta_draws  = np.load(str(beta_path))
        elif model.trace_ is not None:
            # Fallback: re-extract from trace
            n_feat = len(model.feature_names_)
            model._alpha_draws = (
                model.trace_.posterior["alpha"].values.flatten().astype(np.float32)
            )
            model._beta_draws = (
                model.trace_.posterior["beta"].values.reshape(-1, n_feat).astype(np.float32)
            )

        # ELBO history
        elbo_path = dir_path / "elbo_history.npy"
        if elbo_path.exists():
            model.elbo_history_ = np.load(str(elbo_path))

        logger.info("BayesianFraudModel loaded from %s", dir_path)
        return model


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
    _OUTPUT_DIR    = _MODELS_DIR / "bayes_lr_v1"

    # ── Load enriched splits ───────────────────────────────────────────────
    if (_FEATURE_DIR / "train.parquet").exists():
        logger.info("Loading pre-built feature splits …")
        train_f, val_f, test_f = load_splits(_FEATURE_DIR)
    else:
        logger.info("Running feature pipeline first …")
        train_r, val_r, test_r = load_splits(_PROCESSED_DIR)
        pipe = FeaturePipeline()
        train_f, val_f, test_f = pipe.fit_transform_splits(train_r, val_r, test_r)

    # ── Train ──────────────────────────────────────────────────────────────
    config = BayesianConfig(
        n_sample_max        = 50_000,
        min_fraud_in_sample = 2_000,
        n_advi_iterations   = 30_000,
        n_posterior_samples = 2_000,
        n_prediction_samples= 500,
    )
    bayes_model = BayesianFraudModel(config)
    bayes_model.fit(train_f)

    # ── Predict on validation set ──────────────────────────────────────────
    logger.info("Predicting on validation set …")
    val_predictions = bayes_model.predict(val_f)

    fraud_mask = val_predictions["true_label"] == 1
    legit_mask = val_predictions["true_label"] == 0

    print("\n=== Posterior Predictive Summary ===")
    print(f"Val rows: {len(val_predictions):,}")
    print(f"\nFraud transactions (n={fraud_mask.sum():,}):")
    print(val_predictions[fraud_mask][["mean_proba", "uncertainty"]].describe().round(4))
    print(f"\nLegit transactions (n={legit_mask.sum():,}):")
    print(val_predictions[legit_mask][["mean_proba", "uncertainty"]].describe().round(4))

    # ── Evaluate PR-AUC ───────────────────────────────────────────────────
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = val_predictions["true_label"].values
    y_pred = val_predictions["mean_proba"].values
    print(f"\nVal PR-AUC  : {average_precision_score(y_true, y_pred):.4f}")
    print(f"Val ROC-AUC : {roc_auc_score(y_true, y_pred):.4f}")

    # ── Coefficient summary ────────────────────────────────────────────────
    print("\n=== Coefficient Posterior Summary (top 10) ===")
    coef_df = bayes_model.get_coefficient_summary()
    print(coef_df.head(10)[
        ["feature", "mean", "sd", "hdi_3%", "hdi_97%", "significant", "direction"]
    ].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    bayes_model.save(_OUTPUT_DIR)
    print(f"\nAll artefacts saved to: {_OUTPUT_DIR}")
