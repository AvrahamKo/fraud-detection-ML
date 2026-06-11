# E-Commerce Fraud Decisioning & Analytics Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2980B9?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyMC-Bayesian%20Inference-E67E22?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-00BFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud-27AE60?style=for-the-badge"/>
</p>

---

## Mission

A hybrid, production-grade fraud decisioning framework that combines a **high-throughput gradient boosting pipeline** (LightGBM) for automated transaction blocking with a **probabilistic programming layer** (PyMC Bayesian Logistic Regression via ADVI) to capture and exploit epistemic uncertainty under temporal concept drift.

The core thesis: point-estimate models are blind to their own ignorance. When adversarial fraud patterns mutate faster than retraining cycles, the *width* of a model's posterior distribution is as operationally valuable as the *mean*. High epistemic uncertainty is a direct signal for human review routing — independent of the raw fraud score.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IEEE-CIS Dataset                             │
│              590,540 transactions · 434 raw features                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  01_eda.ipynb   │  Class imbalance · Missingness
                    │  EDA Layer      │  signals · Temporal cyclicality
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   splitter.py   │  Chronological train/val/test
                    │  Time-Aware     │  split — zero future leakage
                    │     Split       │  60% / 20% / 20%
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  pipeline.py    │  Velocity (1h/24h/7d rolling)
                    │  Feature Eng.   │  Spend-spike ratios · Identity
                    │   ~54 features  │  mismatch · Temporal flags
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐         ┌──────────▼──────────┐
     │  lgbm_trainer   │         │     bayes_lr.py      │
     │  LightGBM +     │         │  PyMC ADVI · Top-15  │
     │  Optuna/PR-AUC  │         │  interpretable feats │
     │  488 features   │         │  Posterior uncertainty│
     └────────┬────────┘         └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  evaluator.py   │  Metrics matrix · PR/ROC curves
                    │  Head-to-Head   │  Calibration · FN recovery
                    │   Evaluation    │  Disagreement analysis
                    └─────────────────┘
```

---

## Analytical Results — Test Set

> **Test partition:** 118,108 transactions · **3.44% fraud rate** (severe imbalance) · strictly chronological, never seen during training or tuning.

### Metrics Matrix

| Metric | LightGBM | Bayesian LR | Edge |
|---|---|---|---|
| **PR-AUC** *(primary)* | **0.4618** | 0.0895 | LightGBM |
| ROC-AUC | **0.8699** | 0.7176 | LightGBM |
| Gini Coefficient | **0.7398** | 0.4351 | LightGBM |
| KS Statistic | **0.5845** | 0.3601 | LightGBM |
| Brier Score ↓ | 0.0411 | **0.0346** | Bayesian |
| **ECE (Calibration) ↓** | 0.0804 | **0.0084** | Bayesian |
| F1 @ Optimal Threshold | **0.4569** | 0.1893 | LightGBM |

> PR-AUC is the primary metric throughout. ROC-AUC is a secondary reference — at 3.44% fraud prevalence, a trivial model achieves ~0.96 ROC-AUC by always predicting legitimate. PR-AUC directly penalises poor precision at high recall and is the standard in production payments risk.

---

## Strategic Insights

### 1 · Temporal Concept Drift Validates the Hybrid Architecture

LightGBM's PR-AUC dropped from **0.56 on the validation set to 0.46 on the test set** — a measurable degradation attributable to temporal concept drift. Fraud behaviour shifted between the validation and test time windows, and the gradient booster, lacking a mechanism to express its own uncertainty, continued to output confident scores on patterns it had degraded coverage of.

This is the foundational justification for the hybrid architecture. A model that *knows* it's operating outside its training distribution can flag that uncertainty for human review before a chargeback is filed.

### 2 · The Calibration Masterclass

The Bayesian LR achieved an **Expected Calibration Error of 0.0084** against LightGBM's **0.0804** — a 9.6× improvement. This is not a minor academic distinction.

In a live payments risk engine, a model score is not used only for binary decisioning. It feeds:
- **Risk-based pricing**: insurance premiums, surcharges on high-risk merchants
- **Bayesian updating**: downstream models treating the score as a prior
- **Regulatory reporting**: regulators expect that a score of 0.30 actually corresponds to ~30% fraud probability

LightGBM's scores are miscalibrated — a score of 0.30 is a rank, not a probability. The Bayesian model's scores *are* probabilities in the calibration-theoretic sense, making them directly usable in financial risk pricing without post-hoc isotonic recalibration.

### 3 · The Disagreement Regime

The two models disagreed on **13.36% of test transactions** (|LGBM score − Bayes mean| > 0.20). Within this disagreement regime:

| Model | Accuracy in Disagreements |
|---|---|
| **Bayesian LR** | **80.0%** |
| LightGBM | 72.2% |

Transactions where the models disagree are precisely the *ambiguous, boundary-case* transactions — the ones where a gradient booster's inductive bias diverges from the Bayesian posterior. In this regime, the Bayesian model's richer uncertainty representation translates directly into higher accuracy.

---

## The Business Case: False-Negative Recovery via Uncertainty Routing

This section quantifies the operational ROI of the hybrid system — the argument you make to a fraud operations director, not just a data science team.

### The Problem

At its optimal operating threshold, **LightGBM missed 2,602 fraud transactions** — a **64.0% miss rate** on the test set's 4,064 fraud cases. These are chargebacks waiting to happen. At an average e-commerce order value of ~$100 and a chargeback penalty multiplier of ~1.5×, each missed fraud costs the merchant ~$150. The 2,602 missed frauds represent a **~$390,000 exposure in a single time window**.

### The Bayesian Uncertainty Signal

The Bayesian model's epistemic uncertainty (HDI width) is **1.61× higher on fraud transactions than on legitimate ones**. This is the key finding: even when LightGBM scores a fraud transaction as legitimate (a false negative), the Bayesian model tends to flag it as uncertain — because it genuinely hasn't seen enough signal to be confident.

### The Simulation

By routing all transactions with Bayesian HDI width above the **p75 uncertainty threshold** into a Manual Review Queue:

```
┌─────────────────────────────────────────────────────────────────┐
│  MANUAL REVIEW QUEUE — SIMULATION RESULTS (Test Set)           │
├─────────────────────────────────────────────────────────────────┤
│  Operating point          │  p75 Bayesian uncertainty           │
│  Review queue size        │  25.0% of all transactions          │
│  LightGBM FNs recovered   │  41.1% of missed frauds             │
│  Queue precision          │  8.25% (vs 3.44% baseline rate)     │
│  Precision lift           │  2.4× above random sampling         │
└─────────────────────────────────────────────────────────────────┘
```

In practical terms: a review team spending 25% of its capacity on the uncertainty-flagged queue will surface **4 in 10 frauds that the automated system missed**, with **2.4× better precision** than random sampling. This is a measurable, CFO-presentable reduction in chargeback exposure — without increasing false positives or blocking more legitimate customers.

The false-negative recovery curve across all operating points is saved to `reports/figures/fig_04_fn_recovery_curve.png`.

---

## Repository Structure

```
fraud-detection-ML/
│
├── data/
│   ├── raw/                        # IEEE-CIS source CSVs (not tracked by git)
│   ├── interim/                    # Merged parquet (post-ingestion)
│   └── processed/
│       ├── train.parquet           # Time-aware splits
│       ├── val.parquet
│       ├── test.parquet
│       └── features/               # Post-pipeline enriched splits
│
├── models/
│   ├── lgbm_v1/
│   │   ├── model.txt               # LightGBM booster (portable text format)
│   │   ├── trainer_meta.json       # Feature names, config, scale_pos_weight
│   │   ├── best_params.json        # Optuna best hyperparameters
│   │   ├── optuna_study.pkl        # Full trial history
│   │   └── feature_importance_gain.csv
│   └── bayes_lr_v1/
│       ├── trace.nc                # ArviZ InferenceData (NetCDF4)
│       ├── alpha_draws.npy         # Cached intercept posterior samples
│       ├── beta_draws.npy          # Cached coefficient posterior samples
│       ├── scaler.pkl              # Fitted StandardScaler
│       ├── elbo_history.npy        # ADVI convergence curve
│       ├── coefficient_summary.csv # Posterior means + 94% HDI per feature
│       └── meta.json
│
├── notebooks/
│   └── 01_eda.ipynb                # EDA: imbalance, missingness, temporal
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py               # Load + merge + memory optimisation (~55% RAM reduction)
│   │   └── splitter.py             # Chronological split with overlap assertions
│   ├── features/
│   │   ├── velocity.py             # Rolling count/sum/mean: 3 entities × 3 windows
│   │   ├── behavioral.py           # Spend spikes, temporal flags, amount ratios
│   │   ├── identity_mismatch.py    # Email mismatch, address consistency, identity presence
│   │   └── pipeline.py             # FeaturePipeline: fit/transform with leakage boundary
│   ├── models/
│   │   ├── traditional/
│   │   │   └── lgbm_trainer.py     # LightGBM + Optuna on PR-AUC + SHAP
│   │   └── bayesian/
│   │       └── bayes_lr.py         # PyMC ADVI + numpy posterior caching
│   └── evaluation/
│       └── evaluator.py            # Head-to-head metrics + 5 report figures
│
├── reports/
│   ├── figures/                    # All generated PNGs
│   └── model_comparison_metrics.json
│
├── tests/
│   ├── test_splitter.py
│   ├── test_behavioral.py
│   └── test_velocity.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 0 · Environment Setup

```bash
git clone https://github.com/AvrahamKo/fraud-detection-ML.git
cd fraud-detection-ML
pip install -r requirements.txt
pip install -e .   # installs the src/ package so imports work without sys.path hacks
```

Place the IEEE-CIS Kaggle files into `data/raw/`:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 1 · Ingest & Split

```bash
# Merge, memory-optimise (~55% RAM reduction), and split chronologically
python -m src.ingestion.splitter
```

Outputs `data/processed/train.parquet`, `val.parquet`, `test.parquet`.

### 2 · Feature Engineering

```bash
# Fit encoders on train, transform all splits — zero leakage across boundary
python -m src.features.pipeline
```

Outputs enriched parquets to `data/processed/features/`. Adds ~54 engineered features (velocity, behavioral, identity mismatch) to the base 434 columns.

> **Runtime note:** Velocity features iterate over ~13.5k `card1` groups using `groupby.apply` with `closed="left"` rolling windows. Expect ~5–10 minutes on a modern laptop. For production scale, replace with PySpark `Window.rangeBetween`.

### 3 · Train LightGBM

```bash
python -m src.models.traditional.lgbm_trainer
```

Runs 10 Optuna trials (each capped at 300 rounds with `early_stopping_rounds=15`) optimising PR-AUC on the validation set. Saves model, Optuna study, and feature importance to `models/lgbm_v1/`.

### 4 · Train Bayesian Model

```bash
python -m src.models.bayesian.bayes_lr
```

Runs ADVI on a stratified 50k-row subsample (guaranteeing ≥2,000 fraud examples) for 30,000 gradient steps. Posterior draws are cached as raw NumPy arrays for sub-millisecond inference. Saves ArviZ InferenceData trace, scaler, and coefficient summary to `models/bayes_lr_v1/`.

### 5 · Evaluate

```bash
python -m src.evaluation.evaluator
```

Loads both models, runs predictions on the test set, produces the 5 report figures, prints the executive summary, and saves `reports/model_comparison_metrics.json`.

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **`closed="left"` rolling windows** | Excludes the current transaction from its own velocity calculation — prevents temporal data leakage |
| **`fit` / `transform` split for encoders** | Amount means and address modes are computed on training data only; val/test use stored lookups |
| **PR-AUC as LightGBM eval function** | Aligns training loop, early stopping, and Optuna objective on the same metric |
| **`logit_p=` in PyMC Bernoulli** | Numerically stable sigmoid avoids gradient saturation during ADVI optimisation |
| **Posterior draws cached as NumPy** | Bypasses PyMC graph overhead at inference time; enables ~50k rows/second prediction throughput |
| **JSON over pickle for model metadata** | Pickle stores `__main__.ClassName` — breaks when loading from a different entry point |
| **`reference=dtrain` on LightGBM val set** | Enforces identical feature binning between train and val; without it, early stopping is comparing incomparable feature spaces |

---

## Technical Stack

| Layer | Tools |
|---|---|
| Data | pandas, NumPy, PyArrow (Parquet) |
| Feature Engineering | pandas rolling/groupby, scikit-learn StandardScaler, LabelEncoder |
| Traditional ML | LightGBM, Optuna (TPE sampler + MedianPruner), SHAP |
| Bayesian ML | PyMC (ADVI), ArviZ, SciPy (sigmoid) |
| Evaluation | scikit-learn metrics, matplotlib, seaborn |
| Persistence | LightGBM `.txt`, ArviZ NetCDF4, JSON, NumPy `.npy` |

---

## Generated Figures

All figures are saved to `reports/figures/` after running the evaluator.

| File | Contents |
|---|---|
| `fig_01_pr_roc_curves.png` | PR and ROC curves for both models on a single canvas |
| `fig_02_uncertainty_distribution.png` | Bayesian HDI width histogram + CDF split by fraud/legit class |
| `fig_03_lgbm_vs_bayes_scatter.png` | Score agreement scatter, coloured by epistemic uncertainty |
| `fig_04_fn_recovery_curve.png` | FN recovery rate vs. review queue size across uncertainty thresholds |
| `fig_05_calibration.png` | Reliability diagrams (ECE annotated) for both models |

---

## Author

**Avraham Koslowsky**
Bayesian Machine Learning · Data Systems · Cyber Security
[ko.avraham@gmail.com](mailto:ko.avraham@gmail.com)

---

*Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — Kaggle competition hosted by IEEE Computational Intelligence Society and Vesta Corporation.*
