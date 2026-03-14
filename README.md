# fmri-elastic-net

A Python pipeline for activation and connectome predictive modeling
using elastic net regularization. Designed for SLURM high-performance computing
environments, with a YAML configuration interface and a 3-stage job orchestration
pattern (main analysis → permutation workers → aggregation).

---

## Overview

The pipeline implements nested cross validation, elastic net regression (single- or
multi-task) or classification (binary or multi-class) and optional feature dimensionality
reduction. Two analysis modes are available to suit different sample sizes
and inferential goals. Coefficient significance is based on bootstrap confidence intervals,
probability of direction (pd), and Benjamini-Hochberg FDR correction. Block permutation
testing quantifies the unique contribution of user-defined feature subsets.

---

## Installation

```bash
conda env create -f environment.yaml
conda activate fmri-elastic-net
```

**Requirements** (pinned to tested versions):
- Python 3.10
- numpy 2.2.x, pandas 2.3.x, scipy 1.15.x
- scikit-learn 1.7.x, hdbscan 0.8.x, joblib 1.5.x, pyyaml 6.0.x

---

## Quick Start

### 1. Copy configuration and run templates

```bash
cp config_template.yaml my_project/config.yaml
cp run_template.sh my_project/run.sh
```

### 2. Edit `config.yaml`

At minimum, set:
- `analysis_type`: `"regression"` or `"classification"`
- `analysis_mode`: `"predict"` or `"correlate"` (see Analysis Modes below)
- `covariate_method`: `"none"`, `"incorporate"`, or `"pre_regress"`
- `feature_reduction_method`: `"none"`, `"cluster_pca"`, `"apriori"`, or `"ica"`
- `paths.data_file`: absolute path to input CSV
- `paths.output_dir`: absolute path to output directory
- `cv_params.n_outer_folds`, `cv_params.n_inner_folds`: fold counts (or `"loo"`)
- `cv_params.n_random_search_iter`: hyperparameter search iterations (no default; required)

### 3. Run locally (single machine)

```bash
python fmri-elastic-net.py --config /path/to/config.yaml
```

### 4. Run on SLURM (recommended for large datasets)

Edit `run.sh` to set paths and resource parameters, then submit:

```bash
sh run.sh
```

The orchestrator (`run_fmri-elastic-net.sh`) submits three dependent SLURM jobs:

| Stage | Job name | Description |
|-------|----------|-------------|
| 1 | `EN_Main` | Nested CV, selection frequency, bootstrap, block permutation |
| 2 | `EN_Worker` (array) | Permutation null distribution (parallelized across array jobs) |
| 3 | `EN_Agg` | Aggregates permutation chunks, computes final p-value |

---

## Data Format

The input data file must be a CSV with one row per subject. Required columns:
- **Subject ID**: any string or integer identifier (specified by `data_cols.subject_id_col`)
- **Outcome**: numeric continuous (regression) or integer class labels (classification); specified by `data_cols.post_score_col`
- **Brain features**: numeric columns identified by a substring match (e.g., all columns named `brain_*`); specified by `data_cols.brain_feature_substr`
- **Covariates** (optional): numeric columns; specified by `data_cols.covariate_cols`

Rows with any missing values are removed by listwise deletion before analysis.

---

## Outcome Types

The pipeline supports four distinct outcome types, determined by `analysis_type` and the
shape of the outcome column(s):

| Outcome type | `analysis_type` | Outcome format | sklearn estimator |
|---|---|---|---|
| Single-task regression | `"regression"` | One continuous numeric column | `ElasticNet` |
| Multi-task regression | `"regression"` | Multiple continuous numeric columns | `MultiTaskElasticNet` |
| Binary classification | `"classification"` | One integer column, 2 unique labels | `LogisticRegression(penalty='elasticnet', solver='saga')` |
| Multi-class classification | `"classification"` | One integer column, 3+ unique labels | `LogisticRegression(penalty='elasticnet', solver='saga')` with OVR decomposition |

Multi-task regression (`MultiTaskElasticNet`) has two important constraints:
- **Shared sparsity**: all tasks share the same feature sparsity. Features are jointly
  selected (non-zero) or jointly excluded (zero) for all tasks simultaneously (assumes
  a shared 'system' for predictive features). If different tasks are driven by
  different feature subsets, consider running separate single-task analyses instead.
- **Sample weights via sqrt(w) transformation**: `sample_weight_col` is supported for
  `MultiTaskElasticNet` via a `WeightTransformer` pipeline step that scales both X and Y by
  sqrt(w_i) before fitting. This is algebraically equivalent to weighted least squares in the
  loss term. The L1+L2 penalty is invariant to this transformation; alpha is re-tuned on the
  transformed data.

Multi-task regression is triggered automatically when `data_cols.post_score_col` refers
to multiple columns in the data file.

---

## Analysis Modes

Choosing the right analysis mode is the most consequential configuration decision.
It controls both the L1 ratio search space and the implicit regularization.

### `predict` — Full-spectrum elastic net

- L1 ratio search space: 0.01–0.99 (full elastic net; data-driven regularization balance)
- Model performance evaluated via external validity (nested CV R² or AUC)
- Emphasis on generalization
- **Best use case:** large samples (N ~ 1,000s) where generalization to unseen data
  is the primary scientific question; nested CV selects the optimal L1/L2 balance
  for the data, including Ridge-dominant solutions when features are multicollinear
  or signal is diffuse

### `correlate` — Ridge-dominant elastic net

- L1 ratio search space: 0.001–0.2 (dense, shrinkage-focused solutions)
- Emphasis on stable coefficient estimation; Ridge-dominant regularization reduces variance
  at the cost of sparsity, suitable for multicollinear feature sets
- Model performance is still evaluated via nested CV on held-out folds, but the
  primary goal is reliable, non-zero feature attribution (internal validity)
- **Best use case:** small-to-medium samples (N ~ 100s) where overfitting to noise
  or P>>N is a concern; scenarios where interpretability of all features is desired
  rather than sparse selection

---

## Covariate Methods

| Method | Description | Best use case |
|--------|-------------|---------------|
| `none` | No covariates included | No nuisance variables to control |
| `incorporate` | Covariates entered as features with tunable penalty weight | Covariates are substantively interesting predictors alongside brain features |
| `pre_regress` | Outcome residualized on covariates fold-locally before prediction | Covariates are pure nuisance variables; their unique contribution should be removed |

When `incorporate` is used, a `covariate_penalty_weight` hyperparameter is searched
via a loguniform distribution over `[model_params.covariate_penalty_weight_min,
model_params.covariate_penalty_weight_max]` to control regularization applied to
only covariate columns. Because this adds a third continuous hyperparameter to the
search space, `cv_params.n_random_search_iter` should be increased.

---

## Sample Weights

An optional `data_cols.sample_weight_col` column of non-negative observation weights
is supported for all estimators:

- **ElasticNet / LogisticRegression**: weights passed natively to sklearn's
  `sample_weight` argument; sklearn normalizes internally.
- **MultiTaskElasticNet**: weights applied via sqrt(w_i) pre-transformation of X and Y
  (algebraically equivalent to weighted least squares in the loss term; alpha is
  re-tuned on the transformed data via `WeightTransformer`).

The pipeline normalizes weights to sum to N at load time (Hajek estimator convention;
Lumley, 2010). **Only relative weights matter**; the absolute scale of the raw weight
column has no effect on model behavior. The effective sample size
ESS = (Σwᵢ)² / Σwᵢ² is logged at runtime. ESS/N < 0.5 triggers a warning:
extreme weight heterogeneity reduces effective sample size below half the nominal N,
which may compromise regularization path stability and bootstrap coverage.
No built-in weight trimming is applied — users should examine their weight distribution
and consider trimming extreme weights as a sensitivity analysis before running the
full pipeline.

---

## Feature Reduction Methods

All reduction is applied fold-locally inside the CV loop to prevent data leakage.

| Method | Description | Best use case |
|--------|-------------|---------------|
| `none` | Raw features passed directly to model | Small–medium feature sets; no assumed structure |
| `cluster_pca` | HDBSCAN clustering + 1-component PCA per cluster (fit inside CV) | Questionnaire items, genomic data, or any features expected to form discrete non-overlapping groups |
| `apriori` | Externally-defined cluster map + 1-component PCA per cluster (fit inside CV) | Pre-defined brain networks (e.g., atlas-based parcellation); network structure is theoretically motivated |
| `ica` | FastICA decomposition (fit inside CV); back-projection via activation patterns (Haufe et al., 2014) | Brain activation or connectivity data where regions participate in multiple overlapping networks |

**Note on leakage prevention:** Reduction is strictly fold-local throughout: in the
nested CV stage, reduction is fit on training data only. In the descriptive reporting
stages (selection frequency, bootstrap importance), each iteration fits a fresh reducer
clone on the resampled/subsampled data, preserving the conditional bootstrap framework.

---

## Statistical Inference

### Model performance p-value
Label permutation test: the full nested CV is repeated `n_permutations` times with
shuffled outcome labels. P-value uses Laplace correction:
`(count(null ≥ observed) + 1) / (n_permutations + 1)`.

### Feature importance
Bootstrap confidence intervals (conditional bootstrap: Efron & Tibshirani, 1993):
each iteration fits a fresh reducer clone on resampled brain features, fits the model using
fold-specific hyperparameters (fixed from each fold's inner-CV tuning), and back-projects
coefficients to the original feature space for aggregation. Tuning variance from the nested
CV is therefore propagated into the bootstrap CIs. This ensures meaningful CI and pd
computation across iterations with different reduced spaces.

- **`is_significant`**: primary criterion — CI does not cross zero
- **`is_significant_fdr`**: survives Benjamini-Hochberg FDR correction at q = 0.05
- **`pd`**: probability of direction; p-value approximation `p = 2*(1 - pd)` assumes
  a continuous coefficient distribution. For sparse features with high L1 regularization,
  zero-inflated bootstrap distributions cause `pd` to be near 0.5; use `is_significant`
  as the primary criterion in that case.

### Selection frequency
Repeated 50% subsampling (`n_fold_bootstraps` per fold). Per feature: proportion of
iterations with a non-zero coefficient. Descriptive only — no significance threshold.

### Block permutation
For each user-defined feature block: only that block's columns are row-permuted and
the full nested CV is rerun. Quantifies the unique predictive contribution of a feature
subset (e.g., activation from a specific brain network) beyond the remaining features.
Requires the user to label feature columns with a "block" identifier (e.g. if we want to
know the unique predictive contribution of fronto-limbic "FL" circuits relative to the
rest of the brain, block permutation is applied to features that contain the string
"FL" in its column header).

---

## Output Files

All output files are written to `paths.output_dir`:

| File | Description |
|------|-------------|
| `nested_cv_scores.csv` | Observed model performance (R² or AUC) |
| `model_performance.csv` | Comprehensive evaluation metrics (regression: RMSE, MAE, R², Pearson r; classification: AUC-ROC, Log-Loss, Sensitivity, Specificity, Balanced Accuracy) |
| `confusion_matrix.csv` | Confusion matrix (multi-class classification only) |
| `permutation_null_distribution_{metric}.csv` | Null distribution from label permutation |
| `permutation_result.csv` | Observed score, p-value, n_permutations (aggregate mode) |
| `report_selection_frequency.csv` | Subsampling-based selection frequency per feature/component |
| `report_feature_importance.csv` | Bootstrap CIs, pd, FDR flags per feature/component (all `feature_reduction_method` values) |
| `report_cluster_importance.csv` | Bootstrap CIs, pd, FDR flags at cluster level (`apriori`) |
| `report_block_permutation.csv` | Block-specific observed score and p-value |
| `report_fold_ensemble_importance.csv` | Tier 1 inference: fold-wise t-test mean/SD/CV, t-statistic, p-value, CI, and significance flags per feature |
| `report_fold_diagnostics.csv` | Per-fold hyperparameter records (alpha/C, l1_ratio, penalty_weight) from nested CV |
| `report_fold_params_summary.csv` | Mean ± SD, min, max across K folds for each hyperparameter |
| `report_fold_bootstrap_ci.csv` | Tier 2 inference: pooled fold-wise bootstrap percentile CIs and significance flags per feature |
| `cluster_loadings.csv` | PCA loadings per cluster (`cluster_pca` or `apriori`) |
| `cluster_loadings_fold_{n}.csv` | Per-fold PCA loadings for transparency |
| `ica_mixing_matrix.csv` | ICA mixing matrix A (P × K), activation pattern basis |
| `ica_mixing_matrix_fold_{n}.csv` | Per-fold ICA mixing matrix for transparency |
| `report_{level}_plotting.csv` | Subject-level feature vs. outcome data for visualization |
| `bootstrap_coef_distribution.npz` | Full bootstrap coefficient array (when `save_distributions: true`) |
| `block_perm_null_{label}.csv` | Block-specific permutation null scores (when `save_distributions: true`) |
| `pipeline.log` | Full logging output with timing and diagnostics |

For multi-task regression or multi-class classification, per-task/per-class output
files are written to `output_dir/task_{label}/` subdirectories, with aggregate
summaries written to the top-level `output_dir`.

---

## Configuration Reference

See `config_template.yaml` for all parameters with inline comments and defaults.
See `INPUT_SPECIFICATION.md` for the complete exhaustive specification including
parameter types, ranges, constraints, output schemas, and known edge cases.

---

## Known Limitations

- Bootstrap importance uses fold-specific hyperparameters from each fold's inner-CV
  tuning (not full-dataset hyperparameters). Tuning variance is propagated into Tier 2
  CIs. Each bootstrap iteration re-fits a fresh reducer clone (per-iteration re-reduction),
  preserving the conditional bootstrap framework across iterations with different reduced
  spaces.
- Tier 2 bootstrap CIs are computed from a pooled mixture of iterations with different
  fold-specific hyperparameter configurations. Percentile CIs from this mixture may have
  sub-nominal coverage for threshold-adjacent features; they are best interpreted as
  sensitivity diagnostics (Efron & Tibshirani, 1993, Ch. 13).
- `raw_coef_mean` for reduction methods (cluster_pca, apriori, ica) is approximate:
  the back-projected standardized coefficient is divided by original-feature SD, which
  is not equivalent to a standardized beta from direct regression on original features.
  `std_coef_mean` and `pd` are the primary inferential quantities.
- Selection frequency magnitudes may be elevated because hyperparameters are fixed from
  full-N tuning while each subsample uses N/2. Relative ordering is
  preserved; no significance threshold is applied (purely descriptive).
- LOO cross-validation disables `n_inner_repeats` (repeated CV is undefined for LOO).
