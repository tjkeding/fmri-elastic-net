# fmri-elastic-net

A free-standing Python pipeline for activation and connectome predictive modeling
using elastic net regularization. Designed for SLURM high-performance computing
environments, with a YAML configuration interface and a 3-stage job orchestration
pattern (main analysis → permutation workers → aggregation).

---

## Overview

The pipeline implements nested cross-validated elastic net regression or classification
with optional feature dimensionality reduction applied fold-locally inside the CV loop
to prevent data leakage. Two analysis modes are available to suit different sample sizes
and inferential goals. Statistical inference is based on bootstrap confidence intervals,
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

## Analysis Modes

Choosing the right analysis mode is the most consequential configuration decision.
It controls both the L1 ratio search space and the implicit regularization philosophy.

### `predict` — LASSO-dominant elastic net

- L1 ratio search space: 0.5–0.99 (sparse, feature-selecting solutions)
- Model performance evaluated via external validity (nested CV R² or AUC)
- Emphasis on generalization and sparsity
- **Best use case:** large samples (N ~ 1,000s) where generalization to unseen data
  is the primary scientific question; high-dimensional feature sets where true sparsity
  is expected; feature selection as an objective in itself

### `correlate` — Ridge-dominant elastic net

- L1 ratio search space: 0.001–0.2 (dense, shrinkage-focused solutions)
- Model performance evaluated via internal validity
- Emphasis on stable coefficient estimation and multicollinear feature sets
- **Best use case:** small-to-medium samples (N ~ 100s) where overfitting is a concern;
  brain connectivity/activation data with high inter-feature correlation; scenarios where
  interpretability of all features is desired rather than sparse selection

---

## Covariate Methods

| Method | Description | Best use case |
|--------|-------------|---------------|
| `none` | No covariates included | No nuisance variables to control |
| `incorporate` | Covariates entered as features with tunable penalty weight | Covariates are substantively interesting predictors alongside brain features |
| `pre_regress` | Outcome residualized on covariates fold-locally before prediction | Covariates are pure nuisance variables; their unique contribution should be removed |

When `incorporate` is used, a `covariate_penalty_weight` hyperparameter is searched
over the `model_params.covariate_penalty_weights` list to modulate regularization
applied to covariate columns relative to brain features.

---

## Feature Reduction Methods

All reduction is applied fold-locally inside the CV loop to prevent data leakage.

| Method | Description | Best use case |
|--------|-------------|---------------|
| `none` | Raw features passed directly to model | Small–medium feature sets; no assumed structure |
| `cluster_pca` | HDBSCAN clustering + 1-component PCA per cluster (fit inside CV) | Questionnaire items, genomic data, or any features expected to form discrete non-overlapping groups |
| `apriori` | Externally-defined cluster map + 1-component PCA per cluster (fit inside CV) | Pre-defined brain networks (e.g., atlas-based parcellation); network structure is theoretically motivated |
| `ica` | FastICA decomposition (fit inside CV); back-projection via activation patterns (Haufe et al., 2014) | Brain activation or connectivity data where regions participate in multiple overlapping networks |

**Note on leakage prevention:** For the inferential nested CV stage, reduction is
strictly fold-local (fit on training data only). For the descriptive reporting stages
(selection frequency, bootstrap importance), reduction is fit on the full dataset; a
log message explicitly flags this distinction.

---

## Statistical Inference

### Model performance p-value
Label permutation test: the full nested CV is repeated `n_permutations` times with
shuffled outcome labels. P-value uses Laplace correction:
`(count(null ≥ observed) + 1) / (n_permutations + 1)`.

### Feature importance
Bootstrap confidence intervals (Approach Y conditional bootstrap): each iteration fits
a fresh reducer clone on resampled brain features, fits the model using hyperparameters
fixed from full-data tuning, and back-projects coefficients to the original feature
space for aggregation. This ensures meaningful CI and pd computation across iterations
with different reduced spaces.

- **`is_significant`**: primary criterion — CI does not cross zero
- **`is_significant_fdr`**: survives Benjamini-Hochberg FDR correction at q = 0.05
- **`pd`**: probability of direction; p-value approximation `p = 2*(1 - pd)` assumes
  a continuous coefficient distribution. For sparse features with high L1 regularization,
  zero-inflated bootstrap distributions cause `pd` to be near 0.5; use `is_significant`
  as the primary criterion in that case.

### Selection frequency
Repeated 50% subsampling (`n_bootstraps` iterations). Per feature: proportion of
iterations with a non-zero coefficient. Descriptive only — no significance threshold.

### Block permutation
For each user-defined feature block: only that block's columns are row-permuted and
the full nested CV is rerun. Quantifies the unique predictive contribution of a feature
subset (e.g., activation from a specific brain region) beyond the remaining features.

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
| `report_feature_importance.csv` | Bootstrap CIs, pd, FDR flags (`feature_reduction_method: none`) |
| `report_cluster_importance.csv` | Bootstrap CIs, pd, FDR flags at cluster level (`apriori`) |
| `report_individual_importance.csv` | Back-projected feature-level importance (`apriori`, `cluster_pca`, `ica`) |
| `report_block_permutation.csv` | Block-specific observed score and p-value |
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

- Bootstrap importance conditions on hyperparameters tuned on the full dataset
  (conditional bootstrap). CIs may be marginally narrower than a full double-bootstrap
  that re-tunes per iteration, but this is computationally tractable and standard
  practice for neuroimaging sample sizes.
- Feature reduction in selection frequency and bootstrap stages is fit on the full
  dataset (descriptive pathway). Inferential validity for the nested CV stage is
  preserved because reduction is fold-local there.
- `raw_coef_mean` for reduction methods (cluster_pca, apriori, ica) is approximate:
  the back-projected standardized coefficient is divided by original-feature SD, which
  is not equivalent to a standardized beta from direct regression on original features.
  `std_coef_mean` and `pd` are the primary inferential quantities.
- Selection frequency magnitudes may be systematically elevated because hyperparameters
  are fixed from full-N tuning while each subsample uses N/2. Relative ordering is
  preserved; no significance threshold is applied (purely descriptive).
- LOO cross-validation disables `n_inner_repeats` (repeated CV is undefined for LOO).
