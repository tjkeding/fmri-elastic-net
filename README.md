# fmri-elastic-net

A free-standing Python pipeline for activation and connectome predictive modeling
using elastic net regularization. Designed for SLURM high-performance computing
environments, with a YAML configuration interface and a 3-stage job orchestration
pattern (main analysis → permutation workers → aggregation).

---

## Overview

The pipeline implements nested cross-validated elastic net regression or binary
classification with optional feature dimensionality reduction (inside the CV loop
to prevent data leakage). Statistical inference is based on bootstrap confidence
intervals, probability of direction (pd), and Benjamini-Hochberg FDR correction.
Block permutation testing quantifies the unique contribution of user-defined feature
subsets (e.g., brain activation patterns).

**Key design principles:**

- All feature reduction is applied fold-locally inside the cross-validation loop.
- Bootstrap importance uses Approach Y: each iteration fits a fresh reducer clone on
  resampled brain features, fits the model in reduced space with fixed hyperparameters
  (tuned on full data), then back-projects coefficients to the invariant original feature
  space for aggregation (conditional bootstrap, Efron & Tibshirani, 1993). This ensures
  meaningful CI and pd computation across iterations with different reduced spaces.
- The `pd`-to-p-value conversion (`p = 2*(1 - pd)`) assumes a continuous coefficient
  distribution; for high-sparsity features with L1 regularization, the CI-based
  `is_significant` flag is the primary inference criterion and is unaffected by
  zero-inflation. The p-value and FDR columns are complementary.

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

## Usage

### Local (single machine)

```bash
python fmri-elastic-net.py --config /path/to/config.yaml
```

### SLURM (recommended for large datasets)

1. Copy `run_template.sh` and `config_template.yaml` to your project directory.
2. Edit paths and resource settings in both files.
3. Submit:

```bash
sh run_template.sh
```

The orchestrator (`run_fmri-elastic-net.sh`) submits three dependent SLURM jobs:

| Stage | Job name | Description |
|-------|----------|-------------|
| 1 | `EN_Main` | Nested CV, selection frequency, bootstrap, block permutation |
| 2 | `EN_Worker` (array) | Permutation null distribution (parallelized) |
| 3 | `EN_Agg` | Aggregates permutation chunks, computes final p-value |

---

## Configuration

All options are controlled by a YAML configuration file. Copy `config_template.yaml`
and edit the following required fields:

```yaml
analysis_type:       "regression"        # or "classification"
analysis_mode:       "predict"           # or "correlate"
covariate_method:    "none"              # "none", "incorporate", or "pre_regress"
feature_reduction_method: "none"         # "none", "cluster_pca", "apriori", or "ica"

paths:
  data_file:         "/absolute/path/to/data.csv"
  output_dir:        "/absolute/path/to/output/"
  apriori_clustering_file: null          # required only for feature_reduction_method: "apriori"

cv_params:
  n_outer_folds:     10
  n_inner_folds:     3
  n_inner_repeats:   5
  n_random_search_iter: 20              # REQUIRED; no default

stats_params:
  n_permutations:       10000
  n_bootstraps:         10000
  n_block_permutations: 500
  ci_level:             0.95
  save_distributions:   true            # saves .npz bootstrap array and block perm null CSVs
```

See `config_template.yaml` and `INPUT_SPECIFICATION.md` for all parameters with
descriptions, defaults, and valid ranges.

---

## Outputs

All output files are written to `paths.output_dir`:

| File | Description |
|------|-------------|
| `nested_cv_scores.csv` | Observed model performance score (R² or AUC) |
| `permutation_null_distribution_{metric}.csv` | Null distribution from label permutation |
| `permutation_result.csv` | Observed score, p-value, n_permutations (aggregate mode) |
| `report_selection_frequency.csv` | Bootstrap selection probability per feature/component |
| `report_feature_importance.csv` | Bootstrap CIs, pd, FDR flags (feature_reduction: none) |
| `report_cluster_importance.csv` | Bootstrap CIs, pd, FDR flags (apriori clusters) |
| `report_individual_importance.csv` | Back-projected feature-level importance (apriori, cluster_pca, ica) |
| `report_block_permutation.csv` | Block-specific observed score and p-value |
| `cluster_loadings.csv` | PCA loadings per cluster (cluster_pca or apriori) |
| `ica_mixing_matrix.csv` | ICA mixing matrix A (P × K), activation pattern basis |
| `model_performance.csv` | Comprehensive evaluation metrics (RMSE, MAE, R², Pearson r for regression; AUC-ROC, Log-Loss, Sensitivity, Specificity for classification) |
| `confusion_matrix.csv` | Confusion matrix (multi-class classification only) |
| `bootstrap_coef_distribution.npz` | Full bootstrap coefficient array as compressed NumPy archive (when `save_distributions: true`) |
| `block_perm_null_{label}.csv` | Block-specific permutation null scores (when `save_distributions: true`) |
| `cluster_loadings_fold_{n}.csv` | Per-fold PCA loadings (cluster_pca or apriori; one file per outer fold) |
| `ica_mixing_matrix_fold_{n}.csv` | Per-fold ICA mixing matrix (ica; one file per outer fold) |
| `report_{level}_plotting.csv` | Subject-level feature vs. outcome data for visualization |
| `pipeline.log` | Full logging output with timing and diagnostics |

---

## Analysis Modes

**`predict`** (LASSO-heavy elastic net):
Uses high L1 ratios (0.5–0.99). Model performance evaluated via external validity
(nested CV). Appropriate for large samples (N ~ 1000s) where generalization is
the primary question.

**`correlate`** (Ridge-heavy elastic net):
Uses low L1 ratios (0.001–0.2). Model performance evaluated via internal validity.
More appropriate for small samples (N ~ 100s). Feature importance based on
bootstrap coefficients rather than sparsity.

---

## Feature Reduction Methods

| Method | Description | Use case |
|--------|-------------|----------|
| `none` | Raw features passed directly to model | Small–medium feature sets |
| `cluster_pca` | HDBSCAN clustering + 1-component PCA per cluster (fit inside CV) | Questionnaire, genomic data |
| `apriori` | External cluster map + 1-component PCA per cluster (fit inside CV) | Pre-defined brain networks |
| `ica` | FastICA decomposition (fit inside CV); back-projection via activation patterns per Haufe et al. (2014) | Brain activation/connectivity data with overlapping networks |

---

## Statistical Inference

**Feature significance** is determined by the bootstrap CI for the standardized
coefficient not crossing zero (`is_significant`). Significance after BH-FDR
correction is reported in `is_significant_fdr`.

**Probability of direction** (`pd`) is the proportion of bootstrap samples with
the same sign as the mean coefficient. The p-value approximation `p = 2*(1 - pd)`
follows Makowski et al. (2019). For sparse features where many bootstrap samples
produce exactly zero, `pd` may fall below 0.5 (yielding `p = 1.0` after clipping);
in such cases, `is_significant` from the CI remains the valid primary criterion.

**Block permutation tests** assess the unique contribution of a feature block by
permuting that block's rows and rerunning the full nested CV. Each block's null
distribution is generated independently.

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
- Selection frequency magnitudes may be systematically inflated because hyperparameters
  are fixed from full-N tuning while each subsample uses N/2. Relative ordering is
  preserved; no significance threshold is applied (purely descriptive).
- LOO cross-validation disables `n_inner_repeats` (repeated CV is undefined for LOO).

---

## Testing

```bash
cd /path/to/fmri-elastic-net
conda activate fmri-elastic-net
pytest tests/ -v
```

Test suite: 418 tests across the `tests/` directory.
