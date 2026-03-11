# Input Specification: fmri-elastic-net

Complete specification of all configuration parameters, file formats, environment
requirements, and edge-case behaviors. Optimized for reproducibility and automated
consumption.

---

## 1. Command-Line Interface

```
python fmri-elastic-net.py --config CONFIG [--mode MODE] [--job_id JOB_ID]
                            [--n_jobs N_JOBS] [--skip_main_perm]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | *required* | Absolute path to YAML configuration file |
| `--mode` | str | `main` | Execution mode: `main`, `perm_worker`, or `aggregate` |
| `--job_id` | int | `0` | SLURM array task index (used in `perm_worker` mode) |
| `--n_jobs` | int | `1` | Total number of permutation worker jobs (used in `perm_worker` mode) |
| `--skip_main_perm` | flag | `False` | If set in `main` mode, skips the local permutation test (use when SLURM workers handle permutations) |

### Execution Modes

**`main`**: Runs nested CV (Step 4), selection frequency (Step 6), bootstrap (Step 7),
optionally local permutation test (Step 5), and block permutation (Step 8).

**`perm_worker`**: Runs a subset of permutation iterations (subset determined by
`job_id` and `n_jobs`). Writes `perm_chunk_{job_id}.csv` to output_dir.

**`aggregate`**: Reads all `perm_chunk_*.csv` files and `nested_cv_scores.csv` from
output_dir, computes final p-value, writes `permutation_result.csv`.

---

## 2. Data File (`paths.data_file`)

**Format:** CSV with a header row. One row per subject.

**Required columns** (names controlled by `data_cols.*`):
- Subject ID column (string or integer): specified by `data_cols.subject_id_col`
- Outcome column (numeric): specified by `data_cols.post_score_col`
- Optional covariate columns (numeric): specified by `data_cols.covariate_cols`
- Brain feature columns (numeric): identified by substring `data_cols.brain_feature_substr`

**Missing data**: Rows with any `NaN` or missing values are removed via listwise deletion
before any analysis. A warning is logged with the count of removed rows. If all rows
are removed, the pipeline raises a `ValueError` and exits.

**Classification outcome**: Must be binary (two distinct integer or float values).
Multi-class classification is not currently supported.

**Regression outcome**: Any continuous numeric column.

**Optional column** (advanced):
- `data_cols.sample_weight_col`: Column of non-negative sample weights. If present,
  weights are passed to the model's `fit` call (where supported by sklearn estimator).
  `MultiTaskElasticNet` does not support `sample_weight`; weights are silently ignored
  for that estimator.

---

## 3. Apriori Clustering File (`paths.apriori_clustering_file`)

**Required only when** `feature_reduction_method: "apriori"`.

**Format:** 2-column CSV, no header (header row is ignored). Column 1: feature name
(must exactly match column names in `data_file`). Column 2: cluster label (any
hashable type; strings and integers both accepted).

**Constraint:** Every brain feature column in `data_file` (after substring filtering)
must have an entry in this file. A `ValueError` is raised for any unmatched features.

---

## 4. YAML Configuration Parameters

### Top-Level Required Parameters

| Parameter | Type | Valid values | Description |
|-----------|------|-------------|-------------|
| `analysis_type` | str | `"regression"`, `"classification"` | Determines model class and scoring metric |
| `analysis_mode` | str | `"predict"`, `"correlate"` | Controls L1 ratio search space and implicit regularization philosophy |
| `covariate_method` | str | `"none"`, `"incorporate"`, `"pre_regress"` | How covariates are handled |
| `feature_reduction_method` | str | `"none"`, `"cluster_pca"`, `"apriori"`, `"ica"` | Dimensionality reduction method |

### `paths` Section

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paths.data_file` | str | Yes | Absolute path to input CSV |
| `paths.output_dir` | str | Yes | Absolute path to output directory (created if absent) |
| `paths.apriori_clustering_file` | str or null | Only when `apriori` | Absolute path to 2-column cluster map CSV |

### `n_cores` (integer, required)

Number of CPU cores for joblib `Parallel`. Should match `cpus_per_task` in the
SLURM wrapper. Controls parallelism in permutation tests, bootstrap, and inner CV.

### `data_cols` Section

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data_cols.subject_id_col` | str | Yes | Column name for subject IDs; removed from features |
| `data_cols.post_score_col` | str | Yes | Column name for outcome variable |
| `data_cols.covariate_cols` | list of str | Only when `covariate_method != "none"` | Covariate column names |
| `data_cols.brain_feature_substr` | str | Yes | Substring match to identify brain feature columns |
| `data_cols.sample_weight_col` | str or null | No | Optional sample weight column name |

### `clustering_params` Section (used only when `feature_reduction_method: "cluster_pca"`)

| Parameter | Type | Default | Valid values | Description |
|-----------|------|---------|-------------|-------------|
| `clustering_params.min_cluster_size` | int or `"auto"` | `"auto"` | Integer ≥ 2, or `"auto"` | HDBSCAN minimum cluster size. `"auto"` computes `max(3, ceil(log2(P) * (100/N)^0.25))` |
| `clustering_params.distance_metric` | str | — | `"pearson"`, `"spearman"` | Pairwise similarity metric for distance matrix |
| `clustering_params.sign_handling` | str | — | `"unsigned"`, `"signed"` | `"unsigned"`: `1 - |sim|`; `"signed"`: `(1 - sim) / 2`. Has no effect for `dcor` (planned) |

Distance matrix formula:
- `unsigned`: `sqrt(2 * (1 - |sim|))`
- `signed`: `sqrt(2 * (1 - sim))`

Both are proper metric space mappings into [0, 2].

### `ica_params` Section (used only when `feature_reduction_method: "ica"`)

| Parameter | Type | Default | Valid values | Description |
|-----------|------|---------|-------------|-------------|
| `ica_params.n_components` | int or `"auto"` | `"auto"` | Integer ≥ 1, or `"auto"` | Number of ICs. `"auto"` uses Parallel Analysis; threshold controlled by `stats_params.ci_level` |
| `ica_params.max_iter` | int | `1000` | Integer ≥ 100 | FastICA maximum iterations for convergence |
| `ica_params.random_state` | int | `42` | Any integer | Random seed for FastICA initialization |

**ICA technical notes:**
- Whitening uses `whiten='unit-variance'` (scikit-learn FastICA).
- The mixing matrix `A` (shape P × K) is stored as `mixing_unnorm_` without
  column-wise normalization.
- Back-projection uses activation patterns `A @ beta_IC` per Haufe et al. (2014,
  NeuroImage), rather than filter patterns `pinv(A) @ beta_IC`. Activation patterns
  are more interpretable and stable for neuroimaging feature attribution.
- Parallel Analysis determines K by comparing observed eigenvalues to the
  `ci_level × 100`th percentile of eigenvalues from random (Gaussian noise) matrices
  of the same shape, over 100 iterations.

### `model_params` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_params.l1_ratio_predict` | list of float | `[0.01, 0.05, ..., 0.99]` | L1 ratio search grid for `predict` mode |
| `model_params.l1_ratio_correlate` | list of float | `[0.001, ..., 0.2]` | L1 ratio search grid for `correlate` mode |
| `model_params.alpha_min_predict` | float | `0.001` | Minimum alpha (regularization strength) for `predict` mode |
| `model_params.alpha_max_predict` | float | `100.0` | Maximum alpha for `predict` mode |
| `model_params.n_alphas_predict` | int | `50` | Not directly used at runtime (alpha is sampled continuously from `loguniform`); present for documentation |
| `model_params.alpha_min_correlate` | float | `0.001` | Minimum alpha for `correlate` mode |
| `model_params.alpha_max_correlate` | float | `100.0` | Maximum alpha for `correlate` mode |
| `model_params.n_alphas_correlate` | int | `50` | See `n_alphas_predict` note |
| `model_params.covariate_penalty_weights` | list of float | `[1.0, 0.1, 0.01, 0.001]` | Covariate feature scaling factors searched in `RandomizedSearchCV`. A value of 1.0 = no adjustment; 0.001 = extreme covariate privilege (features scaled ×1000 relative to brain features). Ignored when `covariate_method: "none"` |
| `model_params.regression_metric` | str | `"neg_root_mean_squared_error"` | Inner CV scoring for regression (auto-used; placeholder for future expansion) |
| `model_params.classification_metric` | str | `"neg_log_loss"` | Inner CV scoring for classification (auto-used) |

**Model selection by analysis_type:**
- `regression`: `ElasticNet` (single outcome) or `MultiTaskElasticNet` (multi-column outcome)
- `classification`: `LogisticRegression(penalty='elasticnet', solver='saga')`

**Alpha parameterization for classification:**
Because `LogisticRegression` uses `C = 1/alpha`, the alpha search range is inverted:
`C_search = loguniform(1/alpha_max, 1/alpha_min)`.

### `cv_params` Section

| Parameter | Type | Default | Valid values | Description |
|-----------|------|---------|-------------|-------------|
| `cv_params.n_outer_folds` | int or `"loo"` | — | Integer ≥ 2, or `"loo"` | Outer CV folds for performance estimation |
| `cv_params.n_inner_folds` | int or `"loo"` | — | Integer ≥ 2, or `"loo"` | Inner CV folds for hyperparameter tuning |
| `cv_params.n_inner_repeats` | int | `1` | Integer ≥ 1 | Number of repeated inner CV rounds. Ignored when `n_inner_folds: "loo"` |
| `cv_params.n_random_search_iter` | int | **NO DEFAULT — required** | Integer ≥ 1 | Number of hyperparameter combinations sampled per `RandomizedSearchCV` call. Applied uniformly to all stages. Recommended: 20–50 for the default 2-parameter space |
| `cv_params.random_state` | int | — | Any integer | Global random seed for CV splitters and `RandomizedSearchCV` |

**Note on LOO:**
When `n_outer_folds: "loo"` or `n_inner_folds: "loo"`, `LeaveOneOut` is used.
For LOO inner CV with regression, the scoring metric switches from
`neg_root_mean_squared_error` to `neg_mean_squared_error` (RMSE is not meaningful
for single-sample folds).

**CV splitter types by analysis_type:**
- `classification` (non-LOO): `StratifiedKFold` / `RepeatedStratifiedKFold`
- `regression` (non-LOO): `KFold` / `RepeatedKFold`

### `stats_params` Section

| Parameter | Type | Default | Valid values | Description |
|-----------|------|---------|-------------|-------------|
| `stats_params.n_permutations` | int | `10000` | Integer ≥ 0 | Number of label permutations for model p-value. Set to `0` to skip |
| `stats_params.n_bootstraps` | int | `10000` | Integer ≥ 1 | Bootstrap iterations for feature importance CIs and selection frequency |
| `stats_params.n_block_permutations` | int | `500` | Integer ≥ 1 | Permutations per block in block permutation tests. Typical range: 100–1000 |
| `stats_params.ci_level` | float | `0.95` | (0.0, 1.0) | Confidence level for bootstrap CIs. Also used as threshold for Parallel Analysis eigenvalue percentile |
| `stats_params.save_distributions` | bool | `true` | `true`, `false` | If `true`, saves the full bootstrap coefficient array (`bootstrap_coef_distribution.npz`, compressed NumPy archive) and per-block permutation null scores (`block_perm_null_{label}.csv`). Set to `false` if storage is constrained (e.g., high-dimensional data with many bootstraps). |

### `block_permutation_tests` Section (optional)

Defines feature blocks to test for unique contribution. Each entry is a label-to-definition mapping:

```yaml
block_permutation_tests:
  brain_block: "brain"        # substring match: all columns containing "brain"
  roi_block:                  # explicit list of column names
    - "roi_vmPFC"
    - "roi_amygdala"
```

- **String value**: selects all brain feature columns whose name contains the string.
- **List value**: selects brain feature columns whose name is in the list.
- Omit this section entirely to skip block permutation testing.

Block permutation p-values use a one-sided test: proportion of null scores ≥ observed
score (Laplace-corrected: `(count + 1) / (n_perms + 1)`).

---

## 5. Pipeline Stages and Outputs

### Step 1: Data Loading and Preprocessing (`load_and_prep_data`)

- Reads CSV, drops rows with any missing values (listwise deletion).
- Computes and logs top-10 univariate Spearman correlations between brain features
  and outcome (sanity check).
- Logs N:P ratio diagnostic (warns if `P_brain > N/5`).
- Loads apriori map if `feature_reduction_method: "apriori"`.

**Output:** No files. Internal data structures only.

### Step 2: Feature Reduction (fold-local, inside CV)

Applied inside all CV loops via `TransformerMixin` classes. Three transformers:

- **`ClusterPCATransformer`**: HDBSCAN → per-cluster PCA (1 component). Singleton
  ("noise") features (HDBSCAN label −1) are passed through as raw features.
- **`AprioriTransformer`**: Externally-defined cluster map → per-cluster PCA. Cluster
  structure fixed; PCA is refit per fold on training data.
- **`ICATransformer`**: StandardScaler → FastICA. Parallel Analysis determines K
  automatically. Mixing matrix stored as `mixing_unnorm_` for back-projection.

For selection frequency and bootstrap (descriptive stages), reducers are fit on the
full dataset. A log message notes this explicitly.

### Step 4: Nested CV (`run_nested_cv`)

Outer CV: performance estimation (R² for regression, AUC for classification).
Inner CV: hyperparameter tuning via `RandomizedSearchCV`.
Fold-local reduction is applied per outer fold (fit on training data only).
Covariates are prepended to brain features; `CovariateScaler` applies
`penalty_weight` scaling before `StandardScaler`.

Per-fold reducer outputs are saved for transparency:
- `cluster_loadings_fold_{n}.csv` (cluster_pca or apriori)
- `ica_mixing_matrix_fold_{n}.csv` (ica)

After nested CV, `_compute_evaluation_metrics` computes a comprehensive evaluation
report (RMSE, MAE, R², Pearson r for regression; AUC-ROC, Log-Loss, Sensitivity,
Specificity, Balanced Accuracy for classification).

**Output:** `nested_cv_scores.csv`, `model_performance.csv`, optionally `confusion_matrix.csv` (multi-class only)

### Step 5: Permutation Test (`run_permutation_test`)

Label permutation (Y shuffled globally). Each iteration runs the full nested CV.
P-value: `(count(null ≥ observed) + 1) / (n_perms + 1)`.

**Output (local/aggregate):**
- `permutation_null_distribution_{metric}.csv`
- `permutation_result.csv` (aggregate mode only)

### Step 6: Selection Frequency (`run_selection_frequency`)

Repeated half-sample subsampling (50% without replacement, `n_bootstraps` iterations).
Per feature: proportion of iterations where the model coefficient is non-zero.
Descriptive only — no significance threshold applied.

**Output:** `report_selection_frequency.csv`

### Step 7: Bootstrap Importance (`run_bootstrap`)

Approach Y conditional bootstrap: each iteration fits a fresh reducer clone on
resampled brain features, fits the model in reduced space using hyperparameters
fixed from full-data tuning, and back-projects coefficients to the invariant
original brain feature space via `_backproject_coef_original_space`. Aggregation
(mean, CI, pd) is computed in the original feature space, ensuring comparability
across iterations with different reducer fits.

Bootstrap CIs use quantiles at `(alpha/2, 1-alpha/2)`. Feature significance: CI does
not cross zero (`is_significant`). FDR correction: BH-FDR at q = 0.05 applied
independently per output CSV.

**Approximation note on `raw_coef_mean` with reduction methods:** dividing the
back-projected standardized coefficient by the original-feature SD is not equivalent
to a standardized beta from direct regression on original features. `std_coef_mean`
and `pd` are the primary inferential quantities; `raw_coef_mean` provides an
approximate scale-conversion.

**Selection frequency note:** hyperparameters are fixed from full-N tuning, while
each subsample uses N/2 samples. Regularization is therefore weaker in each
subsample, which may inflate absolute selection frequencies. Relative ordering is
preserved and no significance threshold is applied (purely descriptive).

**Output (varies by `feature_reduction_method`):**
- `report_feature_importance.csv` (reduction: none)
- `report_cluster_importance.csv` + `report_individual_importance.csv` (apriori)
- `report_individual_importance.csv` (cluster_pca)
- `report_individual_importance.csv` + `ica_mixing_matrix.csv` (ica)
- `cluster_loadings.csv` (apriori, cluster_pca)
- `report_{cluster|individual}_plotting.csv`
- `bootstrap_coef_distribution.npz` (when `save_distributions: true`)

### Step 8: Block Permutation (`run_block_perms`)

For each block: permute only that block's columns by row, run full nested CV,
build null distribution. Uses `n_block_permutations` iterations per block.
Observed score is the nested CV score from Step 4.

**Output:** `report_block_permutation.csv`, optionally `block_perm_null_{label}.csv` per block (when `save_distributions: true`)

---

## 6. Output File Schemas

### `nested_cv_scores.csv`
| Column | Type | Description |
|--------|------|-------------|
| `score` | float | Nested CV performance (R² or AUC) |

### `model_performance.csv`
Comprehensive evaluation metrics computed post-hoc from concatenated outer-fold predictions. Schema varies by `analysis_type`:

**Regression (single-output):** columns `metric`, `value`, `cv_type`. Metrics: `RMSE`, `MAE`, `R2`, `Pearson_r`, `Pearson_p`.

**Regression (multi-task):** columns `task`, `metric`, `value`, `cv_type`. Task values: `task_0`, `task_1`, ..., `macro`. Same metrics as above; macro uses `uniform_average`.

**Classification (binary):** columns `metric`, `value`. Metrics: `Log_Loss`, `AUC_ROC`, `Balanced_Accuracy`, `Sensitivity`, `Specificity`.

**Classification (multi-class):** columns `scope`, `class`, `metric`, `value`. Per-class metrics: `AUC_ROC`, `Sensitivity`, `Specificity`. Macro metrics: `Log_Loss`, `AUC_ROC`, `Balanced_Accuracy`.

### `confusion_matrix.csv`
Produced only for multi-class classification. Square matrix of predicted vs. true class counts (no header/index labels). Shape K × K where K = number of classes.

### `bootstrap_coef_distribution.npz`
Produced only when `save_distributions: true`. Compressed NumPy archive with arrays:
- `coef_dist`: shape `(B, P)` for single-output or `(B, K, P)` for multi-output. B = bootstrap iterations, K = tasks/classes, P = original brain features.
- `feature_names`: string array of original brain feature names, length P.
- `task_labels`: string array of task/class labels (multi-output only).

### `block_perm_null_{label}.csv`
Produced once per block when `save_distributions: true`. Single column `null_score` containing the permutation null distribution for that block.

### `permutation_result.csv`
| Column | Type | Description |
|--------|------|-------------|
| `observed_score` | float | Nested CV score |
| `p_value` | float | Laplace-corrected one-sided p-value |
| `n_permutations` | int | Total permutations in null distribution |

### `report_selection_frequency.csv`
| Column | Type | Description |
|--------|------|-------------|
| `feature` | str | Feature/component name |
| `selection_probability` | float | Proportion of subsamples with non-zero coefficient [0, 1] |

### `report_feature_importance.csv` / `report_cluster_importance.csv`
| Column | Type | Description |
|--------|------|-------------|
| `feature` / `cluster_id` | str | Feature or cluster identifier |
| `std_coef_mean` | float | Mean standardized coefficient across bootstrap samples |
| `std_ci_low` | float | Lower bootstrap CI bound (standardized) |
| `std_ci_high` | float | Upper bootstrap CI bound (standardized) |
| `raw_coef_mean` | float | Mean coefficient in original units (std_coef / feature_std) |
| `raw_ci_low` | float | Lower CI in original units |
| `raw_ci_high` | float | Upper CI in original units |
| `pd` | float | Probability of direction: max(Pr(coef>0), Pr(coef<0)) |
| `is_significant` | bool | CI does not cross zero |
| `p_value` | float | Approximate p-value from pd: `clip(2*(1-pd), 0, 1)` |
| `is_significant_fdr` | bool | Survives BH-FDR correction at q=0.05 |

### `report_individual_importance.csv`
Same schema as above, with column `feature` (raw feature name) and
`source_cluster` (apriori/cluster_pca) or absent (ica, none).

### `report_block_permutation.csv`
| Column | Type | Description |
|--------|------|-------------|
| `block` | str | Block label from config |
| `observed_score` | float | Nested CV score (same as Step 4) |
| `p_value` | float | Block-specific permutation p-value |

---

## 7. Environment Requirements

- **OS:** Linux or macOS (SLURM orchestrator uses bash)
- **Python:** 3.10 (required; tested against 418/418 tests)
- **Conda:** Mamba or Conda; environment defined in `environment.yaml`
- **SLURM:** Required only for distributed permutation testing; local mode works without SLURM

---

## 8. Edge Cases and Known Limitations

| Condition | Behavior |
|-----------|----------|
| All rows missing | `ValueError` raised; pipeline exits |
| `covariate_method: "none"` | `covariate_cols` ignored; `covariate_penalty_weights` forced to `[1.0]` |
| `n_inner_repeats` with LOO inner CV | `n_inner_repeats` silently ignored; `LeaveOneOut` is used |
| P = 1 feature (after reduction) | `_parallel_analysis` returns 1 component; `_compute_distance_matrix` returns zero matrix |
| `min_cluster_size` larger than P | HDBSCAN assigns all features to noise (label −1); each treated as singleton |
| `MultiTaskElasticNet` with `sample_weight` | `sample_weight` silently ignored (not supported by estimator) |
| Bootstrap iteration failure (singular matrix etc.) | Iteration result is `None`; excluded from CI computation. A warning is logged if >5% of iterations fail |
| `pd < 0.5` (zero-inflated sparse coefficients) | `p_value` is clipped to 1.0; `is_significant` (CI-based) is unaffected and remains the primary criterion |
| Block definition matches 0 columns | Block is skipped with a warning log entry |
| Multiple blocks (shared seed) | All blocks use the same permutation seed sequence (`RandomState(42)`), introducing positive correlation between block null distributions. BH-FDR correction across blocks is valid under PRDS (Benjamini & Yekutieli, 2001) |
| Conditional bootstrap CI width | Bootstrap conditions on full-data hyperparameters (standard practice). CIs are slightly narrower than a full double-bootstrap but adequate for typical neuroimaging sample sizes and smooth hyperparameter surfaces |
| `p_value` with zero-inflated bootstrap distributions | For high-L1-sparsity features, many bootstrap samples produce exactly zero, causing `pd` to be near 0.5 and `p_value` to be near 1.0 after clipping. This is conservative and correct. The rare failure mode (pd ≈ 1.0 when most non-zero samples have the same sign) may be anti-conservative; `is_significant` (CI-based) and `is_significant_fdr` remain valid and are the primary inference criteria |
| `raw_coef_mean` with reduction methods | Dividing the back-projected standardized coefficient by original-feature SD is an approximation — not a proper standardized beta from direct regression on original features. `std_coef_mean` and `pd` are the primary inferential quantities |
| Selection frequency magnitude | Hyperparameters fixed from full-N tuning are weaker on N/2 subsamples, potentially inflating absolute selection frequencies. Relative ordering is preserved; no significance threshold is applied |
