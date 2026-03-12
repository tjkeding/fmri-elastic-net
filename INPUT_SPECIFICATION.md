# Input Specification: fmri-elastic-net

Complete specification of all configuration parameters, file formats, pipeline
internals, output schemas, environment requirements, and edge-case behaviors.
Optimized for reproducibility and automated consumption.

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

**`main`**: Runs nested CV (Step 4), selection frequency (Step 6), bootstrap importance
(Step 7), optionally local permutation test (Step 5), and block permutation (Step 8).
When `--skip_main_perm` is set, Step 5 is deferred to perm_worker jobs.

**`perm_worker`**: Runs a subset of permutation iterations determined by `job_id`
and `n_jobs`. Seeds are split via `np.array_split` over the global seed array.
Writes `perm_chunk_{job_id}.csv` to `output_dir`.

**`aggregate`**: Reads all `perm_chunk_*.csv` files and `nested_cv_scores.csv` from
`output_dir`, concatenates the null distribution, computes final Laplace-corrected
p-value, writes `permutation_result.csv` and `permutation_null_distribution_{metric}.csv`.

---

## 2. Data File (`paths.data_file`)

**Format:** CSV with a header row. One row per subject.

**Required columns** (names controlled by `data_cols.*`):
- Subject ID column (string or integer): specified by `data_cols.subject_id_col`
- Outcome column (numeric): specified by `data_cols.post_score_col`
- Optional covariate columns (numeric): specified by `data_cols.covariate_cols`
- Brain feature columns (numeric): identified by substring `data_cols.brain_feature_substr`

**Missing data**: Rows with any `NaN` or missing values are removed via listwise
deletion before any analysis. A warning is logged with the count of removed rows.
If all rows are removed, the pipeline raises a `ValueError` and exits.

**Classification outcome**: Integer or float class labels. Binary (2 classes) and
multi-class (3+ classes) are both fully supported. Multi-class uses one-vs-rest (OVR)
decomposition internally for per-class AUC, sensitivity, and specificity.
`LogisticRegression(penalty='elasticnet', solver='saga')` is used for all classification.

**Regression outcome**: Any continuous numeric column. Multi-column outcomes trigger
`MultiTaskElasticNet` automatically.

**Optional column** (advanced):
- `data_cols.sample_weight_col`: Column of non-negative sample weights. If present,
  weights are passed to the model's `fit` call where supported. `MultiTaskElasticNet`
  does not support `sample_weight`; weights are silently ignored for that estimator.

---

## 3. Apriori Clustering File (`paths.apriori_clustering_file`)

**Required only when** `feature_reduction_method: "apriori"`.

**Format:** 2-column CSV, no header (first row is ignored). Column 1: feature name
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
| `analysis_mode` | str | `"predict"`, `"correlate"` | Controls L1 ratio search space and regularization philosophy |
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
In `perm_worker` mode, `n_jobs` is set to `1` per-task to avoid nested parallelism.

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
| `clustering_params.min_cluster_size` | int or `"auto"` | `"auto"` | Integer ≥ 2, or `"auto"` | HDBSCAN minimum cluster size. `"auto"` computes `max(3, ceil(log2(P) * (100/N)^0.25))` where P = brain features, N = samples |
| `clustering_params.distance_metric` | str | — | `"pearson"`, `"spearman"` | Pairwise similarity metric for distance matrix |
| `clustering_params.sign_handling` | str | — | `"unsigned"`, `"signed"` | `"unsigned"`: `sqrt(2*(1-|sim|))`; `"signed"`: `sqrt(2*(1-sim))`. Has no effect for `dcor` (planned) |

Distance matrix formula:
- `unsigned`: `sqrt(2 * (1 - |sim|))` — treats positive and negative correlations equivalently
- `signed`: `sqrt(2 * (1 - sim))` — positively correlated features are closer

Both formulas are proper metric space mappings into [0, 2]. NaN similarity values
(from constant features) are treated as zero similarity (maximum distance).

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
  of the same shape, over 100 iterations. When P > N, the analysis operates in sample
  covariance eigenspace (N × N) for computational efficiency.

### `model_params` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_params.l1_min_predict` | float | `0.5` | Lower bound of uniform L1 ratio search for `predict` mode |
| `model_params.l1_max_predict` | float | `0.99` | Upper bound of uniform L1 ratio search for `predict` mode |
| `model_params.l1_min_correlate` | float | `0.001` | Lower bound of uniform L1 ratio search for `correlate` mode |
| `model_params.l1_max_correlate` | float | `0.2` | Upper bound of uniform L1 ratio search for `correlate` mode |
| `model_params.alpha_min_predict` | float | `0.001` | Minimum alpha (regularization strength) for `predict` mode |
| `model_params.alpha_max_predict` | float | `100.0` | Maximum alpha for `predict` mode |
| `model_params.alpha_min_correlate` | float | `0.001` | Minimum alpha for `correlate` mode |
| `model_params.alpha_max_correlate` | float | `100.0` | Maximum alpha for `correlate` mode |
| `model_params.covariate_penalty_weights` | list of float | `[1.0, 0.1, 0.01, 0.001]` | Covariate feature scaling factors searched in `RandomizedSearchCV`. A value of 1.0 = no adjustment; 0.001 = extreme covariate privilege (covariate columns scaled ×1000 relative to brain features before standardization). Ignored when `covariate_method: "none"` |
| `model_params.regression_metric` | str | `"neg_root_mean_squared_error"` | RESERVED — not currently active. Inner CV scoring is hardcoded to `neg_root_mean_squared_error` |
| `model_params.classification_metric` | str | `"neg_log_loss"` | RESERVED — not currently active. Inner CV scoring is hardcoded to `neg_log_loss` |

**Constraint on L1 bounds:** `0 < l1_min < l1_max <= 1.0`. A `ValueError` is raised
if this constraint is violated.

**Model selection by analysis_type:**
- `regression`: `ElasticNet` (single outcome) or `MultiTaskElasticNet` (multi-column outcome; triggered when `post_score_col` resolves to multiple columns)
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
| `cv_params.n_random_search_iter` | int | **NO DEFAULT — required** | Integer ≥ 1 | Number of hyperparameter combinations sampled per `RandomizedSearchCV` call. Applied uniformly to all stages (nested CV, permutation null, selection frequency, bootstrap, block permutation). Recommended: 20–50 for the default 2-parameter space |
| `cv_params.random_state` | int | — | Any integer | Global random seed for CV splitters and `RandomizedSearchCV` |

**LOO behavior:**
When `n_outer_folds: "loo"` or `n_inner_folds: "loo"`, `LeaveOneOut` is used.
For LOO inner CV with regression, the scoring metric switches from
`neg_root_mean_squared_error` to `neg_mean_squared_error` (RMSE is not meaningful
for single-sample folds). `n_inner_repeats` is silently ignored for LOO inner CV.

**CV splitter types by analysis_type:**
- `classification` (non-LOO): `StratifiedKFold` / `RepeatedStratifiedKFold`
- `regression` (non-LOO): `KFold` / `RepeatedKFold`

### `stats_params` Section

| Parameter | Type | Default | Valid values | Description |
|-----------|------|---------|-------------|-------------|
| `stats_params.n_permutations` | int | `10000` | Integer ≥ 0 | Number of label permutations for model p-value. Set to `0` to skip |
| `stats_params.n_bootstraps` | int | `10000` | Integer ≥ 1 | Bootstrap iterations for feature importance CIs and selection frequency |
| `stats_params.n_block_permutations` | int | `500` | Integer ≥ 1 | Permutations per block in block permutation tests. Typical range: 100–1000 |
| `stats_params.ci_level` | float | `0.95` | (0.0, 1.0) | Confidence level for bootstrap CIs. Also used as percentile threshold for Parallel Analysis eigenvalue comparison |
| `stats_params.save_distributions` | bool | `true` | `true`, `false` | If `true`, saves the full bootstrap coefficient array (`bootstrap_coef_distribution.npz`) and per-block permutation null scores (`block_perm_null_{label}.csv`). Set to `false` if storage is constrained |

### `block_permutation_tests` Section (optional)

Defines feature blocks to test for unique contribution. Each entry is a label-to-definition mapping:

```yaml
block_permutation_tests:
  brain_block: "brain"        # substring match: all columns containing "brain"
  roi_block:                  # explicit list of column names
    - "roi_vmPFC"
    - "roi_amygdala"
```

- **String value**: selects all brain feature columns whose name contains that string.
- **List value**: selects brain feature columns whose name appears in the list.
- Omit this section entirely to skip block permutation testing.
- A block definition that matches 0 columns is skipped with a warning log entry.

Block permutation p-values use a one-sided Laplace-corrected test:
`(count(null ≥ observed) + 1) / (n_perms + 1)`.

---

## 5. Pipeline Stages and Internal Logic

### Step 1: Data Loading and Preprocessing (`load_and_prep_data`)

- Reads CSV via `pd.read_csv`; raises `FileNotFoundError` if absent.
- Listwise deletion: drops rows with any `NaN`. Logs count of dropped rows.
- Isolates outcome (`Y`), subject IDs, sample weights (if specified), covariates, and brain features.
- When `covariate_method: "none"`, forces `covariate_penalty_weights = [1.0]` in config.
- Computes and logs top-10 absolute univariate Spearman correlations between brain features
  and outcome as a data sanity check (uses first column for multi-task outcome).
- Logs N:P diagnostic: warns if `P_brain > N/5`.
- Loads apriori map if `feature_reduction_method: "apriori"`; validates all brain feature
  columns are covered; raises `ValueError` for unmatched features.
- Stores `{N, P_brain, ceiling, np_ratio, apriori_map}` in `config['_runtime']`.
- Does NOT apply any feature reduction; all reduction is deferred to CV loops.

**Output:** No files. Returns `(X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map)`.

### Step 2: Feature Reduction Transformers (fold-local, inside CV)

All three transformers implement `sklearn.base.TransformerMixin` and are cloned and
refit independently per CV fold (or per bootstrap/subsample iteration).

**`ClusterPCATransformer`:**
1. Computes pairwise distance matrix via `_compute_distance_matrix` using the configured
   `distance_metric` and `sign_handling`.
2. Fits `hdbscan.HDBSCAN(metric='precomputed')` on the distance matrix.
3. Features assigned to label −1 (noise/singletons) are passed through as raw features.
4. For each cluster, fits `StandardScaler` + `PCA(n_components=1)` on training features.
5. Caches a `loading_matrix_` of shape `(n_reduced × P)` for fast back-projection.

**`AprioriTransformer`:**
1. Reads the externally-defined cluster map from `apriori_map` (loaded at Step 1).
2. Cluster structure is invariant across folds; only PCA is refit per fold.
3. For each cluster, fits `StandardScaler` + `PCA(n_components=1)` on training features.
4. Caches `loading_matrix_` identically to `ClusterPCATransformer`.

**`ICATransformer`:**
1. Fits `StandardScaler` on training features.
2. Determines K via `_parallel_analysis` when `n_components: "auto"` (Parallel Analysis,
   100 random matrices, threshold at `ci_level × 100`th percentile).
3. Fits `FastICA(whiten='unit-variance', n_components=K)`.
4. Stores unnormalized mixing matrix as `mixing_unnorm_` (shape P × K).
5. `transform` applies `scaler` then `ica_.transform` to produce IC activations.

**Descriptor-stage behavior:** For selection frequency and bootstrap (Steps 6–7),
reducers are fit on the full dataset. A log message explicitly flags this distinction
from the fold-local inferential pathway.

### Step 3: Model and Hyperparameter Distribution (`create_model_and_param_dist`)

Constructs an sklearn `Pipeline` with three steps:
1. `StandardScaler` (step name `scaler`) — standardizes all features to zero mean/unit variance
2. `CovariateScaler` (step name `cov_scaler`) — scales covariate columns by `1/penalty_weight`
   before they enter the elastic net. A `penalty_weight < 1.0` reduces regularization on covariates
   relative to brain features (covariate-privileged). `penalty_weight = 1.0` is a no-op.
3. Estimator (step name `model`):
   - `ElasticNet(max_iter=10000, selection='random')` — single-output regression
   - `MultiTaskElasticNet(max_iter=10000, selection='random')` — multi-output regression
   - `LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000)` — classification

Hyperparameter distribution for `RandomizedSearchCV`:
- `model__l1_ratio`: `uniform(l1_min, l1_max - l1_min)` — mode-specific bounds
- `model__alpha` or `model__C`: `loguniform(a_min, a_max)` (regression) or `loguniform(1/a_max, 1/a_min)` (classification)
- `cov_scaler__penalty_weight`: discrete list from `covariate_penalty_weights` (only when `covariate_method != "none"`)

Primary performance metric:
- Regression: `neg_root_mean_squared_error` (inner CV hyperparameter selection); R² (outer CV performance)
- Classification: `neg_log_loss` (inner CV); AUC-ROC (outer CV performance)

**Output:** No files. Returns `(pipeline, param_dist, scoring, metric)`.

### Step 4: Nested CV (`run_nested_cv`)

**Outer loop** (performance estimation):
- CV splitter: `StratifiedKFold` (classification) or `KFold` (regression); `LeaveOneOut` if `"loo"`
- Per fold: fit reducer on training brain features only; transform both splits
- Per-fold reducer outputs saved: `cluster_loadings_fold_{n}.csv` (cluster_pca/apriori),
  `ica_mixing_matrix_fold_{n}.csv` (ica)
- When `covariate_method: "incorporate"`: covariates prepended to reduced brain features
  (`covariate_indices = range(n_covariates)`)
- When `covariate_method: "pre_regress"`: outcome residualized on covariates fold-locally
  via `_local_residualize` (nuisance regression strictly within fold)

**Inner loop** (hyperparameter tuning):
- `RandomizedSearchCV` with `n_random_search_iter` iterations
- CV splitter: `RepeatedStratifiedKFold` / `RepeatedKFold` (when `n_inner_repeats > 1`)
  or `StratifiedKFold` / `KFold` (when `n_inner_repeats == 1`); `LeaveOneOut` if `"loo"`
- For LOO inner CV with regression: scoring switches to `neg_mean_squared_error`

**Evaluation metrics** (post-hoc, from `_compute_evaluation_metrics`):
- Computed from concatenated outer-fold predictions (not inner CV scores)
- Regression: RMSE, MAE, R², Pearson r (and p-value)
- Classification (binary): Log_Loss, AUC_ROC, Balanced_Accuracy, Sensitivity, Specificity
- Classification (multi-class): per-class AUC, Sensitivity, Specificity via OVR;
  macro Log_Loss, AUC_ROC, Balanced_Accuracy; confusion matrix (K × K)

**Output:** `nested_cv_scores.csv`, `model_performance.csv`, optionally `confusion_matrix.csv`

### Step 5: Permutation Test (`run_permutation_test`)

- Seeds: `np.random.RandomState(42).randint(0, 1e9, n_permutations)` — fully reproducible
- Each iteration: shuffle Y globally via `sklearn.utils.shuffle`, run full nested CV
  via `_run_cv_fold_loop` (same metric as observed score: R² or AUC from concatenated
  outer predictions — ensures null and observed are on the same scale)
- `ConvergenceWarning` suppressed per-task (expected for permuted data)
- P-value: `(count(null ≥ observed) + 1) / (n_perms + 1)` (Laplace-corrected, one-sided)
- In worker mode: writes `perm_chunk_{job_id}.csv`
- In local mode: writes null distribution and computes p-value directly

**Output:**
- `permutation_null_distribution_{metric}.csv`
- `permutation_result.csv` (aggregate mode only)

### Step 6: Selection Frequency (`run_selection_frequency`)

- Setup: calls `_fit_full_data_model` to fit reducer on full data and tune hyperparameters
- Per iteration (n_bootstraps total, `RandomState(43)` seed sequence):
  1. Subsample 50% of subjects without replacement
  2. Fit fresh reducer clone on subsampled brain features (Approach Y)
  3. Fit model with `best_params` (fixed hyperparameters — no re-tuning)
  4. Back-project `coef_ != 0` indicators to original feature space via
     `_backproject_coef_original_space`
- Aggregation: mean of binary non-zero indicators across iterations = selection probability
- Multi-output: per-task files written to `output_dir/task_{label}/`; union aggregate
  (selected in ≥ 1 task) written to top-level `output_dir`

**Output:** `report_selection_frequency.csv` (and per-task files for multi-output)

### Step 7: Bootstrap Importance (`run_bootstrap`)

**Approach Y conditional bootstrap** (Efron & Tibshirani, 1993, Ch. 13):
- Setup: calls `_fit_full_data_model`; saves `reducer_full`, `best_params`, `feat_std_map`
- Per iteration (`_boot_task`):
  1. Weight-aware bootstrap resample with replacement (uses `weights` probability vector
     if `sample_weight_col` is specified)
  2. Fit fresh reducer clone on resampled brain features
  3. Fit model with `best_params` (no re-tuning); catch `ConvergenceWarning` per iteration
  4. Back-project coefficients to original brain feature space via
     `_backproject_coef_original_space`
- Failed iterations (exception raised) return `None` and are excluded; a warning is
  logged if > 5% of iterations fail
- Convergence failures tracked and logged as a count

**Aggregation:**
- `std_coef_mean`: mean standardized coefficient across bootstrap samples
- `std_ci_low/high`: quantiles at `(alpha/2, 1-alpha/2)`
- `raw_coef_mean = std_coef_mean / feature_std` (approximate — see limitations)
- `pd`: `max(Pr(coef > 0), Pr(coef < 0))` across bootstrap samples
- `is_significant`: CI does not cross zero
- `p_value`: `clip(2 * (1 - pd), 0, 1)`
- `is_significant_fdr`: BH-FDR at q = 0.05 applied independently per output CSV

**Output files by `feature_reduction_method`:**
- `none`: `report_feature_importance.csv`
- `cluster_pca`: `report_individual_importance.csv`, `cluster_loadings.csv`
- `apriori`: `report_cluster_importance.csv`, `report_individual_importance.csv`, `cluster_loadings.csv`
- `ica`: `report_individual_importance.csv`, `ica_mixing_matrix.csv`
- All methods: `report_{cluster|individual}_plotting.csv` (subject-level visualization data for significant features)
- When `save_distributions: true`: `bootstrap_coef_distribution.npz`

### Step 8: Block Permutation (`run_block_perms`)

- For each block defined in `block_permutation_tests`:
  1. Identify columns matching block definition (substring or list)
  2. Skip block if 0 columns match (log warning)
  3. For each of `n_block_permutations` iterations:
     - Copy full X_brain; permute only the block columns (row shuffle, independent permutation)
     - Run full nested CV via `_run_cv_fold_loop`
  4. Null distribution: `n_block_permutations` scores from permuted data
  5. P-value: `(count(null ≥ observed) + 1) / (n_block_permutations + 1)`
  6. Observed score: nested CV score from Step 4 (stored in `config['_runtime']`)
- Seed sequence: `RandomState(42)` — same seed root across all blocks (see edge cases)

**Output:** `report_block_permutation.csv`; optionally `block_perm_null_{label}.csv` per block

---

## 6. Output File Schemas

### `nested_cv_scores.csv`
| Column | Type | Description |
|--------|------|-------------|
| `score` | float | Nested CV performance (R² for regression, AUC for classification) |

### `model_performance.csv`
Comprehensive evaluation metrics from concatenated outer-fold predictions. Schema varies:

**Regression (single-output):** columns `metric`, `value`, `cv_type`.
Metrics: `RMSE`, `MAE`, `R2`, `Pearson_r`, `Pearson_p`. `cv_type` = `"KFold"` or `"LOO"`.

**Regression (multi-task):** columns `task`, `metric`, `value`, `cv_type`.
Task values: `task_0`, `task_1`, ..., `macro`. Macro uses `uniform_average`. Same metrics;
macro row omits `Pearson_r` and `Pearson_p`.

**Classification (binary):** columns `metric`, `value`.
Metrics: `Log_Loss`, `AUC_ROC`, `Balanced_Accuracy`, `Sensitivity`, `Specificity`.

**Classification (multi-class):** columns `scope`, `class`, `metric`, `value`.
Per-class (`scope = "per_class"`): `AUC_ROC`, `Sensitivity`, `Specificity`.
Macro (`scope = "macro"`, `class = "all"`): `Log_Loss`, `AUC_ROC`, `Balanced_Accuracy`.

### `confusion_matrix.csv`
Produced only for multi-class classification. Shape K × K. Header row contains integer
column indices 0 through K−1. No row index.

### `permutation_null_distribution_{metric}.csv`
Single column `null_{metric}` containing the full null distribution. Written by local
permutation test or `aggregate` mode.

### `permutation_result.csv`
| Column | Type | Description |
|--------|------|-------------|
| `observed_score` | float | Nested CV score |
| `p_value` | float | Laplace-corrected one-sided p-value |
| `n_permutations` | int | Total permutations in null distribution |

### `report_selection_frequency.csv`
| Column | Type | Description |
|--------|------|-------------|
| `feature` | str | Feature/component name (original brain feature names when reduction is active) |
| `selection_probability` | float | Proportion of subsamples with non-zero coefficient [0, 1] |

### `report_feature_importance.csv` / `report_cluster_importance.csv`
| Column | Type | Description |
|--------|------|-------------|
| `feature` / `cluster_id` | str | Feature or cluster identifier |
| `std_coef_mean` | float | Mean standardized coefficient across bootstrap samples |
| `std_ci_low` | float | Lower bootstrap CI bound (standardized) |
| `std_ci_high` | float | Upper bootstrap CI bound (standardized) |
| `raw_coef_mean` | float | Mean coefficient in approximate original units (`std_coef_mean / feature_std`) |
| `raw_ci_low` | float | Lower CI in approximate original units |
| `raw_ci_high` | float | Upper CI in approximate original units |
| `pd` | float | Probability of direction: `max(Pr(coef>0), Pr(coef<0))` across bootstrap samples |
| `is_significant` | bool | CI does not cross zero — primary inference criterion |
| `p_value` | float | Approximate p-value from pd: `clip(2*(1-pd), 0, 1)` |
| `is_significant_fdr` | bool | Survives BH-FDR correction at q = 0.05 |

### `report_individual_importance.csv`
Same schema as `report_feature_importance.csv` with column `feature` (original feature name).
For `apriori` and `cluster_pca`, also includes `source_cluster` (cluster label to which the feature belongs).

### `report_block_permutation.csv`
| Column | Type | Description |
|--------|------|-------------|
| `block` | str | Block label from config |
| `observed_score` | float | Nested CV score from Step 4 |
| `p_value` | float | Block-specific permutation p-value |

### `bootstrap_coef_distribution.npz`
Produced only when `save_distributions: true`. Compressed NumPy archive with:
- `coef_dist`: shape `(B, P)` for single-output or `(B, K, P)` for multi-output.
  B = bootstrap iterations, K = tasks/classes, P = original brain features.
- `feature_names`: string array of original brain feature names, length P.
- `task_labels`: string array of task/class labels (multi-output only).

### `block_perm_null_{label}.csv`
Produced once per block when `save_distributions: true`. Single column `null_score`
containing the permutation null distribution for that block.

### `cluster_loadings.csv` / `cluster_loadings_fold_{n}.csv`
| Column | Type | Description |
|--------|------|-------------|
| `cluster` | str | Cluster label (e.g., `Cluster_0`, `Noise`) |
| `feature` | str | Original feature name |
| `loading` | float | PCA component 1 loading coefficient |

### `ica_mixing_matrix.csv` / `ica_mixing_matrix_fold_{n}.csv`
DataFrame with rows = original brain features (index), columns = IC labels
(`IC_1`, `IC_2`, ..., `IC_K`). Values are unnormalized mixing matrix entries.

### `report_{cluster|individual}_plotting.csv`
Subject-level partial association data for significant features. Contains subject IDs,
partial residuals of the feature (after regressing out all other features), and outcome.
Only written if at least one significant feature exists. Produced by
`calculate_visualization_data`.

---

## 7. Multi-Output Behavior

Multi-task regression is triggered when `post_score_col` resolves to multiple columns
(DataFrame Y with shape N × K). Multi-class classification is triggered when the
outcome column contains more than 2 unique class labels.

**Output organization:**
- Per-task/per-class files are written to `output_dir/task_{label}/` subdirectories
- Aggregate/union files are written to top-level `output_dir`
- `model_performance.csv` uses `task`, `scope`, or `class` columns to distinguish
  per-task and macro-averaged metrics

**Bootstrap coefficient shape:**
- Single-output regression: `(B, P)`
- Multi-task regression: `(B, K, P)`
- Binary classification: squeezed to `(B, P)` (`coef_` shape (1, P) squeezed to (P,))
- Multi-class classification: `(B, K, P)` where K = number of classes

---

## 8. Environment Requirements

- **OS:** Linux or macOS (SLURM orchestrator uses bash)
- **Python:** 3.10 (required)
- **Conda:** Mamba or Conda; environment defined in `environment.yaml`
- **SLURM:** Required only for distributed permutation testing; local mode works without SLURM

All package versions are pinned in `environment.yaml`.

---

## 9. SLURM Orchestration

`run_fmri-elastic-net.sh` accepts 5 positional arguments:
```
sh run_fmri-elastic-net.sh <CONFIG_PATH> <LOG_DIR> <MEM_GB> <CPUS_PER_TASK> <N_JOBS>
```

- `N_JOBS`: number of permutation worker array jobs. Must evenly divide
  `stats_params.n_permutations` for clean chunk distribution.
- All three jobs use the same `CPUS_PER_TASK` and `MEM_PER_NODE` resources.
- The aggregation job waits for both EN_Main and EN_Worker to complete via
  `--dependency=afterok:${JOB_ID_MAIN}:${JOB_ID_WORKERS}`.
- `PARTITION` must be set to the site-specific SLURM partition name.
- `ENV_SETUP` may need to be prepended with HPC-specific module load commands.

---

## 10. Edge Cases and Known Limitations

| Condition | Behavior |
|-----------|----------|
| All rows missing | `ValueError` raised; pipeline exits |
| `covariate_method: "none"` | `covariate_cols` ignored; `covariate_penalty_weights` forced to `[1.0]` |
| `n_inner_repeats` with LOO inner CV | `n_inner_repeats` silently ignored; `LeaveOneOut` used |
| P = 1 feature (after reduction) | `_parallel_analysis` returns 1 component; `_compute_distance_matrix` returns zero matrix |
| `min_cluster_size` larger than P | HDBSCAN assigns all features to noise (label −1); each treated as singleton |
| `MultiTaskElasticNet` with `sample_weight` | `sample_weight` silently ignored (not supported by estimator) |
| Bootstrap iteration failure | Iteration result is `None`; excluded from CI computation. Warning logged if > 5% of iterations fail |
| `pd < 0.5` (zero-inflated sparse coefficients) | `p_value` is clipped to 1.0; `is_significant` (CI-based) is unaffected and remains the primary criterion |
| `pd ≈ 1.0` (anti-conservative failure mode) | May occur when nearly all non-zero bootstrap samples share the same sign. Rare in practice; `is_significant` and `is_significant_fdr` remain valid |
| Block definition matches 0 columns | Block is skipped with a warning log entry |
| Multiple blocks (shared seed root) | All blocks use the same seed sequence root (`RandomState(42)`), introducing positive correlation between block null distributions. BH-FDR correction across blocks is valid under PRDS (Benjamini & Yekutieli, 2001) |
| Conditional bootstrap CI width | Bootstrap conditions on full-data hyperparameters. CIs are slightly narrower than a full double-bootstrap but adequate for typical neuroimaging sample sizes |
| `raw_coef_mean` with reduction methods | Dividing the back-projected standardized coefficient by original-feature SD is an approximation — not a standardized beta from direct regression on original features. `std_coef_mean` and `pd` are the primary inferential quantities |
| Selection frequency magnitude | Hyperparameters fixed from full-N tuning are weaker on N/2 subsamples, potentially inflating absolute selection frequencies. Relative ordering is preserved; no significance threshold is applied |
| LOO with classification | `StratifiedKFold` is used for non-LOO; `LeaveOneOut` cannot stratify. With very small N, class proportions per fold may be unbalanced |
| Spearman distance for P = 2 | `scipy.stats.spearmanr` returns a scalar correlation for two columns; the code explicitly converts this to a 2×2 matrix |
| NaN similarity values | Constant features produce undefined correlation; treated as zero similarity (maximum distance = `sqrt(2)`) before HDBSCAN |
