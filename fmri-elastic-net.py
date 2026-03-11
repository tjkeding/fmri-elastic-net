"""
Elastic Net Regression and Binary/Multi-Class Classification Pipeline

FEATURES:
- Flexible data loading (covariates + brain features) with listwise deletion and N:P diagnostics.
- Feature Reduction: Raw Features, HDBSCAN Clustering + PCA, Apriori Clustering, ICA.
  All reduction methods are applied fold-locally inside the CV loop to prevent data leakage.
- Approach Y Bootstrap: each iteration fits a fresh reducer clone on resampled brain features,
  fits the model in reduced space with fixed hyperparameters, and back-projects coefficients
  to the invariant original feature space for meaningful aggregation across iterations.
- Covariate Handling: Incorporate, Pre-Regress (Cross-Val), or None.
- Two modes for Analysis: Predict (Lasso) vs. Correlate (Ridge/Mapping).
- Validation: Supports Repeated K-Fold and LOO for small datasets.
- Statistical Inference: Bootstrap CIs + Probability of Direction (pd) + BH-FDR correction.
- Comprehensive Evaluation Metrics: RMSE/MAE/R²/Pearson r (regression); AUC-ROC/Log-Loss/
  Sensitivity/Specificity/Balanced Accuracy (classification). Written to model_performance.csv.
- Outputs: Standardized AND approximate Raw Coefficients; distribution archives when save_distributions=true.
- Multi-task regression foundational support (MultiTaskElasticNet; per-task reporting).
- Full Compatibility: Classification (AUC/LogLoss) & Regression (RMSE/R2).

STATISTICAL NOTES:
- The pd-to-p-value conversion (p = 2*(1 - pd)) assumes a continuous coefficient
  distribution. For sparse features with L1 regularization, bootstrap distributions can
  be zero-inflated: many iterations produce exactly zero, causing pd < 0.5 and p > 1.0
  before clipping. In such cases p is clipped to 1.0. The CI-based is_significant flag
  is the primary inference criterion and is unaffected by zero-inflation. The p_value
  and is_significant_fdr columns are complementary and should be interpreted alongside
  is_significant (Makowski et al., 2019).
- Bootstrap importance (Approach Y): each iteration fits a fresh reducer clone on resampled
  brain features, fits the model in reduced space using fixed hyperparameters (conditional
  bootstrap, Efron & Tibshirani, 1993, Ch. 13), and back-projects coefficients to the
  invariant original brain feature space via _backproject_coef_original_space. CIs may
  be marginally narrower than a full double-bootstrap that re-tunes hyperparameters per
  iteration.
- raw_coef_mean with reduction methods (cluster_pca, apriori, ica) is approximate: the
  back-projected standardized coefficient is divided by original-feature SD, which is not
  equivalent to a standardized beta from direct regression on original features. The
  std_coef_mean and pd columns are the primary inferential quantities.
- ICA feature back-projection uses the activation pattern (A @ beta_IC, Haufe et al.,
  2014, NeuroImage) rather than the filter pattern (pinv(A) @ beta_IC). Activation
  patterns represent signal co-variation in feature space and are more interpretable and
  stable for neuroimaging feature attribution.
- Selection frequency hyperparameters are fixed from full-N tuning; each N/2 subsample
  sees weaker regularization, which may inflate absolute selection frequencies. Relative
  ordering is preserved; no significance threshold is applied (Meinshausen & Bühlmann, 2010).

Written by: Taylor J. Keding, Ph.D.
"""

import os
import sys
import glob
import yaml
import logging
import warnings
import argparse

# Data Handling
import pandas as pd
import numpy as np

# Parallel Processing
from joblib import Parallel, delayed

# Scientific / Clustering
import hdbscan
from scipy.special import expit
from scipy.stats import spearmanr, loguniform, uniform
from sklearn.decomposition import PCA, FastICA

# Scikit-Learn
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, StratifiedKFold,
    RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression, MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import (
    roc_auc_score, r2_score, mean_absolute_error, mean_squared_error,
    log_loss, balanced_accuracy_score, recall_score, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr

# Suppress ConvergenceWarning globally only for perm workers where it is expected;
# elsewhere we count and log occurrences (see run_bootstrap).
# UserWarning suppression removed — address specific warnings as they arise.


# --- Helpers ---
def _local_residualize(X_cov, Y, tr, te):
    """Perform nuisance regression strictly within fold to prevent statistical leakage."""
    if X_cov.empty:
        return Y.iloc[tr], Y.iloc[te]
    lr = LinearRegression()
    lr.fit(X_cov.iloc[tr], Y.iloc[tr])
    return Y.iloc[tr] - lr.predict(X_cov.iloc[tr]), Y.iloc[te] - lr.predict(X_cov.iloc[te])


def _bh_fdr(p_values, q=0.05):
    """
    Benjamini-Hochberg FDR correction. Returns a boolean array of rejected hypotheses.
    Implemented without scipy.stats.false_discovery_control for compatibility.
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.zeros(n, dtype=bool)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    threshold = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= threshold
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.where(below)[0][-1]
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx[:max_idx + 1]] = True
    return rejected


def _get_task_labels(Y, config):
    """Return a list of string labels for each task/class in Y.

    For multi-task regression, returns column names. For classification, returns
    sorted unique class labels. For single-output regression, returns ['single'].
    """
    if config['analysis_type'] == 'regression':
        if hasattr(Y, 'columns') and Y.ndim > 1:
            return [str(c) for c in Y.columns]
        return ['single']
    else:
        return [str(c) for c in sorted(np.unique(Y))]


# --- Custom Transformer ---
class CovariateScaler(BaseEstimator, TransformerMixin):
    """Scale covariate columns by 1/penalty_weight before entering the elastic net.

    A penalty_weight < 1.0 effectively reduces the regularization applied to
    covariate features relative to brain features, protecting important covariates
    from shrinkage. A penalty_weight of 1.0 is a no-op.

    Parameters
    ----------
    covariate_indices : list of int or None
        Column indices of covariate features. If None, transform is a no-op.
    penalty_weight : float
        Must be > 0. Scale factor applied as 1/penalty_weight to covariate columns.
        Searched as a hyperparameter in RandomizedSearchCV when covariate_method != 'none'.
    """
    def __init__(self, covariate_indices=None, penalty_weight=1.0):
        self.covariate_indices = covariate_indices
        self.penalty_weight = penalty_weight

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X):
        if self.covariate_indices is None or self.penalty_weight == 1.0:
            return X
        if self.penalty_weight <= 0:
            raise ValueError(
                f"CovariateScaler: penalty_weight must be positive, got {self.penalty_weight}"
            )
        X_transformed = X.copy()
        scale_factor = 1.0 / self.penalty_weight
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed.iloc[:, self.covariate_indices] *= scale_factor
        else:
            X_transformed[:, self.covariate_indices] *= scale_factor
        return X_transformed


# --- Feature Reduction Transformers (applied fold-locally inside CV) ---

class _ClusterPCABase(BaseEstimator, TransformerMixin):
    """Shared base for ClusterPCATransformer and AprioriTransformer.

    Provides common transform, get_feature_names_out, get_loadings, and
    _fit_cluster_pcas logic. Subclasses implement fit() to determine cluster
    assignments, then call _fit_cluster_pcas(X_df, cluster_ids_by_feature).
    """

    def _cluster_label(self, cid):
        """Return display label for a cluster ID. Override for noise handling."""
        return f"Cluster_{cid}"

    def _fit_cluster_pcas(self, X_df, cluster_map):
        """Fit per-cluster PCA from a cluster assignment map. Shared by subclasses."""
        self.pcas_ = {}
        for cid in np.unique(cluster_map.values):
            feats = cluster_map[cluster_map == cid].index.tolist()
            feats = [f for f in feats if f in X_df.columns]
            if not feats:
                continue
            if len(feats) > 1:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_df[feats])
                pca = PCA(n_components=1, random_state=42)
                pca.fit(X_scaled)
                self.pcas_[cid] = (feats, scaler, pca)
            else:
                self.pcas_[cid] = (feats, None, None)
        self._cache_loading_matrix()

    def _cache_loading_matrix(self):
        """Build and cache loading_matrix_ (n_reduced x P) for fast back-projection."""
        reduced_names = self.get_feature_names_out()
        P = len(self.feature_names_in_)
        n_red = len(reduced_names)
        feat_idx = {f: i for i, f in enumerate(self.feature_names_in_)}
        mat = np.zeros((n_red, P))
        loadings = self.get_loadings()
        loadings_by_cluster = {}
        for row in loadings:
            loadings_by_cluster.setdefault(row['cluster'], []).append(row)
        for i, r_name in enumerate(reduced_names):
            if r_name in loadings_by_cluster:
                for lr in loadings_by_cluster[r_name]:
                    if lr['feature'] in feat_idx:
                        mat[i, feat_idx[lr['feature']]] = lr['loading']
            elif r_name in feat_idx:
                mat[i, feat_idx[r_name]] = 1.0
        self.loading_matrix_ = mat

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'pcas_')
        if not self.pcas_:
            raise AttributeError("Transformer has not been fitted yet (pcas_ is empty).")
        X_df = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X
        result = {}
        for cid, (feats, scaler, pca) in self.pcas_.items():
            c_name = self._cluster_label(cid)
            if pca is not None:
                result[c_name] = pca.transform(scaler.transform(X_df[feats]))[:, 0]
            else:
                result[feats[0]] = X_df[feats[0]].values
        return pd.DataFrame(result, index=X_df.index if hasattr(X_df, 'index') else None)

    def get_feature_names_out(self):
        result = []
        for cid, (feats, _, pca) in self.pcas_.items():
            result.append(self._cluster_label(cid) if pca is not None else feats[0])
        return result

    def get_loadings(self):
        """Returns a list of dicts: {cluster, feature, loading} for CSV export."""
        rows = []
        for cid, (feats, _, pca) in self.pcas_.items():
            c_name = self._cluster_label(cid)
            if pca is not None:
                for f, l in zip(feats, pca.components_[0], strict=True):
                    rows.append({'cluster': c_name, 'feature': f, 'loading': l})
            else:
                rows.append({'cluster': c_name, 'feature': feats[0], 'loading': 1.0})
        return rows


class ClusterPCATransformer(_ClusterPCABase):
    """
    Fit HDBSCAN clustering + per-cluster PCA on training data only.
    Applied inside the CV loop to prevent leakage of test-fold information.
    """
    def __init__(self, config):
        self.config = config
        self.cluster_map_ = None
        self.pcas_ = {}
        self.feature_names_in_ = None

    def _cluster_label(self, cid):
        return "Noise" if cid == -1 else f"Cluster_{cid}"

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if isinstance(X_df.columns[0], int):
            X_df.columns = self.feature_names_in_ if self.feature_names_in_ else list(range(X_df.shape[1]))
        self.feature_names_in_ = list(X_df.columns)
        dist_mat = _compute_distance_matrix(
            X_df.values,
            self.config['clustering_params']['distance_metric'],
            self.config['clustering_params']['sign_handling']
        )
        min_size = self.config['clustering_params']['min_cluster_size']
        if min_size == 'auto':
            min_size = max(3, int(np.ceil(np.log2(X_df.shape[1]) * (100 / X_df.shape[0]) ** 0.25)))
        labels = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='precomputed').fit_predict(dist_mat)
        self.cluster_map_ = pd.Series(labels, index=self.feature_names_in_, name='cluster_id')
        self._fit_cluster_pcas(X_df, self.cluster_map_)
        return self


class AprioriTransformer(_ClusterPCABase):
    """
    Fit PCA within externally-defined clusters (apriori map).
    Cluster structure is fixed externally; PCA is refit per fold on training data only.
    Network-level inference is valid because the cluster structure is invariant across folds.
    """
    def __init__(self, apriori_map, config):
        self.apriori_map = apriori_map
        self.config = config
        self.pcas_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = list(X_df.columns)
        cluster_map = pd.Series(
            {f: cid for f, cid in self.apriori_map.items() if f in X_df.columns},
            name='cluster_id'
        )
        self._fit_cluster_pcas(X_df, cluster_map)
        return self


class ICATransformer(BaseEstimator, TransformerMixin):
    """
    Fit FastICA (Option A: whiten='unit-variance') directly on standardized features.
    No intermediate PCA step; mixing_unnorm_ = ica_.mixing_ maps IC space to
    standardized feature space (shape P x K) for back-projection.
    Applied inside the CV loop to prevent leakage.
    """
    def __init__(self, config):
        self.config = config
        self.scaler_ = None
        self.ica_ = None
        self.mixing_unnorm_ = None  # ica_.mixing_: P x K in standardized feature space
        self.feature_names_in_ = None
        self.ic_names_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = list(X_df.columns)
        ica_cfg = self.config.get('ica_params', {})
        n_comp_cfg = ica_cfg.get('n_components', 'auto')
        N, P = X_df.shape
        if n_comp_cfg == 'auto':
            K = min(
                _parallel_analysis(
                    X_df.values,
                    self.config['stats_params']['ci_level'],
                    random_state=ica_cfg.get('random_state', 42)
                ),
                N, P
            )
        else:
            K = int(n_comp_cfg)
        self.scaler_ = StandardScaler().fit(X_df)
        X_std = self.scaler_.transform(X_df)
        # Option A: FastICA with unit-variance whitening acts directly on standardized
        # features. mixing_ is P x K and maps IC activations back to feature space.
        self.ica_ = FastICA(
            n_components=K,
            max_iter=ica_cfg.get('max_iter', 1000),
            random_state=ica_cfg.get('random_state', 42),
            whiten='unit-variance'
        )
        self.ica_.fit(X_std)
        # mixing_unnorm_: P x K, no column-wise normalization (F5 fix retained)
        self.mixing_unnorm_ = self.ica_.mixing_
        self.ic_names_ = [f"IC_{i + 1}" for i in range(K)]
        return self

    def transform(self, X):
        # feature_names_in_ set at fit time; DataFrame wrapping only when X arrives as ndarray
        X_df = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X
        X_std = self.scaler_.transform(X_df)
        X_ica = self.ica_.transform(X_std)
        return pd.DataFrame(X_ica, columns=self.ic_names_,
                            index=X_df.index if hasattr(X_df, 'index') else None)

    def get_feature_names_out(self):
        return self.ic_names_


# --- Logging ---
def setup_logging(output_dir, job_id=None):
    """Configure root logger to write to both a file and stdout.

    Parameters
    ----------
    output_dir : str
        Directory where pipeline.log will be written.
    job_id : int or None
        If provided and > 0, prefixes all log messages with [Job {job_id}]
        for disambiguation in SLURM array environments.
    """
    log_file = os.path.join(output_dir, 'pipeline.log')
    log_fmt = '%(asctime)s [%(levelname)-8s] %(message)s'
    if job_id is not None and job_id > 0:
        log_fmt = f'[Job {job_id}] ' + log_fmt
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )


# --- Step 1: Load & Prep ---
def load_and_prep_data(config, output_dir):
    """Load data from CSV, apply listwise deletion, and prepare X_brain, X_cov, and Y.

    Performs all pre-pipeline data operations: file reading, missing-data handling,
    covariate isolation, brain feature subsetting, apriori map loading, and N:P
    diagnostic logging. Reduction is NOT applied here; all dimensionality reduction
    is deferred to the CV loop (run_nested_cv) and descriptive stages.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary (loaded from YAML).
    output_dir : str
        Output directory path (used for logging setup context only).

    Returns
    -------
    X_brain : pd.DataFrame
        Brain feature columns (shape N x P_brain).
    X_cov : pd.DataFrame
        Covariate columns (shape N x P_cov). Empty DataFrame when covariate_method='none'.
    Y : pd.Series or pd.DataFrame
        Outcome variable(s). Series for single-output, DataFrame for multi-task regression.
    weights : pd.Series or None
        Sample weights if sample_weight_col is specified; None otherwise.
    subj_ids : pd.Series
        Subject identifier column.
    active_covs : list of str
        Active covariate column names. Empty list when covariate_method='none'.
    apriori_map : pd.Series or None
        Feature-to-cluster mapping for apriori reduction; None for all other methods.

    Raises
    ------
    FileNotFoundError
        If paths.data_file does not exist.
    ValueError
        If all rows are dropped after listwise deletion, required covariates are missing,
        or the apriori clustering file is missing/invalid/incomplete.
    """
    logging.info("--- Loading and Preprocessing Data ---")
    data_cfg = config['data_cols']
    try:
        df = pd.read_csv(config['paths']['data_file'])
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Data file not found: {config['paths']['data_file']}") from err

    # Missing data handling (F10): listwise deletion with informative logging
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logging.warning(
            f"Listwise deletion: {n_dropped} rows removed due to missing data "
            f"({n_before} -> {len(df)} rows)."
        )
    if len(df) == 0:
        raise ValueError("CRITICAL: All rows dropped after listwise deletion. Check input data.")

    post_col = data_cfg['post_score_col']
    Y = df[post_col]
    subj_ids = df[data_cfg['subject_id_col']]
    weights_col = data_cfg.get("sample_weight_col")
    weights = df[weights_col] if weights_col else None

    # Covariates
    cov_method = config['covariate_method']
    covariate_cols = data_cfg.get('covariate_cols', [])
    if cov_method == 'none':
        logging.info("Covariate Method: 'none' (Bypass)")
        X_cov = pd.DataFrame(index=Y.index)
        active_covs = []
        config['model_params']['covariate_penalty_weights'] = [1.0]
    else:
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing covariates: {missing}")
        X_cov = df[covariate_cols]
        active_covs = list(X_cov.columns)

    # Brain Features
    brain_substr = data_cfg['brain_feature_substr']
    drop_cols = [data_cfg['subject_id_col']] + (post_col if isinstance(post_col, list) else [post_col])
    all_cols = df.columns.drop(drop_cols, errors='ignore')
    brain_cols = [c for c in all_cols if brain_substr in c]
    X_brain = df[brain_cols]
    logging.info(
        f"Initial Load: {len(Y)} samples. {len(active_covs)} Covariates, {len(brain_cols)} Brain Features."
    )

    # Sanity Check
    corrs = X_brain.corrwith(Y if Y.ndim == 1 else Y.iloc[:, 0])
    top_10 = corrs.abs().sort_values(ascending=False).head(10)
    logging.info(f"--- Data Sanity Check: Top 10 Univariate Correlations ---\n{top_10}")

    # Load apriori map if needed (reduction itself applied inside CV)
    red_method = config['feature_reduction_method']
    apriori_map = None
    if red_method == 'apriori':
        map_path = config['paths'].get('apriori_clustering_file')
        if not map_path or not os.path.exists(map_path):
            raise ValueError("CRITICAL: 'apriori' selected but file invalid.")
        try:
            apriori_df = pd.read_csv(map_path, header=None)
            apriori_map = pd.Series(
                apriori_df.iloc[:, 1].values,
                index=apriori_df.iloc[:, 0].values,
                name='cluster_id'
            )
        except Exception as e:
            raise ValueError(f"Error reading apriori file: {e}") from e
        missing_feats = [f for f in X_brain.columns if f not in apriori_map.index]
        if missing_feats:
            raise ValueError(f"CRITICAL: Missing definitions for {missing_feats[0]}...")

    # Store runtime metadata
    N = len(Y)
    P_brain = len(X_brain.columns)
    ceiling = N // 5
    np_ratio = N / P_brain if P_brain > 0 else float('inf')
    config.setdefault('_runtime', {}).update(
        {'N': N, 'P_brain': P_brain, 'ceiling': ceiling, 'np_ratio': np_ratio,
         'apriori_map': apriori_map}
    )
    if P_brain > ceiling:
        logging.warning(
            f"N:P DIAGNOSTIC: P_brain ({P_brain}) exceeds N/5 ceiling ({ceiling}). N:P={np_ratio:.2f}"
        )
    else:
        logging.info(f"N:P Diagnostic: P_brain={P_brain}, ceiling={ceiling}, N:P={np_ratio:.2f} — acceptable.")

    # Return raw X_brain and X_cov separately; reduction applied inside CV loops
    return X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map


# --- Step 2: Feature Reduction Helpers (used by Transformers and distance-based functions) ---
def _parallel_analysis(X, ci_level, n_iterations=100, random_state=42):
    """Determine number of components using Parallel Analysis (Efficient O(N^3) approach)."""
    N, P = X.shape
    # Guard for P=1: single-feature parallel analysis is undefined
    if P == 1:
        return 1
    X_centered = X - np.mean(X, axis=0)
    if P > N:
        obs_eigenvalues = np.linalg.eigvalsh(np.dot(X_centered, X_centered.T) / (N - 1))[::-1]
    else:
        cov_mat = np.cov(X, rowvar=False)
        # np.cov on 1-column returns scalar; guard ensures P>=2 here
        obs_eigenvalues = np.linalg.eigvalsh(cov_mat)[::-1]

    rng = np.random.RandomState(random_state)
    sim_eigenvalues = []
    for _ in range(n_iterations):
        noise = rng.normal(0, 1, size=(N, P))
        noise_centered = noise - np.mean(noise, axis=0)
        if P > N:
            sim_ev = np.linalg.eigvalsh(np.dot(noise_centered, noise_centered.T) / (N - 1))[::-1]
        else:
            sim_ev = np.linalg.eigvalsh(np.cov(noise, rowvar=False))[::-1]
        sim_eigenvalues.append(sim_ev)

    threshold = np.percentile(sim_eigenvalues, ci_level * 100, axis=0)
    K = np.sum(obs_eigenvalues > threshold[:len(obs_eigenvalues)])
    return max(1, K)


def _compute_distance_matrix(X_values, distance_metric, sign_handling):
    """Compute distance matrix for HDBSCAN (Proper Metric Space mapping)."""
    P = X_values.shape[1]
    # Guard for P=1: distance matrix is trivially zero
    if P == 1:
        return np.zeros((1, 1))
    if distance_metric == 'spearman':
        corr_result = spearmanr(X_values, axis=0)
        sim_mat = corr_result.statistic if hasattr(corr_result, 'statistic') else corr_result.correlation
        # spearmanr returns scalar for P=2; convert to 2x2 matrix
        if np.ndim(sim_mat) == 0:
            sim_mat = np.array([[1.0, float(sim_mat)], [float(sim_mat), 1.0]])
    elif distance_metric == 'pearson':
        sim_mat = np.corrcoef(X_values, rowvar=False)
    else:
        raise ValueError(f"Distance metric {distance_metric} not implemented.")
    # Handle NaN from constant features: treat as zero similarity (maximum distance)
    sim_mat = np.nan_to_num(sim_mat, nan=0.0)
    if sign_handling == 'unsigned':
        dist_mat = np.sqrt(2 * (1 - np.abs(sim_mat)))
    else:
        dist_mat = np.sqrt(2 * (1 - sim_mat))
    return np.clip(dist_mat, 0, 2)


def _make_reducer(config, apriori_map=None):
    """
    Factory: return an unfitted Transformer for the configured reduction method, or None.
    """
    red_method = config['feature_reduction_method']
    if red_method == 'cluster_pca':
        return ClusterPCATransformer(config)
    elif red_method == 'apriori':
        return AprioriTransformer(apriori_map, config)
    elif red_method == 'ica':
        return ICATransformer(config)
    elif red_method == 'none':
        return None
    else:
        raise ValueError("Incorrect 'feature_reduction_method'. Use 'none', 'cluster_pca', 'apriori', or 'ica'.")


def _apply_reducer_fold(reducer_template, X_brain_tr, X_brain_te):
    """
    Fit a fresh reducer clone on training brain features, transform both folds.
    Returns (X_brain_tr_reduced, X_brain_te_reduced, fitted_reducer).
    """
    if reducer_template is None:
        return X_brain_tr, X_brain_te, None
    r = clone(reducer_template)
    r.feature_names_in_ = list(X_brain_tr.columns)
    r.fit(X_brain_tr)
    X_tr_red = r.transform(X_brain_tr)
    X_te_red = r.transform(X_brain_te)
    return X_tr_red, X_te_red, r


def _apply_reducer_full(reducer_template, X_brain):
    """
    Fit a reducer on the full brain feature set (for descriptive reporting only, not inference).
    Returns fitted reducer.
    """
    if reducer_template is None:
        return None
    r = clone(reducer_template)
    r.feature_names_in_ = list(X_brain.columns)
    r.fit(X_brain)
    return r


def _backproject_coef_original_space(coef_reduced, reducer, original_feature_names):
    """Back-project reduced-space coefficients to original feature space.

    Each bootstrap/selection-frequency iteration produces coefficients in the
    iteration's own reduced space. Back-projection maps these into the invariant
    original feature space so that aggregation (mean, CI, pd) is meaningful
    across iterations with different reducer fits.

    Parameters
    ----------
    coef_reduced : ndarray, shape (n_reduced,) or (K, n_reduced)
        Coefficients in the reduced feature space (from model.coef_).
    reducer : fitted TransformerMixin or None
        The reducer used in this iteration. If None, features are already
        in original space and coef_reduced is returned unchanged.
    original_feature_names : list of str, length P
        Column names of the original brain features.

    Returns
    -------
    coef_original : ndarray, shape (P,) or (K, P)
        Coefficients projected back into original feature space.

    Notes
    -----
    - ClusterPCATransformer / AprioriTransformer: uses PCA loadings from
      get_loadings(). Loading matrix has shape (n_clusters x P); result =
      coef_reduced @ loading_matrix (handles singleton clusters with loading=1.0).
    - ICATransformer: uses the unnormalized mixing matrix (A, shape P x K_ic).
      Result = coef_reduced @ mixing_unnorm_.T, following the activation pattern
      framework (Haufe et al., 2014, NeuroImage).
    - None: coef_reduced is returned unchanged.
    """
    if reducer is None:
        return coef_reduced

    P = len(original_feature_names)
    is_2d = coef_reduced.ndim == 2  # (K_tasks, n_reduced)

    if hasattr(reducer, 'get_loadings'):
        # Use cached loading matrix if available (built during fit); otherwise reconstruct
        if hasattr(reducer, 'loading_matrix_'):
            loading_matrix = reducer.loading_matrix_
        else:
            loadings = pd.DataFrame(reducer.get_loadings())
            reduced_names_ordered = reducer.get_feature_names_out()
            n_red = len(reduced_names_ordered)
            loading_matrix = np.zeros((n_red, P))
            feat_idx = {f: i for i, f in enumerate(original_feature_names)}
            for i, c_name in enumerate(reduced_names_ordered):
                cluster_rows = loadings[loadings['cluster'] == c_name]
                if cluster_rows.empty:
                    if c_name in feat_idx:
                        loading_matrix[i, feat_idx[c_name]] = 1.0
                else:
                    for _, lr in cluster_rows.iterrows():
                        if lr['feature'] in feat_idx:
                            loading_matrix[i, feat_idx[lr['feature']]] = lr['loading']
        if is_2d:
            return coef_reduced @ loading_matrix   # (K, n_red) @ (n_red, P) -> (K, P)
        else:
            return coef_reduced @ loading_matrix   # (n_red,) @ (n_red, P) -> (P,) -- matrix mult

    elif hasattr(reducer, 'mixing_unnorm_'):
        # ICA activation pattern: A @ beta_IC, Haufe et al. (2014)
        # mixing_unnorm_: P x K_ic
        A = reducer.mixing_unnorm_  # shape (P, K_ic)
        if is_2d:
            return coef_reduced @ A.T  # (K_tasks, K_ic) @ (K_ic, P) -> (K_tasks, P)
        else:
            return coef_reduced @ A.T  # (K_ic,) @ (K_ic, P) -> (P,)

    else:
        # Unknown reducer type: return as-is (safe fallback)
        return coef_reduced


# --- Step 3: Model ---
SCORING_REGRESSION = 'neg_root_mean_squared_error'
SCORING_CLASSIFICATION = 'neg_log_loss'
SCORING_REGRESSION_LOO = 'neg_mean_squared_error'


def create_model_and_param_dist(config, all_feature_names, active_covariate_names, Y=None):
    """
    Build sklearn Pipeline and RandomizedSearchCV param_dist.
    Covariates are always prepended; cov_indices = range(n_covariates).
    """
    mode = config['analysis_mode']
    model_cfg = config['model_params']
    seed = config['cv_params']['random_state']
    # Covariates are prepended, so indices are always the first len(active_covariate_names) positions
    n_covs = len(active_covariate_names)
    cov_indices = list(range(n_covs)) if n_covs > 0 else []

    if mode == 'predict':
        l1_min = model_cfg.get('l1_min_predict', 0.5)
        l1_max = model_cfg.get('l1_max_predict', 0.99)
    else:
        l1_min = model_cfg.get('l1_min_correlate', 0.001)
        l1_max = model_cfg.get('l1_max_correlate', 0.2)
    if not (0 < l1_min < l1_max <= 1.0):
        raise ValueError(
            f"Invalid L1 ratio bounds for mode '{mode}': l1_min={l1_min}, l1_max={l1_max}. "
            f"Must satisfy 0 < l1_min < l1_max <= 1.0."
        )
    l1_dist = uniform(l1_min, l1_max - l1_min)
    if mode == 'correlate':
        a_min, a_max = model_cfg.get('alpha_min_correlate', 0.001), model_cfg.get('alpha_max_correlate', 100.0)
    else:
        a_min, a_max = model_cfg.get('alpha_min_predict', 0.001), model_cfg.get('alpha_max_predict', 100.0)
    alpha_dist = loguniform(a_min, a_max)

    if config['analysis_type'] == 'regression':
        is_mt = Y is not None and Y.ndim > 1 and Y.shape[1] > 1
        model = MultiTaskElasticNet(max_iter=10000, selection='random', random_state=seed) if is_mt \
            else ElasticNet(max_iter=10000, selection='random', random_state=seed)
        scoring, metric = SCORING_REGRESSION, 'r2'
        alpha_param = 'model__alpha'
        alpha_search = alpha_dist
    else:
        model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, random_state=seed)
        scoring, metric = SCORING_CLASSIFICATION, 'roc_auc'
        alpha_param = 'model__C'
        alpha_search = loguniform(1.0 / a_max, 1.0 / a_min)

    param_dist = {
        'model__l1_ratio': l1_dist,
        alpha_param: alpha_search,
    }

    # Wire covariate_penalty_weights from config (F9 fix): only when covariates are active
    if config['covariate_method'] != 'none' and cov_indices:
        cov_pw_cfg = model_cfg.get('covariate_penalty_weights', [1.0, 0.1, 0.01, 0.001])
        param_dist['cov_scaler__penalty_weight'] = cov_pw_cfg

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('cov_scaler', CovariateScaler(covariate_indices=cov_indices if cov_indices else None)),
        ('model', model)
    ])
    return pipeline, param_dist, scoring, metric


def get_inner_cv(config):
    """Return the inner CV splitter for hyperparameter tuning.

    Returns StratifiedKFold/RepeatedStratifiedKFold for classification,
    KFold/RepeatedKFold for regression, or LeaveOneOut if n_inner_folds is "loo".
    n_inner_repeats is ignored when n_inner_folds is "loo".
    """
    n_folds = config['cv_params']['n_inner_folds']
    seed = config['cv_params']['random_state']
    if isinstance(n_folds, str) and n_folds.lower() == 'loo':
        return LeaveOneOut()
    n_repeats = config['cv_params'].get('n_inner_repeats', 1)
    if n_repeats <= 1:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) \
            if config['analysis_type'] == 'classification' \
            else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed) \
        if config['analysis_type'] == 'classification' \
        else RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)


def get_outer_cv(config):
    """Return the outer CV splitter for performance estimation.

    Returns StratifiedKFold for classification, KFold for regression, or
    LeaveOneOut if n_outer_folds is "loo". Outer CV does not support repeated splits.
    """
    n_outer = config['cv_params']['n_outer_folds']
    seed = config['cv_params']['random_state']
    if isinstance(n_outer, str) and n_outer.lower() == 'loo':
        return LeaveOneOut()
    return StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed) \
        if config['analysis_type'] == 'classification' \
        else KFold(n_splits=n_outer, shuffle=True, random_state=seed)


def _adjust_scoring_for_loo(scoring, inner_cv):
    """Substitute neg_mean_squared_error for neg_root_mean_squared_error when inner CV is LOO.

    RMSE is undefined for single-sample test folds (LOO). neg_mean_squared_error
    is functionally equivalent for hyperparameter ranking in that setting.
    """
    if isinstance(inner_cv, LeaveOneOut) and scoring == SCORING_REGRESSION:
        return SCORING_REGRESSION_LOO
    return scoring


# --- Step 4: Nested CV ---

def _compute_evaluation_metrics(Y_true, Y_pred, Y_prob, config, out_dir, is_loo=False):
    """Compute comprehensive evaluation metrics from nested CV predictions.

    Supplements the primary selection metric (nested CV score) with a richer set
    of evaluation metrics aligned with neuroimaging publication standards (Shen et al.,
    2017; Finn et al., 2015). Selection metrics remain unchanged (proper scoring rules
    are used for hyperparameter selection). Evaluation metrics are computed post-hoc
    from outer CV predictions.

    Parameters
    ----------
    Y_true : ndarray, shape (N,) or (N, K)
        True outcome values across all outer CV folds.
    Y_pred : ndarray, shape (N,) or (N, K)
        Predicted values across all outer CV folds.
    Y_prob : ndarray, shape (N, n_classes) or None
        Predicted probabilities (classification only; None for regression).
    config : dict
        Pipeline configuration dictionary.
    out_dir : str
        Output directory for model_performance.csv (and confusion_matrix.csv).
    is_loo : bool
        Whether the outer CV used LeaveOneOut (affects metric labeling).
    """
    analysis_type = config['analysis_type']
    rows = []

    if analysis_type == 'regression':
        is_multitask = Y_true.ndim > 1 and Y_true.shape[1] > 1
        label = 'LOO' if is_loo else 'KFold'
        if is_multitask:
            K = Y_true.shape[1]
            for k in range(K):
                yt_k = Y_true[:, k]
                yp_k = Y_pred[:, k]
                rmse_k = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
                mae_k = float(mean_absolute_error(yt_k, yp_k))
                r2_k = float(r2_score(yt_k, yp_k))
                r_k, p_r_k = pearsonr(yt_k, yp_k)
                rows += [
                    {'task': f'task_{k}', 'metric': 'RMSE', 'value': rmse_k, 'cv_type': label},
                    {'task': f'task_{k}', 'metric': 'MAE', 'value': mae_k, 'cv_type': label},
                    {'task': f'task_{k}', 'metric': 'R2', 'value': r2_k, 'cv_type': label},
                    {'task': f'task_{k}', 'metric': 'Pearson_r', 'value': float(r_k), 'cv_type': label},
                    {'task': f'task_{k}', 'metric': 'Pearson_p', 'value': float(p_r_k), 'cv_type': label},
                ]
            rows += [
                {'task': 'macro', 'metric': 'RMSE', 'value': float(np.sqrt(mean_squared_error(Y_true, Y_pred, multioutput='uniform_average'))), 'cv_type': label},
                {'task': 'macro', 'metric': 'MAE', 'value': float(mean_absolute_error(Y_true, Y_pred, multioutput='uniform_average')), 'cv_type': label},
                {'task': 'macro', 'metric': 'R2', 'value': float(r2_score(Y_true, Y_pred, multioutput='uniform_average')), 'cv_type': label},
            ]
        else:
            rmse = float(np.sqrt(mean_squared_error(Y_true, Y_pred)))
            mae = float(mean_absolute_error(Y_true, Y_pred))
            r2 = float(r2_score(Y_true, Y_pred))
            r_val, p_r = pearsonr(Y_true, Y_pred)
            rows = [
                {'metric': 'RMSE', 'value': rmse, 'cv_type': label},
                {'metric': 'MAE', 'value': mae, 'cv_type': label},
                {'metric': 'R2', 'value': r2, 'cv_type': label},
                {'metric': 'Pearson_r', 'value': float(r_val), 'cv_type': label},
                {'metric': 'Pearson_p', 'value': float(p_r), 'cv_type': label},
            ]
    else:
        # Classification
        n_classes = Y_prob.shape[1] if Y_prob is not None else len(np.unique(Y_true))
        is_multiclass = n_classes > 2
        if is_multiclass:
            classes = sorted(np.unique(Y_true))
            Y_bin = label_binarize(Y_true, classes=classes)
            for i, cls in enumerate(classes):
                auc_i = float(roc_auc_score(Y_bin[:, i], Y_prob[:, i]))
                sens_i = float(recall_score(Y_true, Y_pred, labels=[cls], average='macro', zero_division=0))
                Y_bin_i = (np.array(Y_true) == cls).astype(int)
                Y_pred_i = (np.array(Y_pred) == cls).astype(int)
                spec_i = float(recall_score(Y_bin_i, Y_pred_i, pos_label=0, zero_division=0))
                rows += [
                    {'scope': 'per_class', 'class': str(cls), 'metric': 'AUC_ROC', 'value': auc_i},
                    {'scope': 'per_class', 'class': str(cls), 'metric': 'Sensitivity', 'value': sens_i},
                    {'scope': 'per_class', 'class': str(cls), 'metric': 'Specificity', 'value': spec_i},
                ]
            macro_logloss = float(log_loss(Y_true, Y_prob))
            macro_auc = float(roc_auc_score(Y_true, Y_prob, multi_class='ovr'))
            macro_bal_acc = float(balanced_accuracy_score(Y_true, Y_pred))
            rows += [
                {'scope': 'macro', 'class': 'all', 'metric': 'Log_Loss', 'value': macro_logloss},
                {'scope': 'macro', 'class': 'all', 'metric': 'AUC_ROC', 'value': macro_auc},
                {'scope': 'macro', 'class': 'all', 'metric': 'Balanced_Accuracy', 'value': macro_bal_acc},
            ]
            cm = confusion_matrix(Y_true, Y_pred)
            pd.DataFrame(cm).to_csv(os.path.join(out_dir, 'confusion_matrix.csv'), index=False)
        else:
            logloss = float(log_loss(Y_true, Y_prob)) if Y_prob is not None else float('nan')
            auc = float(roc_auc_score(Y_true, Y_prob[:, 1])) if Y_prob is not None else float('nan')
            bal_acc = float(balanced_accuracy_score(Y_true, Y_pred))
            sensitivity = float(recall_score(Y_true, Y_pred, pos_label=1, zero_division=0))
            specificity = float(recall_score(Y_true, Y_pred, pos_label=0, zero_division=0))
            rows = [
                {'metric': 'Log_Loss', 'value': logloss},
                {'metric': 'AUC_ROC', 'value': auc},
                {'metric': 'Balanced_Accuracy', 'value': bal_acc},
                {'metric': 'Sensitivity', 'value': sensitivity},
                {'metric': 'Specificity', 'value': specificity},
            ]

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'model_performance.csv'), index=False)
    logging.info(f"Evaluation metrics written to {os.path.join(out_dir, 'model_performance.csv')}")


def run_nested_cv(config, X_brain, Y, weights, X_cov, active_covs, apriori_map=None):
    """
    Nested CV with fold-local feature reduction (F1 fix: reduction inside CV loop).
    Returns the CV performance score.
    """
    logging.info("--- Nested CV ---")
    outer = get_outer_cv(config)
    inner = get_inner_cv(config)
    n_iter = config['cv_params']['n_random_search_iter']
    inner_scoring_adj = _adjust_scoring_for_loo(SCORING_REGRESSION if config['analysis_type'] == 'regression' else SCORING_CLASSIFICATION, inner)
    is_pre = config['covariate_method'] == 'pre_regress'
    reducer_template = _make_reducer(config, apriori_map)

    y_preds, y_probs, y_trues = [], [], []
    # Use a 1-D array for CV splitting: KFold ignores Y values for regression;
    # for multi-task regression use first column; for multi-class use Y directly.
    split_Y = Y.iloc[:, 0] if hasattr(Y, 'ndim') and Y.ndim > 1 else Y
    for fold_idx, (tr, te) in enumerate(outer.split(X_brain, split_Y)):
        # Fold-local reduction: fit on training brain features only
        X_brain_tr = X_brain.iloc[tr].reset_index(drop=True)
        X_brain_te = X_brain.iloc[te].reset_index(drop=True)
        X_brain_tr_red, X_brain_te_red, fold_reducer = _apply_reducer_fold(reducer_template, X_brain_tr, X_brain_te)

        # Save per-fold reducer outputs for transparency (Approach Y, T2 decision)
        if fold_reducer is not None:
            out_dir = config['paths']['output_dir']
            if hasattr(fold_reducer, 'get_loadings'):
                pd.DataFrame(fold_reducer.get_loadings()).to_csv(
                    os.path.join(out_dir, f'cluster_loadings_fold_{fold_idx}.csv'), index=False
                )
            elif hasattr(fold_reducer, 'mixing_unnorm_'):
                pd.DataFrame(
                    fold_reducer.mixing_unnorm_,
                    index=fold_reducer.feature_names_in_,
                    columns=fold_reducer.ic_names_
                ).to_csv(
                    os.path.join(out_dir, f'ica_mixing_matrix_fold_{fold_idx}.csv')
                )

        # Recombine: covariates prepended (cov_indices = range(n_covs))
        if not X_cov.empty and config['covariate_method'] != 'pre_regress':
            X_cov_tr = X_cov.iloc[tr].reset_index(drop=True)
            X_cov_te = X_cov.iloc[te].reset_index(drop=True)
            X_tr = pd.concat([X_cov_tr, X_brain_tr_red], axis=1)
            X_te = pd.concat([X_cov_te, X_brain_te_red], axis=1)
        else:
            X_tr = X_brain_tr_red
            X_te = X_brain_te_red

        Y_tr, Y_te = Y.iloc[tr], Y.iloc[te]
        if is_pre and not X_cov.empty:
            Y_tr, Y_te = _local_residualize(X_cov, Y, tr, te)

        all_feats = list(X_tr.columns)
        pipeline, param_dist, scoring, metric = create_model_and_param_dist(
            config, all_feats, active_covs if config['covariate_method'] != 'pre_regress' else [], Y=Y
        )

        search = RandomizedSearchCV(
            pipeline, param_dist,
            scoring=inner_scoring_adj,
            cv=inner,
            n_jobs=config['n_cores'],
            n_iter=n_iter,
            refit=True,
            random_state=42
        )
        fit_params = {}
        if weights is not None and not isinstance(pipeline.named_steps['model'], MultiTaskElasticNet):
            fit_params['model__sample_weight'] = weights.iloc[tr].values
        search.fit(X_tr, Y_tr.values if hasattr(Y_tr, 'values') else Y_tr, **fit_params)

        y_preds.append(search.predict(X_te))
        y_trues.append(Y_te.values if hasattr(Y_te, 'values') else Y_te)
        if config['analysis_type'] == 'classification':
            y_probs.append(search.predict_proba(X_te))

    Y_true = np.concatenate(y_trues)
    Y_pred = np.concatenate(y_preds)

    if config['analysis_type'] == 'classification':
        Y_prob = np.concatenate(y_probs)
        score = roc_auc_score(Y_true, Y_prob, multi_class='ovr') if Y_prob.shape[1] > 2 \
            else roc_auc_score(Y_true, Y_prob[:, 1])
    else:
        score = r2_score(Y_true, Y_pred, multioutput='uniform_average')

    logging.info(f"Nested CV Score: {score:.4f}")
    pd.DataFrame({'score': [score]}).to_csv(
        os.path.join(config['paths']['output_dir'], 'nested_cv_scores.csv'), index=False
    )

    # Expanded evaluation metrics (C4)
    try:
        _compute_evaluation_metrics(
            Y_true, Y_pred,
            np.concatenate(y_probs) if y_probs else None,
            config,
            config['paths']['output_dir'],
            is_loo=isinstance(outer, LeaveOneOut)
        )
    except Exception as exc:
        logging.warning(f"_compute_evaluation_metrics failed (non-fatal): {exc}")

    return score


# --- Step 5: Permutation ---
def _run_cv_fold_loop(X_brain, Y, X_cov, active_covs, config, seed, apriori_map=None):
    """Run full nested CV fold loop and return R²/AUC on concatenated outer-fold predictions.

    Shared by _run_perm_task and _run_block_perm_task (F1 clean refactor).
    Callers prepare X_brain (possibly block-permuted) and Y (possibly shuffled)
    before invoking this helper.

    Metric computed identically to run_nested_cv: R² (regression) or AUC (classification)
    from concatenated outer-fold predictions. This ensures the null distribution is on the
    same scale as the observed score, preventing the metric mismatch caused by
    search.score() returning the inner-loop scoring metric (neg_RMSE / neg_log_loss).
    """
    outer = get_outer_cv(config)
    inner = get_inner_cv(config)
    n_iter = config['cv_params']['n_random_search_iter']
    inner_scoring_adj = _adjust_scoring_for_loo(
        SCORING_REGRESSION if config['analysis_type'] == 'regression' else SCORING_CLASSIFICATION,
        inner
    )
    is_pre = config['covariate_method'] == 'pre_regress'
    reducer_template = _make_reducer(config, apriori_map)

    # Use first column of Y for outer split when Y is multi-output
    Y_split = Y.iloc[:, 0] if (hasattr(Y, 'iloc') and Y.ndim > 1 and Y.shape[1] > 1) else Y

    y_preds = []
    y_trues = []
    y_probs = []
    for tr, te in outer.split(X_brain, Y_split):
        X_brain_tr = X_brain.iloc[tr].reset_index(drop=True)
        X_brain_te = X_brain.iloc[te].reset_index(drop=True)
        X_brain_tr_red, X_brain_te_red, _ = _apply_reducer_fold(reducer_template, X_brain_tr, X_brain_te)

        if not X_cov.empty and not is_pre:
            X_cov_tr = X_cov.iloc[tr].reset_index(drop=True)
            X_cov_te = X_cov.iloc[te].reset_index(drop=True)
            X_tr = pd.concat([X_cov_tr, X_brain_tr_red], axis=1)
            X_te = pd.concat([X_cov_te, X_brain_te_red], axis=1)
        else:
            X_tr = X_brain_tr_red
            X_te = X_brain_te_red

        Y_tr, Y_te = Y.iloc[tr], Y.iloc[te]
        if is_pre and not X_cov.empty:
            Y_tr, Y_te = _local_residualize(X_cov, Y, tr, te)

        all_feats = list(X_tr.columns)
        pipeline, param_dist, scoring, _ = create_model_and_param_dist(
            config, all_feats, active_covs if not is_pre else [], Y=Y
        )
        search = RandomizedSearchCV(
            pipeline, param_dist,
            scoring=inner_scoring_adj, cv=inner,
            n_jobs=1, n_iter=n_iter, refit=True, random_state=seed
        )
        search.fit(X_tr, Y_tr.values if hasattr(Y_tr, 'values') else Y_tr)
        y_preds.append(search.predict(X_te))
        y_trues.append(Y_te.values if hasattr(Y_te, 'values') else Y_te)
        if config['analysis_type'] == 'classification':
            y_probs.append(search.predict_proba(X_te))

    Y_true = np.concatenate(y_trues)
    Y_pred = np.concatenate(y_preds)

    if config['analysis_type'] == 'classification':
        Y_prob = np.concatenate(y_probs)
        score = roc_auc_score(Y_true, Y_prob, multi_class='ovr') if Y_prob.shape[1] > 2 \
            else roc_auc_score(Y_true, Y_prob[:, 1])
    else:
        score = r2_score(Y_true, Y_pred, multioutput='uniform_average')

    return score


def _run_perm_task(X_brain, Y, X_cov, active_covs, config, seed, apriori_map=None):
    """
    Single permutation iteration: shuffle Y, run full nested CV.
    ConvergenceWarnings suppressed per-task since permuted data frequently fails to converge.
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    rs = np.random.RandomState(seed)
    Y_shuf = sklearn_shuffle(Y, random_state=rs)
    return _run_cv_fold_loop(X_brain, Y_shuf, X_cov, active_covs, config, seed, apriori_map)


def run_permutation_test(config, X_brain, Y, weights, X_cov, active_covs, actual, job_id, n_jobs, is_worker, apriori_map=None):
    """Run label-permutation test to estimate the null distribution of model performance.

    In worker mode (is_worker=True), runs a subset of total permutations determined
    by job_id and n_jobs, then writes perm_chunk_{job_id}.csv. In local mode
    (is_worker=False), runs all permutations and writes the full null distribution CSV.
    P-value uses Laplace correction: (count(null >= observed) + 1) / (n_perms + 1).
    """
    logging.info("--- Permutation Test ---")
    n_perms = config['stats_params']['n_permutations']
    if n_perms == 0:
        return
    seeds = np.random.RandomState(42).randint(0, int(1e9), n_perms)
    my_seeds = np.array_split(seeds, n_jobs)[job_id] if is_worker else seeds
    res = Parallel(n_jobs=config['n_cores'])(
        delayed(_run_perm_task)(X_brain, Y, X_cov, active_covs, config, s, apriori_map)
        for s in my_seeds
    )
    metric = config.get('_runtime', {}).get('metric', 'score')
    if is_worker:
        pd.DataFrame(res, columns=[f'null_{metric}']).to_csv(
            os.path.join(config['paths']['output_dir'], f'perm_chunk_{job_id}.csv'), index=False
        )
    else:
        p = (np.sum(np.array(res) >= actual) + 1) / (len(res) + 1)
        logging.info(f"P-value: {p:.4f}")
        pd.DataFrame(res, columns=[f'null_{metric}']).to_csv(
            os.path.join(config['paths']['output_dir'], f'permutation_null_distribution_{metric}.csv'), index=False
        )


# --- Step 6: Selection Frequency (formerly Stability Selection, F7 rename) ---
def _fit_full_data_model(config, X_brain, Y, weights, X_cov, active_covs, apriori_map):
    """
    Shared setup for descriptive stages (selection frequency, bootstrap):
    fit reducer on full data, assemble X_full, tune hyperparameters, return fitted model.

    Note: reducer is fit on full data (not cross-validated) because these stages are
    descriptive — see run_nested_cv for the inferential (fold-local) pathway.
    """
    is_pre = config['covariate_method'] == 'pre_regress'
    n_iter = config['cv_params']['n_random_search_iter']

    reducer_template = _make_reducer(config, apriori_map)
    reducer_full = _apply_reducer_full(reducer_template, X_brain)
    # Feature reduction is fit on the full dataset here because this is a descriptive
    # stage. Inferential validity is preserved in run_nested_cv where reduction is
    # strictly fold-local (fit on training data only).
    if reducer_full is not None:
        logging.info(
            "Descriptive stage: feature reducer fit on full dataset (not fold-local). "
            "See run_nested_cv for the inferential (fold-local) pathway."
        )
    X_brain_red = reducer_full.transform(X_brain) if reducer_full is not None else X_brain

    if not X_cov.empty and not is_pre:
        X_full = pd.concat(
            [X_cov.reset_index(drop=True), X_brain_red.reset_index(drop=True)], axis=1
        )
    else:
        X_full = X_brain_red.reset_index(drop=True)

    all_feats = list(X_full.columns)
    pipeline, param_dist, scoring, _ = create_model_and_param_dist(
        config, all_feats, active_covs if not is_pre else [], Y=Y
    )

    fit_Y = Y.reset_index(drop=True) if hasattr(Y, 'reset_index') else pd.Series(Y)
    if is_pre and not X_cov.empty:
        logging.warning(
            "pre_regress: hyperparameter tuning uses full-sample residuals. "
            "This is a minor approximation; individual iteration estimates are unaffected."
        )
        lr = LinearRegression().fit(X_cov, Y)
        fit_Y = pd.Series(Y.values - lr.predict(X_cov), name=Y.name if hasattr(Y, 'name') else 'Y')

    fit_params = {}
    if weights is not None and not isinstance(pipeline.named_steps['model'], MultiTaskElasticNet):
        fit_params['model__sample_weight'] = weights.values

    search = RandomizedSearchCV(
        pipeline, param_dist, scoring=scoring, n_jobs=config['n_cores'],
        n_iter=n_iter, cv=get_inner_cv(config), refit=True, random_state=42
    )
    search.fit(X_full, fit_Y.values, **fit_params)
    best_model = search.best_estimator_
    feat_std_map = pd.Series(best_model.named_steps['scaler'].scale_, index=all_feats)
    best_params = search.best_params_

    return X_full, best_model, reducer_full, feat_std_map, fit_Y, fit_params, best_params


def run_selection_frequency(config, X_brain, Y, weights, X_cov, active_covs, apriori_map=None):
    """Bootstrap selection frequency: a descriptive measure of feature inclusion robustness.

    Approach Y (T5 decision): each subsample iteration fits a fresh reducer clone on
    the subsampled brain features, fits the model in reduced space with fixed
    hyperparameters (tuned on full data), back-projects selection indicators to the
    invariant original feature space using _backproject_coef_original_space.

    Aggregation of selection indicators in original feature space is meaningful across
    iterations with different reducer fits. No significance threshold is applied;
    selection frequency is purely descriptive.
    """
    logging.info('--- Selection Frequency (descriptive) ---')
    n_iter = config['stats_params']['n_bootstraps']
    n_cores = config['n_cores']
    is_pre = config['covariate_method'] == 'pre_regress'
    original_feature_names = list(X_brain.columns)

    X_full, best_model, _, _, _, fit_params, best_params = _fit_full_data_model(
        config, X_brain, Y, weights, X_cov, active_covs, apriori_map
    )
    full_feat_names = list(X_full.columns)
    reducer_template = _make_reducer(config, apriori_map)

    # Determine output feature names: same logic as run_bootstrap
    red_method = config['feature_reduction_method']
    out_feat_names = full_feat_names if red_method == 'none' else original_feature_names

    def subsample_task(seed):
        rng_sub = np.random.RandomState(seed)
        idx = rng_sub.choice(len(Y), len(Y) // 2, replace=False)

        # Resample brain features, fit fresh reducer (Approach Y)
        X_brain_sub = X_brain.iloc[idx].reset_index(drop=True)
        reducer_sub = None
        if reducer_template is not None:
            reducer_sub = clone(reducer_template)
            reducer_sub.feature_names_in_ = original_feature_names
            reducer_sub.fit(X_brain_sub)
            X_brain_sub_red = reducer_sub.transform(X_brain_sub)
        else:
            X_brain_sub_red = X_brain_sub

        # Recombine with covariates (incorporate method)
        if not X_cov.empty and not is_pre:
            X_cov_sub = X_cov.iloc[idx].reset_index(drop=True)
            X_sub = pd.concat([X_cov_sub, X_brain_sub_red], axis=1)
        else:
            X_sub = X_brain_sub_red

        Y_arr = Y.values if hasattr(Y, 'values') else np.array(Y)
        Y_sub = Y_arr[idx]
        if is_pre and not X_cov.empty:
            X_cov_sub_pr = X_cov.iloc[idx].reset_index(drop=True)
            lr_sub = LinearRegression().fit(X_cov_sub_pr, Y_sub)
            Y_sub = Y_sub - lr_sub.predict(X_cov_sub_pr)

        all_feats_sub = list(X_sub.columns)
        m, _, _, _ = create_model_and_param_dist(
            config, all_feats_sub,
            active_covs if not is_pre else [],
            Y=Y
        )
        m.set_params(**best_params)

        if weights is not None and not isinstance(m.named_steps['model'], MultiTaskElasticNet):
            m.fit(X_sub, Y_sub, model__sample_weight=weights.values[idx])
        else:
            m.fit(X_sub, Y_sub)

        c = m.named_steps['model'].coef_
        # Binary classification: coef_ is (1, P) — squeeze to (P,) for single-output path.
        # Multi-class (K>=2): coef_ is (K, P) — preserve for per-class reporting.
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.squeeze(axis=0)
        c_reduced = c  # shape (P,) or (K, P)

        # Split covariate and brain coefficients when incorporate + reduction is active
        n_covs_sub = len(active_covs) if not is_pre else 0
        if reducer_sub is not None and n_covs_sub > 0:
            if c_reduced.ndim == 2:
                c_brain_reduced = c_reduced[:, n_covs_sub:]
            else:
                c_brain_reduced = c_reduced[n_covs_sub:]
            c_orig = _backproject_coef_original_space(c_brain_reduced, reducer_sub, original_feature_names)
        else:
            c_orig = _backproject_coef_original_space(c_reduced, reducer_sub, original_feature_names)
        return (c_orig != 0).astype(int)

    results = Parallel(n_jobs=n_cores)(
        delayed(subsample_task)(s) for s in np.random.RandomState(43).randint(0, int(1e9), n_iter)
    )

    # Detect multi-output (each result is either (P,) or (K, P))
    sample_res = results[0]
    is_multi = sample_res.ndim == 2  # True for multi-task regression or multi-class classification

    if not is_multi:
        # Single-output: results is a list of (P,) arrays
        pd.DataFrame({
            'feature': out_feat_names,
            'selection_probability': np.mean(results, axis=0)
        }).to_csv(
            os.path.join(config['paths']['output_dir'], 'report_selection_frequency.csv'), index=False
        )
    else:
        # Multi-output: stack to (n_iter, K, P), write per-task files + union aggregate
        sel_array = np.stack(results, axis=0)  # (n_iter, K, P)
        task_labels = _get_task_labels(Y, config)
        out_dir = config['paths']['output_dir']
        for k, lbl in enumerate(task_labels):
            out_dir_k = os.path.join(out_dir, f'task_{lbl}')
            os.makedirs(out_dir_k, exist_ok=True)
            pd.DataFrame({
                'feature': out_feat_names,
                'selection_probability': sel_array[:, k, :].mean(axis=0)
            }).to_csv(os.path.join(out_dir_k, 'report_selection_frequency.csv'), index=False)
        # Union aggregate: feature selected in ≥1 task per iteration
        union_sel = (sel_array.sum(axis=1) > 0).astype(int)
        pd.DataFrame({
            'feature': out_feat_names,
            'selection_probability': union_sel.mean(axis=0)
        }).to_csv(os.path.join(out_dir, 'report_selection_frequency.csv'), index=False)


# --- Step 7: Bootstrap & Reporting ---
def _boot_task(X_brain, Y, weights, seed, config, best_params, reducer_template,
               apriori_map=None, X_cov=None, active_covs=None):
    """Single bootstrap iteration (Approach Y: per-iteration reduction + back-projection).

    Each iteration: (1) weight-aware resample, (2) fit fresh reducer clone on
    resampled brain features, (3) fit model in reduced space using best_params
    (hyperparameters fixed from full-data tuning), (4) back-project coefficients
    to original feature space via _backproject_coef_original_space.

    Aggregation in the invariant original feature space is meaningful across
    iterations with different reducer fits. Follows the conditional bootstrap
    framework (Efron & Tibshirani, 1993): hyperparameters fixed, full pipeline
    (reduction + model) re-estimated per iteration.

    Parameters
    ----------
    X_brain : pd.DataFrame
        Original (pre-reduction) brain features, shape (N, P).
    Y : pd.Series or pd.DataFrame
        Outcome variable(s).
    weights : pd.Series or None
        Sample weights for weight-aware resampling (F13 fix).
    seed : int
        Random seed for this iteration.
    config : dict
        Pipeline configuration.
    best_params : dict
        Best hyperparameters from full-data RandomizedSearchCV.best_params_.
    reducer_template : unfitted TransformerMixin or None
        Unfitted reducer factory; cloned and refit per iteration.
    apriori_map : pd.Series or None
        Apriori cluster map (passed through to AprioriTransformer).
    X_cov : pd.DataFrame or None
        Covariate features (required when covariate_method == 'pre_regress').
    active_covs : list of str or None
        Active covariate column names (required when covariate_method == 'incorporate').

    Returns
    -------
    (coef_original, converged) : tuple
        coef_original : ndarray, shape (P,) — coefficients in original feature space.
        converged : bool — False if ConvergenceWarning was raised.
    None on failure.
    """
    rng_boot = np.random.RandomState(seed)
    is_pre = config['covariate_method'] == 'pre_regress'
    active_covs = active_covs or []
    original_feature_names = list(X_brain.columns)

    try:
        # Weight-aware bootstrap resampling (F13 fix)
        if weights is not None:
            p = weights.values / weights.values.sum()
            idx = rng_boot.choice(len(Y), len(Y), replace=True, p=p)
        else:
            idx = rng_boot.choice(len(Y), len(Y), replace=True)

        # Resample brain features and fit fresh reducer
        X_brain_boot = X_brain.iloc[idx].reset_index(drop=True)
        reducer_boot = None
        if reducer_template is not None:
            reducer_boot = clone(reducer_template)
            reducer_boot.feature_names_in_ = original_feature_names
            reducer_boot.fit(X_brain_boot)
            X_brain_boot_red = reducer_boot.transform(X_brain_boot)
        else:
            X_brain_boot_red = X_brain_boot

        # Recombine with covariates if incorporate method
        if X_cov is not None and not X_cov.empty and not is_pre:
            X_cov_boot = X_cov.iloc[idx].reset_index(drop=True)
            X_boot = pd.concat([X_cov_boot, X_brain_boot_red], axis=1)
        else:
            X_boot = X_brain_boot_red

        # Outcome: residualize for pre_regress
        Y_arr = Y.values if hasattr(Y, 'values') else np.array(Y)
        Y_boot = Y_arr[idx]
        if is_pre and X_cov is not None and not X_cov.empty:
            X_cov_boot_pr = X_cov.iloc[idx].reset_index(drop=True)
            lr_boot = LinearRegression().fit(X_cov_boot_pr, Y_boot)
            Y_boot = Y_boot - lr_boot.predict(X_cov_boot_pr)

        # Build pipeline from best_params (no re-tuning)
        all_feats_boot = list(X_boot.columns)
        n_covs_boot = len(active_covs) if not is_pre else 0

        pipeline_boot, _, _, _ = create_model_and_param_dist(
            config, all_feats_boot,
            active_covs if not is_pre else [],
            Y=Y if not hasattr(Y, 'values') else (pd.DataFrame(Y_boot, columns=Y.columns) if Y.ndim > 1 else pd.Series(Y_boot))
        )
        # Set best hyperparameters directly (no RandomizedSearchCV)
        pipeline_boot.set_params(**best_params)

        converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if weights is not None and not isinstance(pipeline_boot.named_steps['model'], MultiTaskElasticNet):
                pipeline_boot.fit(X_boot, Y_boot,
                                  model__sample_weight=weights.values[idx])
            else:
                pipeline_boot.fit(X_boot, Y_boot)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

        c = pipeline_boot.named_steps['model'].coef_
        # Binary classification: coef_ is (1, P) — squeeze to (P,) for single-output path.
        # Multi-class (K>=2): coef_ is (K, P) — preserve for per-class reporting.
        # Regression single-output: coef_ is (P,) — unchanged.
        # Multi-task regression: coef_ is (K, P) — preserved.
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.squeeze(axis=0)
        c_reduced = c  # shape (P,) or (K, P)

        # Back-project to original feature space.
        # When covariates are incorporated AND reduction is active, the model's coef_
        # has shape (n_covs + n_brain_reduced,) for single-output, or
        # (K, n_covs + n_brain_reduced) for multi-output. The loading/mixing matrix only
        # covers brain features, so split off the covariate columns before back-projecting.
        if reducer_boot is not None and n_covs_boot > 0:
            if c_reduced.ndim == 2:
                c_brain_reduced = c_reduced[:, n_covs_boot:]
            else:
                c_brain_reduced = c_reduced[n_covs_boot:]
            c_original = _backproject_coef_original_space(c_brain_reduced, reducer_boot, original_feature_names)
        else:
            c_original = _backproject_coef_original_space(c_reduced, reducer_boot, original_feature_names)

        return c_original, converged
    except Exception as exc:
        logging.debug("Bootstrap iteration failed: %s", exc, exc_info=True)
        return None


def calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model, report_df, level, X_brain_raw=None):
    """Compute subject-level feature-vs-outcome data for visualization and write to CSV.

    For each significant feature/component in report_df, computes the partial
    association between that feature and the outcome after partialling out the
    linear contribution of all other features. Results are written to
    report_{level}_plotting.csv. No file is written if no significant features exist.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    X_full : pd.DataFrame
        Full feature matrix (covariates + reduced brain features) used for fitting.
    Y : pd.Series
        Outcome variable.
    weights : pd.Series or None
        Sample weights (unused in visualization; reserved for API consistency).
    subject_ids : pd.Series
        Subject identifier column.
    best_model : fitted sklearn Pipeline
        Full-data pipeline (scaler + cov_scaler + model).
    report_df : pd.DataFrame
        Importance report containing is_significant, std_coef_mean, and a feature
        identifier column (feature, cluster_id, or component_id).
    level : str
        Label for output file naming: 'cluster' or 'individual'.
    X_brain_raw : pd.DataFrame or None
        Raw (pre-reduction) brain features. Used to look up original feature values
        for ICA back-projection results where feature names refer to raw columns.
    """
    analysis_type = config['analysis_type']
    out_dir = config['paths']['output_dir']
    mask = report_df['is_significant'] == True
    candidates = report_df[mask]
    if candidates.empty:
        return
    coeffs = best_model.named_steps['model'].coef_
    intercept = getattr(best_model.named_steps['model'], 'intercept_', 0.0)
    if coeffs.ndim > 1:
        # For visualization, use the mean across tasks/classes as a summary of the
        # linear contribution of each feature. Per-task visualization is produced
        # via per-task calls to _compute_importance_report in run_bootstrap.
        coeffs = coeffs.mean(axis=0)
    X_scaled = pd.DataFrame(best_model.named_steps['scaler'].transform(X_full), columns=X_full.columns, index=X_full.index)
    linear_pred_full = X_scaled.dot(coeffs) + (intercept.mean() if isinstance(intercept, np.ndarray) else intercept)
    feat_col = 'component_id' if 'component_id' in report_df.columns else (
        'cluster_id' if 'cluster_id' in report_df.columns else 'feature'
    )
    assert Y.ndim == 1, f"calculate_visualization_data expects 1-D Y, got shape {Y.shape}"
    plot_data = []
    Y_arr = Y.values if hasattr(Y, 'values') else Y
    subject_ids_arr = subject_ids.values if hasattr(subject_ids, 'values') else subject_ids
    for _, row in candidates.iterrows():
        f_name = row[feat_col]
        if X_brain_raw is not None and f_name in X_brain_raw.columns:
            f_val_raw = X_brain_raw[f_name]
        elif f_name in X_full.columns:
            f_val_raw = X_full[f_name]
        else:
            continue
        f_weight = row['std_coef_mean']
        f_val_scaled = (f_val_raw - f_val_raw.mean()) / (f_val_raw.std() + 1e-10)
        lin_contrib = f_weight * f_val_scaled
        cov_score = linear_pred_full - lin_contrib
        y_val = (Y_arr - cov_score.values) if analysis_type == 'regression' else expit(lin_contrib + cov_score.mean())
        for i in range(len(Y_arr)):
            plot_data.append({
                'subject_id': subject_ids_arr[i],
                'outcome_raw': Y_arr[i],
                'feature_name': f_name,
                'y_axis_value': y_val[i] if hasattr(y_val, '__getitem__') else y_val.iloc[i]
            })
    if plot_data:
        pd.DataFrame(plot_data).to_csv(os.path.join(out_dir, f'report_{level}_plotting.csv'), index=False)


def _build_individual_report_df(all_feats, stats, feat_col='feature'):
    """Build the standard individual-level importance DataFrame from shared statistics.

    Used by all four report branches to avoid duplicating the 8-column construction.
    Returns the DataFrame with FDR columns already applied.
    """
    df = pd.DataFrame({
        feat_col: all_feats,
        'std_coef_mean': stats['std_means'].values,
        'std_ci_low': stats['std_ci_low'].values,
        'std_ci_high': stats['std_ci_high'].values,
        'raw_coef_mean': stats['raw_means'].values,
        'raw_ci_low': stats['raw_ci_low'].values,
        'raw_ci_high': stats['raw_ci_high'].values,
        'pd': stats['pd_val'].values,
        'is_significant': stats['is_sig'].values
    })
    return _add_fdr_columns(df)


def _add_fdr_columns(df, pd_col='pd'):
    """
    Derive p-values from probability of direction and apply BH-FDR at q=0.05.
    Adds 'p_value' and 'is_significant_fdr' columns in-place.
    """
    pd_vals = df[pd_col].values
    p_values = np.clip(2 * (1 - pd_vals), 0.0, 1.0)
    is_sig_fdr = _bh_fdr(p_values, q=0.05)
    df = df.copy()
    df['p_value'] = p_values
    df['is_significant_fdr'] = is_sig_fdr
    return df



def _compute_importance_preamble(df_coef, all_feats, feat_std_map, config, best_model):
    """
    Compute shared statistics for all reduction methods: std/raw means, CIs, pd, is_significant.
    F6 fix: accounts for CovariateScaler penalty_weight in raw coefficient conversion.

    Notes
    -----
    With Approach Y (per-iteration reduction + back-projection), df_coef is always
    in the original brain feature space. When feature_reduction_method != 'none',
    df_coef contains only brain features (no covariates), so the covariate penalty-
    weight adjustment is only applied when the df_coef columns match the full
    model feature set (i.e., feature_reduction_method == 'none').
    """
    out_dir = config['paths']['output_dir']
    alpha = 1.0 - config['stats_params']['ci_level']
    std_means = df_coef.mean()
    std_ci_low = df_coef.quantile(alpha / 2)
    std_ci_high = df_coef.quantile(1 - alpha / 2)

    # Align feat_std_map to df_coef columns (may differ after Approach Y back-projection)
    feat_std_map_aligned = feat_std_map.reindex(df_coef.columns).fillna(1.0)

    pw = best_model.named_steps['cov_scaler'].penalty_weight
    cov_idx = best_model.named_steps['cov_scaler'].covariate_indices
    feat_std_map_adj = feat_std_map_aligned.copy()
    # Only adjust covariate indices when df_coef actually contains the full model feature set
    # (i.e., no reduction was applied, so covariates appear in df_coef columns)
    red_method = config['feature_reduction_method']
    if cov_idx and pw != 1.0 and red_method == 'none':
        feat_std_map_adj.iloc[cov_idx] = feat_std_map_adj.iloc[cov_idx] / pw

    raw_means = np.divide(std_means, feat_std_map_adj, out=np.zeros_like(std_means), where=feat_std_map_adj != 0)
    raw_ci_low = np.divide(std_ci_low, feat_std_map_adj, out=np.zeros_like(std_ci_low), where=feat_std_map_adj != 0)
    raw_ci_high = np.divide(std_ci_high, feat_std_map_adj, out=np.zeros_like(std_ci_high), where=feat_std_map_adj != 0)

    is_sig = (std_ci_low > 0) | (std_ci_high < 0)
    pd_val = pd.concat([(df_coef > 0).mean(), (df_coef < 0).mean()], axis=1).max(axis=1)

    return dict(
        std_means=std_means, std_ci_low=std_ci_low, std_ci_high=std_ci_high,
        raw_means=raw_means, raw_ci_low=raw_ci_low, raw_ci_high=raw_ci_high,
        is_sig=is_sig, pd_val=pd_val, out_dir=out_dir, alpha=alpha
    )


def _report_apriori(df_coef, all_feats, stats, config, active_covs,
                    reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model):
    """Apriori: individual feature report + network-level aggregation via apriori cluster map.

    With Approach Y, df_coef is in original brain feature space. Network-level
    statistics are produced by aggregating individual feature bootstrap distributions
    to cluster level using the apriori map from reducer_full (mean coefficient per cluster).
    """
    out_dir = stats['out_dir']
    alpha = stats['alpha']

    # --- Individual feature report (Approach Y: df_coef already in original space) ---
    indiv_df = _build_individual_report_df(all_feats, stats)
    indiv_df.to_csv(os.path.join(out_dir, 'report_individual_importance.csv'), index=False)

    # --- Network-level report: aggregate to cluster via apriori map ---
    try:
        if reducer_full is not None and hasattr(reducer_full, 'get_loadings'):
            loadings = pd.DataFrame(reducer_full.get_loadings())
            cluster_rows = []
            for c_name in loadings['cluster'].unique():
                cluster_feats = loadings[loadings['cluster'] == c_name]['feature'].tolist()
                cluster_feats_in = [f for f in cluster_feats if f in df_coef.columns]
                if not cluster_feats_in:
                    continue
                cluster_boot = df_coef[cluster_feats_in].mean(axis=1)  # mean across features
                c_mean = cluster_boot.mean()
                c_low = cluster_boot.quantile(alpha / 2)
                c_high = cluster_boot.quantile(1 - alpha / 2)
                c_pd = max(float((cluster_boot > 0).mean()), float((cluster_boot < 0).mean()))
                cluster_rows.append({
                    'cluster_id': c_name,
                    'std_coef_mean': c_mean,
                    'std_ci_low': c_low,
                    'std_ci_high': c_high,
                    'pd': c_pd,
                    'is_significant': bool((c_low > 0) or (c_high < 0))
                })
            if cluster_rows:
                net_rep = pd.DataFrame(cluster_rows)
                net_rep = _add_fdr_columns(net_rep)
                net_rep.to_csv(os.path.join(out_dir, 'report_cluster_importance.csv'), index=False)
                calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model,
                                             net_rep, 'cluster', X_brain)
    except (KeyError, ValueError, IndexError) as exc:
        logging.warning(f"Apriori network-level aggregation failed: {exc}")
        logging.debug("Back-projection traceback:", exc_info=True)


def _report_cluster_pca(df_coef, all_feats, stats, config, active_covs,
                        reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model):
    """Cluster PCA: individual feature report in original feature space (Approach Y).

    With Approach Y, df_coef is already in original brain feature space after
    back-projection in each bootstrap iteration. Report feature-level statistics directly.
    """
    out_dir = stats['out_dir']
    indiv_df = _build_individual_report_df(all_feats, stats)
    indiv_df.to_csv(os.path.join(out_dir, 'report_individual_importance.csv'), index=False)
    calculate_visualization_data(
        config, X_full, Y, weights, subject_ids, best_model,
        indiv_df.rename(columns={'feature': 'cluster_id'}), 'individual', X_brain
    )


def _report_ica(df_coef, all_feats, stats, config, active_covs,
                reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model):
    """ICA: individual feature report in original feature space (Approach Y).

    With Approach Y, each bootstrap iteration back-projects IC-space coefficients
    to original brain feature space using the activation pattern (Haufe et al., 2014):
       feature_coef = A @ beta_IC  (A = mixing_unnorm_, shape P x K_ic)
    df_coef is therefore already in original brain feature space — report directly.

    The full-data ICA mixing matrix is saved for transparency (descriptive).
    """
    out_dir = stats['out_dir']

    # Save full-data mixing matrix for transparency
    if reducer_full is not None and hasattr(reducer_full, 'mixing_unnorm_'):
        mixing_df = pd.DataFrame(
            reducer_full.mixing_unnorm_,
            index=reducer_full.feature_names_in_,
            columns=reducer_full.ic_names_
        )
        mixing_df.to_csv(os.path.join(out_dir, 'ica_mixing_matrix.csv'))

    # Report features directly (back-projection already done in _boot_task)
    indiv_rep = _build_individual_report_df(all_feats, stats)
    indiv_rep.to_csv(os.path.join(out_dir, 'report_individual_importance.csv'), index=False)
    calculate_visualization_data(
        config, X_full, Y, weights, subject_ids, best_model,
        indiv_rep, 'individual', X_brain
    )


def _report_none(df_coef, all_feats, stats, config, active_covs,
                 reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model):
    """No reduction: report raw feature coefficients directly."""
    out_dir = stats['out_dir']
    rep = _build_individual_report_df(all_feats, stats)
    rep.to_csv(os.path.join(out_dir, 'report_feature_importance.csv'), index=False)
    calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model, rep, 'individual', None)


def _compute_importance_report(df_coef, all_feats, feat_std_map, config, active_covs, reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model):
    """
    Dispatcher: compute shared statistics, then delegate to reduction-specific branch.
    Applies BH-FDR correction (F4 fix). F6 fix preserved in preamble.
    """
    stats = _compute_importance_preamble(df_coef, all_feats, feat_std_map, config, best_model)
    red_method = config['feature_reduction_method']
    branch_args = (df_coef, all_feats, stats, config, active_covs,
                   reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model)
    if red_method == 'apriori':
        _report_apriori(*branch_args)
    elif red_method == 'cluster_pca':
        _report_cluster_pca(*branch_args)
    elif red_method == 'ica':
        _report_ica(*branch_args)
    else:
        _report_none(*branch_args)


def run_bootstrap(config, X_brain, Y, weights, subject_ids, X_cov, active_covs, apriori_map=None):
    """
    Bootstrap feature importance. Reducer fit once on full data (descriptive).
    Individual bootstrap iterations correctly clone and refit on resampled data.
    """
    logging.info("--- Bootstrap Importance ---")
    n_boot = config['stats_params']['n_bootstraps']

    X_full, best_model, reducer_full, feat_std_map, _, _, best_params = _fit_full_data_model(
        config, X_brain, Y, weights, X_cov, active_covs, apriori_map
    )
    reducer_template = _make_reducer(config, apriori_map)
    original_feature_names = list(X_brain.columns)

    # Pass X_cov for both incorporate (to reassemble X_boot with covariates) and
    # pre_regress (to residualize Y within each iteration).
    cov_method = config['covariate_method']
    X_cov_for_boot = X_cov if (cov_method != 'none' and X_cov is not None and not X_cov.empty) else None
    res = Parallel(n_jobs=config['n_cores'])(
        delayed(_boot_task)(
            X_brain, Y, weights, s, config, best_params, reducer_template,
            apriori_map=apriori_map,
            X_cov=X_cov_for_boot,
            active_covs=active_covs
        )
        for s in np.random.RandomState(42).randint(0, int(1e9), n_boot)
    )

    valid_res = [r for r in res if r is not None]
    n_failed = len(res) - len(valid_res)
    n_total = len(res)
    if n_failed > 0:
        pct = 100 * n_failed / n_total
        msg = f"Bootstrap: {n_failed}/{n_total} iterations failed ({pct:.1f}%)"
        if pct > 50.0:
            raise RuntimeError(
                f"{msg}. Majority of bootstrap iterations failed — results would be "
                f"unreliable. Check for dimension mismatches (e.g., incorporate + "
                f"reduction) or data issues."
            )
        elif pct > 5.0:
            logging.warning(msg + " — exceeds 5% threshold; CIs may be unreliable.")
        else:
            logging.info(msg)

    n_conv_warn = sum(1 for _, w in valid_res if not w)

    # Determine df_coef column names.
    # - No reduction: _boot_task returns coefficients for X_boot.columns (covariates + brain).
    #   Use X_full.columns which matches the assembly order in _boot_task.
    # - Reduction: _boot_task back-projects to original brain feature space (X_brain.columns).
    full_feat_names = list(X_full.columns)
    red_method = config['feature_reduction_method']
    df_coef_cols = full_feat_names if red_method == 'none' else original_feature_names

    if n_conv_warn > 0:
        logging.warning(
            f"ConvergenceWarning in {n_conv_warn}/{len(valid_res)} bootstrap iterations."
        )

    # Save full-data cluster/ICA descriptive outputs
    if reducer_full is not None and hasattr(reducer_full, 'get_loadings'):
        pd.DataFrame(reducer_full.get_loadings()).to_csv(
            os.path.join(config['paths']['output_dir'], 'cluster_loadings.csv'), index=False
        )

    # feat_std_map and all_feats for importance reporting
    if red_method == 'none':
        # feat_std_map is in full feature space (covariates + brain); matches df_coef_cols
        feat_std_map_report = feat_std_map
        all_feats_report = full_feat_names
    else:
        # For reduction cases, df_coef is in original brain feature space (Approach Y).
        # Use X_brain std as feat_std proxy (scaler was fit on reduced-space features).
        # NOTE: std_coef_mean is APPROXIMATE after back-projection — it represents the
        # product of reduced-space standardized coef × loading, divided by original feature
        # SD. This is NOT equivalent to a standardized beta from direct regression on
        # original features. The raw_coef_mean and pd (probability of direction) columns
        # are the primary inferential quantities; std_coef_mean should be interpreted as
        # relative importance within the analysis only.
        brain_std = X_brain.std(ddof=1).replace(0, 1.0)
        feat_std_map_report = brain_std
        all_feats_report = original_feature_names

    # Detect multi-output from the first valid result
    sample_coef = valid_res[0][0]
    is_multi = sample_coef.ndim == 2  # True for multi-task regression or multi-class

    if not is_multi:
        # Single-output: each result is (P,) — build df_coef (B, P) as before
        df_coef = pd.DataFrame([r[0] for r in valid_res], columns=df_coef_cols)

        # Save distributions if requested
        if config['stats_params'].get('save_distributions', True):
            coef_dist = np.stack([r[0] for r in valid_res], axis=0)
            np.savez_compressed(
                os.path.join(config['paths']['output_dir'], 'bootstrap_coef_distribution.npz'),
                coef_dist=coef_dist,
                feature_names=np.array(df_coef_cols)
            )

        _compute_importance_report(
            df_coef, all_feats_report, feat_std_map_report, config, active_covs,
            reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model
        )
    else:
        # Multi-output: each result is (K, P) — stack to (B, K, P), then loop over tasks/classes
        coef_array = np.stack([r[0] for r in valid_res], axis=0)  # (B, K, P)
        task_labels = _get_task_labels(Y, config)

        # Save joint distribution if requested
        if config['stats_params'].get('save_distributions', True):
            np.savez_compressed(
                os.path.join(config['paths']['output_dir'], 'bootstrap_coef_distribution.npz'),
                coef_dist=coef_array,
                feature_names=np.array(df_coef_cols),
                task_labels=np.array(task_labels)
            )

        for k, lbl in enumerate(task_labels):
            out_dir_k = os.path.join(config['paths']['output_dir'], f'task_{lbl}')
            os.makedirs(out_dir_k, exist_ok=True)
            config_k = {**config, 'paths': {**config['paths'], 'output_dir': out_dir_k}}
            df_coef_k = pd.DataFrame(coef_array[:, k, :], columns=df_coef_cols)
            # Pass single-task Y slice so visualization/reporting operates on 1-D outcome
            Y_k = Y.iloc[:, k] if hasattr(Y, 'iloc') and Y.ndim > 1 else Y
            _compute_importance_report(
                df_coef_k, all_feats_report, feat_std_map_report, config_k, active_covs,
                reducer_full, X_brain, X_full, Y_k, weights, subject_ids, best_model
            )


# --- Step 8: Block Permutation ---
def _run_block_perm_task(X_brain, X_block_cols, Y, X_cov, active_covs, config, seed, apriori_map=None):
    """
    Single block permutation iteration: shuffle block columns (rows), run full nested CV.
    F2 fix: null uses full nested CV with shuffled block columns.
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    rs = np.random.RandomState(seed)
    X_brain_perm = X_brain.copy()
    block_data = X_brain_perm[X_block_cols].values
    shuffled_idx = rs.permutation(len(block_data))
    X_brain_perm[X_block_cols] = block_data[shuffled_idx]
    return _run_cv_fold_loop(X_brain_perm, Y, X_cov, active_covs, config, seed, apriori_map)


def run_block_perms(config, X_brain, Y, weights, X_cov, active_covs, actual_score, apriori_map=None):
    """
    Block permutation test. F2 fix: observed = nested CV score (passed as actual_score).
    Null = full nested CV with shuffled block columns. Uses n_block_permutations config param.
    """
    logging.info("--- Block Permutation ---")
    blocks = config.get('block_permutation_tests')
    if not blocks:
        return
    n_perms = config['stats_params'].get('n_block_permutations', 500)
    n_cores = config['n_cores']
    results = []

    for label, definition in blocks.items():
        if isinstance(definition, str):
            b_cols = [c for c in X_brain.columns if definition in c]
        else:
            b_cols = [c for c in X_brain.columns if c in definition]
        if not b_cols:
            logging.warning(f"Block '{label}': no matching columns found. Skipping.")
            continue
        seeds = np.random.RandomState(42).randint(0, int(1e9), n_perms)
        null_scores = Parallel(n_jobs=n_cores)(
            delayed(_run_block_perm_task)(
                X_brain, b_cols, Y, X_cov, active_covs, config, s, apriori_map
            )
            for s in seeds
        )
        p_value = (np.sum(np.array(null_scores) >= actual_score) + 1) / (len(null_scores) + 1)
        results.append({'block': label, 'observed_score': actual_score, 'p_value': p_value})
        logging.info(f"Block '{label}': observed={actual_score:.4f}, p={p_value:.4f}")

        # Save block permutation null distribution (C5)
        if config['stats_params'].get('save_distributions', True):
            pd.DataFrame({'null_score': null_scores}).to_csv(
                os.path.join(config['paths']['output_dir'], f'block_perm_null_{label}.csv'),
                index=False
            )

    pd.DataFrame(
        results if results else [], columns=['block', 'observed_score', 'p_value']
    ).to_csv(
        os.path.join(config['paths']['output_dir'], 'report_block_permutation.csv'), index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--mode', default='main')
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--skip_main_perm', action='store_true')
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = config['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, args.job_id if args.mode == 'perm_worker' else None)

    # Validate required config parameters
    if 'n_random_search_iter' not in config.get('cv_params', {}):
        sys.exit("CRITICAL: config.cv_params.n_random_search_iter is required but absent. "
                 "Recommended range: 10-100. For a 2-parameter space (alpha, l1_ratio), 20-50 is typically sufficient.")

    try:
        X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map = load_and_prep_data(config, out_dir)

        if args.mode == 'perm_worker':
            run_permutation_test(
                config, X_brain, Y, weights, X_cov, active_covs,
                0.0, args.job_id, args.n_jobs, True, apriori_map
            )
        elif args.mode == 'aggregate':
            chunk_files = sorted(glob.glob(os.path.join(out_dir, 'perm_chunk_*.csv')))
            if not chunk_files:
                logging.error("No perm_chunk_*.csv files found. Ensure perm_worker jobs completed.")
                sys.exit(1)
            null_dist = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
            metric_col = null_dist.columns[0]
            null_vals = null_dist[metric_col].values

            score_file = os.path.join(out_dir, 'nested_cv_scores.csv')
            if not os.path.exists(score_file):
                logging.error(f"nested_cv_scores.csv not found at {score_file}.")
                sys.exit(1)
            observed = pd.read_csv(score_file)['score'].iloc[0]
            p_value = (np.sum(null_vals >= observed) + 1) / (len(null_vals) + 1)
            logging.info(
                f"Aggregate: {len(null_vals)} permutations. Observed={observed:.4f}. p={p_value:.4f}"
            )
            clean_metric = metric_col.replace("null_", "")
            null_dist.to_csv(
                os.path.join(out_dir, f'permutation_null_distribution_{clean_metric}.csv'), index=False
            )
            pd.DataFrame({
                'observed_score': [observed],
                'p_value': [p_value],
                'n_permutations': [len(null_vals)]
            }).to_csv(os.path.join(out_dir, 'permutation_result.csv'), index=False)

        else:  # main mode
            actual = run_nested_cv(config, X_brain, Y, weights, X_cov, active_covs, apriori_map)
            # Store metric for perm worker label
            _, _, _, metric = create_model_and_param_dist(config, ['dummy'], [], Y=Y)
            config.setdefault('_runtime', {})['metric'] = metric

            run_selection_frequency(config, X_brain, Y, weights, X_cov, active_covs, apriori_map)
            run_bootstrap(config, X_brain, Y, weights, subj_ids, X_cov, active_covs, apriori_map)
            if not args.skip_main_perm:
                run_permutation_test(
                    config, X_brain, Y, weights, X_cov, active_covs,
                    actual, 0, 1, False, apriori_map
                )
            run_block_perms(config, X_brain, Y, weights, X_cov, active_covs, actual, apriori_map)

    except Exception:
        logging.error("PIPELINE FAILED", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
