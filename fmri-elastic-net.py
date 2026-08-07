"""
Elastic Net Regression and Binary/Multi-Class Classification Pipeline

FEATURES:
- Flexible data loading (covariates + brain features) with listwise deletion and N:P diagnostics.
- Feature Reduction: Raw Features, HDBSCAN Clustering + PCA, Apriori Clustering, ICA.
  All reduction methods are applied fold-locally inside the CV loop to prevent data leakage.
- Fold-wise Ensemble Architecture: run_nested_cv stores all K fold models (reducer +
  fitted pipeline + best_params + back-projected coefficients per fold). The ensemble IS the
  collection of evaluated fold models — there is no separate full-data model, eliminating
  the evaluation-inference disconnect present in full-data descriptive approaches.
- Two-Tier Inference Framework:
    Tier 1 (fold-level t-test): one-sample t-test across K fold-specific coefficient vectors
    per feature. Captures both sampling and hyperparameter tuning variance. Written to
    report_fold_ensemble_importance.csv.
    Tier 2 (pooled fold-wise bootstrap): n_fold_bootstraps iterations distributed evenly
    across all K folds (minimum 50 per fold); coefficients pooled for percentile CIs.
    Distribution-shape-agnostic; detects non-normality and zero-inflation. Written to
    report_fold_bootstrap_ci.csv.
- Per-iteration re-reduction bootstrap: each iteration fits a fresh reducer clone on resampled
  brain features, fits the model in reduced space with fold-specific fixed hyperparameters, and
  back-projects coefficients to the invariant original feature space for meaningful aggregation.
- Covariate Handling: Incorporate, Pre-Regress (Cross-Val), or None.
- Multi-task Weighted Regression: sqrt(w_i) pre-transformation applied for MultiTaskElasticNet
  + sample_weight (algebraically equivalent to weighted loss; sklearn native sample_weight
  used for all other model types). Implemented via WeightTransformer pipeline step.
- Two modes for Analysis: Predict (full elastic net spectrum, L1 0.01–0.99, data-driven)
  vs. Correlate (Ridge-dominant, L1 0.001–0.2).
- Validation: Supports Repeated K-Fold and LOO for small datasets.
- Comprehensive Evaluation Metrics: RMSE/MAE/R²/Pearson r (regression); AUC-ROC/Log-Loss/
  Sensitivity/Specificity/Balanced Accuracy (classification). Written to model_performance.csv.
- Outputs: Standardized AND approximate Raw Coefficients; distribution archives when save_distributions=true.
- Multi-task regression support (MultiTaskElasticNet; per-task reporting).
- Full Compatibility: Classification (AUC/LogLoss) & Regression (RMSE/R2).

STATISTICAL NOTES:
- The pd-to-p-value conversion (p = 2*(1 - pd)) assumes a continuous coefficient
  distribution. For sparse features with L1 regularization, bootstrap distributions can
  be zero-inflated: many iterations produce exactly zero, causing pd < 0.5 and p > 1.0
  before clipping. In such cases p is clipped to 1.0. The CI-based is_significant flag
  is the primary inference criterion and is unaffected by zero-inflation. The p_value
  and is_significant_fdr columns are complementary and should be interpreted alongside
  is_significant (Makowski et al., 2019).
- Two-tier inference design: Tier 1 (fold-level t-test) and Tier 2 (pooled bootstrap
  percentile CIs) have complementary sensitivity profiles. Tier 1 is intentionally
  anti-conservative: within-fold training set overlap (~(K-1)/K for K-fold CV) induces
  positive correlation between fold coefficient estimates, underestimating SE (Bengio &
  Grandvalet, 2004). This produces a liberal screening test. Tier 2 is conservative due
  to L1 zero-inflation — features with moderate effects are zeroed in a substantial
  fraction of bootstrap iterations, pulling CIs toward zero. The scientific value lies in
  concordance: features significant in both tiers are high-confidence; features significant
  only in Tier 1 are candidates warranting further investigation. Making Tier 1
  better-calibrated (e.g., via Nadeau-Bengio correction or repeated outer CV) would
  converge the two tiers toward the same sensitivity profile, eliminating the
  screening/confirmation distinction.
- Tier 2 bootstrap CIs are computed from a mixture of iterations with different
  hyperparameter configurations (fold-specific best_params). Percentile CIs from this
  mixture may have sub-nominal coverage; they are best interpreted as sensitivity
  diagnostics (Efron & Tibshirani, 1993, Ch. 13).
- raw_coef_mean with reduction methods (cluster_pca, apriori, ica) is approximate: the
  back-projected standardized coefficient is divided by original-feature SD, which is not
  equivalent to a standardized beta from direct regression on original features. The
  std_coef_mean and pd columns are the primary inferential quantities.
- ICA feature back-projection uses the activation pattern (A @ beta_IC, Haufe et al.,
  2014, NeuroImage) rather than the filter pattern (pinv(A) @ beta_IC). Activation
  patterns represent signal co-variation in feature space and are more interpretable and
  stable for neuroimaging feature attribution.
- Selection frequency uses fold-specific best_params; each N/2 subsample sees weaker
  regularization than the full-N tuned model, which may inflate absolute selection
  frequencies. Relative ordering is preserved; no significance threshold is applied
  (Meinshausen & Bühlmann, 2010, JRSS-B).
- Multi-task weighted regression: sqrt(w_i) scaling of X and Y is equivalent to
  weighted least squares in the loss term only. The L1+L2 penalty term is invariant to
  the transformation — this is appropriate because regularization controls model
  complexity independently of observation representativeness. Alpha is re-tuned on
  the transformed data by RandomizedSearchCV (Efron & Hastie, 2021, Ch. 7).
- StandardScaler uses unweighted sample moments (mean and variance) before the
  sqrt(w) WeightTransformer step. This is an intentional design choice: standardization
  is a numerical conditioning step and alpha is re-tuned on the transformed data to
  compensate for scale changes. For typical inverse probability weighting (IPW) ranges
  (~0.5–5.0) the effect on coefficient estimates is negligible. For extreme weight
  ranges (e.g., 0.01–100.0), consider pre-standardizing features using weighted mean
  and variance before pipeline execution.

Written by: Taylor J. Keding, Ph.D.
"""

import os
import sys
import glob
import yaml
import logging
import warnings
import argparse
from math import ceil

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
from scipy.stats import pearsonr, ttest_1samp
from scipy.stats import t as t_dist

# Known-harmless warnings (ConvergenceWarning, Firth overflow/ill-conditioning)
# are suppressed in main(). The _boot_task catch_warnings block still captures
# convergence state for the run_bootstrap logging summary.


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


def _is_multitask(config, Y):
    """Return True if Y is a multi-output regression target.

    Multi-task is defined as: regression mode AND Y is a 2-D array/DataFrame
    with more than one column. Y is always a pandas Series or DataFrame from
    load_and_prep_data, so no hasattr guard is required.
    """
    return (
        config['analysis_type'] == 'regression'
        and Y.ndim > 1
        and Y.shape[1] > 1
    )


def _squeeze_binary_coef(c):
    """Squeeze a (1, P) binary classification coef_ array to (P,).

    Binary LogisticRegression returns coef_ with shape (1, P).
    Multi-class returns (K, P) and must not be squeezed.
    Regression single-output returns (P,) — unchanged.
    Multi-task regression returns (K, P) — unchanged.
    """
    if c.ndim == 2 and c.shape[0] == 1:
        return c.squeeze(axis=0)
    return c


def _strip_covariates(c, n_covs):
    """Strip prepended covariate columns from a coefficient array.

    When covariate_method='incorporate', covariate features are prepended to the
    assembled feature matrix. The model's coef_ therefore includes n_covs leading
    coefficients that must be removed before back-projection to brain feature space.

    Parameters
    ----------
    c : ndarray, shape (P_total,) or (K, P_total)
    n_covs : int
        Number of covariate features prepended. 0 is a no-op.

    Returns
    -------
    c_brain : ndarray
        Coefficient array with covariate columns stripped.
    """
    if n_covs == 0:
        return c
    if c.ndim == 2:
        return c[:, n_covs:]
    return c[n_covs:]


def _strip_protected(c, n_covs, n_moderator_cols, n_reduced):
    """Strip covariate and moderator columns from coefficient array, split into brain blocks.

    Parameters
    ----------
    c : ndarray
        Shape (P_total,) or (K, P_total).
    n_covs : int
        Number of covariate columns prepended to the feature matrix.
    n_moderator_cols : int
        Number of moderator columns following the covariate columns.
    n_reduced : int
        Number of brain features (reduced or raw) in the main block.

    Returns
    -------
    c_brain_main : ndarray
        Coefficients for main brain features.
    c_brain_interaction : ndarray or None
        Coefficients for interaction features, or None when n_moderator_cols == 0.
    """
    n_protected = n_covs + n_moderator_cols
    brain_start = n_protected
    interaction_start = brain_start + n_reduced
    is_2d = c.ndim == 2
    if n_moderator_cols == 0:
        if is_2d:
            return c[:, brain_start:], None
        return c[brain_start:], None
    if is_2d:
        return c[:, brain_start:interaction_start], c[:, interaction_start:]
    return c[brain_start:interaction_start], c[interaction_start:]


def _code_moderator(moderator_series, moderator_type, train_idx, levels_override=None):
    """Fold-local coding of the moderator variable.

    For continuous moderators: center on the training-set mean. For nominal
    moderators: apply deviation (effect) coding relative to the last sorted
    level as the reference category.

    Parameters
    ----------
    moderator_series : pd.Series
        Full-sample moderator values (length N).
    moderator_type : {'continuous', 'nominal'}
        Coding strategy to apply.
    train_idx : array-like of int
        Row indices belonging to the training fold.
    levels_override : list or None, optional
        Pre-computed sorted level list for nominal moderators. When provided,
        overrides training-set-derived levels. Used by bootstrap and selection
        frequency paths to ensure consistent coding dimensions across
        resampled iterations. Default: None (derive from train_idx).

    Returns
    -------
    moderator_coded : pd.DataFrame
        Coded moderator columns (N rows).
    coding_info : dict
        Metadata: 'type', 'train_mean' (continuous) or 'levels' (nominal).
    """
    if moderator_type == 'continuous':
        train_mean = moderator_series.iloc[train_idx].mean()
        centered = moderator_series - train_mean
        moderator_coded = pd.DataFrame({'moderator': centered.values},
                                       index=moderator_series.index)
        coding_info = {'type': 'continuous', 'train_mean': train_mean}
        return moderator_coded, coding_info

    # nominal: deviation (effect) coding
    if levels_override is not None:
        levels = levels_override
    else:
        train_vals = moderator_series.iloc[train_idx]
        levels = sorted(train_vals.unique().tolist())
    K = len(levels)
    ref_level = levels[-1]

    if K == 2:
        col_names = ['moderator']
    else:
        col_names = [f'moderator_contrast_{j + 1}' for j in range(K - 1)]

    codes = np.zeros((len(moderator_series), K - 1), dtype=float)
    for row_i, val in enumerate(moderator_series):
        if val not in levels:
            # Unseen test level: all contrasts remain 0
            continue
        if val == ref_level:
            codes[row_i, :] = -1.0
        else:
            j = levels.index(val)
            codes[row_i, j] = 1.0

    moderator_coded = pd.DataFrame(codes, columns=col_names,
                                   index=moderator_series.index)
    coding_info = {'type': 'nominal', 'levels': levels, 'reference': ref_level}
    return moderator_coded, coding_info


def _construct_interactions(X_reduced_brain, moderator_coded):
    """Element-wise multiply each brain feature column by each moderator column.

    Parameters
    ----------
    X_reduced_brain : pd.DataFrame
        Brain feature matrix (N, n_reduced).
    moderator_coded : pd.DataFrame
        Coded moderator columns (N, n_mod_cols).

    Returns
    -------
    pd.DataFrame
        Interaction columns named '{brain_col}_x_{mod_col}',
        shape (N, n_reduced * n_mod_cols).
    """
    interaction_cols = {}
    for mod_col in moderator_coded.columns:
        for brain_col in X_reduced_brain.columns:
            col_name = f'{brain_col}_x_{mod_col}'
            interaction_cols[col_name] = (
                X_reduced_brain[brain_col].values * moderator_coded[mod_col].values
            )
    return pd.DataFrame(interaction_cols, index=X_reduced_brain.index)


def _firth_logistic(X, y, max_iter=25, tol=1e-5):
    """Firth-penalized logistic regression via IRLS.

    Addresses complete or quasi-complete separation by adding a Jeffreys
    prior penalty (Firth 1993). Self-contained; no external fitting library.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Feature matrix (should not include intercept column).
    y : ndarray, shape (n,)
        Binary outcome (0/1).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence threshold on penalized log-likelihood change.

    Returns
    -------
    beta : ndarray, shape (p,)
        Estimated coefficients (excluding intercept).
    intercept : float
        Estimated intercept.
    converged : bool
        Whether the algorithm converged within max_iter.
    """
    import scipy.linalg
    n, p = X.shape
    beta = np.zeros(p)
    intercept = 0.0
    ll_prev = -np.inf
    converged = False
    eye_p1 = np.eye(p + 1)

    for iteration in range(max_iter):
        eta = X @ beta + intercept
        mu = 1.0 / (1.0 + np.exp(-eta))
        W_diag = mu * (1.0 - mu)
        X_aug = np.column_stack([np.ones(n), X])
        Fisher = X_aug.T @ (W_diag[:, None] * X_aug)
        Fisher_inv = scipy.linalg.solve(Fisher, eye_p1, assume_a='pos')
        H_diag = (X_aug @ Fisher_inv * (W_diag[:, None] * X_aug)).sum(axis=1)
        U = X_aug.T @ (y - mu + H_diag * (0.5 - mu))
        delta = Fisher_inv @ U
        intercept += delta[0]
        beta += delta[1:]

        # Penalized log-likelihood (Firth 1993)
        eta2 = X @ beta + intercept
        mu2 = 1.0 / (1.0 + np.exp(-eta2))
        W2 = mu2 * (1.0 - mu2)
        X_aug2 = np.column_stack([np.ones(n), X])
        Fisher2 = X_aug2.T @ (W2[:, None] * X_aug2)
        sign, logdet = np.linalg.slogdet(Fisher2)
        ll = (
            np.sum(y * np.log(mu2 + 1e-300) + (1.0 - y) * np.log(1.0 - mu2 + 1e-300))
            + 0.5 * (logdet if sign > 0 else np.log(1e-300))
        )
        if iteration > 0 and abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll

    return beta, intercept, converged


def _partial_ridge_refit(X, Y, selected_mask, config, weights=None,
                         n_covs=0, n_moderator_cols=0):
    """Partial Ridge refit for a single bootstrap iteration.

    Selected features receive OLS (regression) or unpenalized logistic
    (classification) treatment; unselected features receive Ridge shrinkage.
    Protected features (covariates + moderator columns) are always forced
    into the selected set. Firth logistic is applied as a fallback when
    complete separation is detected in classification.

    Parameters
    ----------
    X : ndarray or pd.DataFrame, shape (n, p)
        Full feature matrix.
    Y : ndarray, shape (n,) or (n, K_tasks)
        Outcome variable(s).
    selected_mask : bool ndarray, shape (p,)
        True for features selected by the elastic net.
    config : dict
        Pipeline configuration dict.
    weights : ndarray or None
        Sample weights, shape (n,).
    n_covs : int
        Number of covariate columns.
    n_moderator_cols : int
        Number of moderator columns following covariates.

    Returns
    -------
    coef : ndarray
        Refitted coefficients. Shape (p,) for regression/binary, (K, p) for
        multi-class or multi-task.
    firth_used : bool
        True if Firth logistic was invoked for at least one class.
    """
    from sklearn.linear_model import LogisticRegression

    X_arr = np.asarray(X)
    n, p = X_arr.shape

    # Force protected features into the selected set
    selected_mask = selected_mask.copy()
    selected_mask[:n_covs + n_moderator_cols] = True

    S_idx = np.where(selected_mask)[0]
    Sc_idx = np.where(~selected_mask)[0]
    coef = np.zeros(p)
    firth_used = False

    is_multitask = (
        config['analysis_type'] == 'regression'
        and np.asarray(Y).ndim > 1
        and np.asarray(Y).shape[1] > 1
    )

    if config['analysis_type'] == 'regression':
        Y_arr = np.asarray(Y)

        def _refit_single_row(y_vec):
            c = np.zeros(p)
            if len(S_idx) > 0:
                X_S = X_arr[:, S_idx]
                res = np.linalg.lstsq(X_S, y_vec, rcond=None)
                c[S_idx] = res[0]
            if len(Sc_idx) > 0:
                X_Sc = X_arr[:, Sc_idx]
                lam = 1.0 / n
                eye_sc = np.eye(len(Sc_idx))
                c[Sc_idx] = np.linalg.solve(
                    X_Sc.T @ X_Sc + lam * eye_sc, X_Sc.T @ y_vec
                )
            return c

        if is_multitask:
            rows = []
            for k in range(Y_arr.shape[1]):
                rows.append(_refit_single_row(Y_arr[:, k]))
            return np.vstack(rows), False
        else:
            return _refit_single_row(Y_arr.ravel()), False

    # Classification path
    Y_arr = np.asarray(Y).ravel()
    classes = np.unique(Y_arr)
    is_multiclass = len(classes) > 2

    def _refit_binary(y_vec, x_mat, s_idx, sc_idx):
        c_row = np.zeros(x_mat.shape[1])
        firth = False
        if len(s_idx) == 0:
            # Only Ridge unselected
            lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                    max_iter=5000)
            fit_kw = {'sample_weight': weights} if weights is not None else {}
            lr.fit(x_mat, y_vec, **fit_kw)
            c_row[:] = lr.coef_.ravel()
            return c_row, firth

        sqrt_n = np.sqrt(n)
        X_scaled = x_mat.copy()
        X_scaled[:, s_idx] *= sqrt_n
        lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                max_iter=5000)
        fit_kw = {'sample_weight': weights} if weights is not None else {}
        lr.fit(X_scaled, y_vec, **fit_kw)
        coef_scaled = lr.coef_.ravel().copy()
        c_row[:] = coef_scaled
        c_row[s_idx] *= sqrt_n

        # Separation check
        if len(s_idx) > 0 and np.max(np.abs(c_row[s_idx])) > 10.0:
            beta_f, _, _ = _firth_logistic(x_mat[:, s_idx], y_vec)
            c_row[s_idx] = beta_f
            firth = True

        return c_row, firth

    if not is_multiclass:
        c_out, firth_used = _refit_binary(Y_arr, X_arr, S_idx, Sc_idx)
        return c_out, firth_used
    else:
        # One-vs-rest per class
        rows = []
        for cls in classes:
            y_bin = (Y_arr == cls).astype(float)
            c_row, firth_cls = _refit_binary(y_bin, X_arr, S_idx, Sc_idx)
            rows.append(c_row)
            if firth_cls:
                firth_used = True
        return np.vstack(rows), firth_used


def _hotelling_t2(coef_matrix, ci_level):
    """Hotelling's T-squared test for multivariate mean of fold-level coefficients.

    Parameters
    ----------
    coef_matrix : ndarray, shape (n_folds, K_minus_1)
        Per-fold coefficient estimates (one row per fold, one column per
        contrast or feature).
    ci_level : float
        Confidence level (e.g., 0.95); not used in the test statistic itself
        but returned for reference.

    Returns
    -------
    dict
        Keys: 'T2', 'F', 'p_value', 'df1', 'df2', 'ci_level'.
    """
    from scipy.stats import f as f_dist
    n, p_dim = coef_matrix.shape
    df1 = p_dim
    df2 = n - p_dim
    if df2 <= 0:
        logging.warning(
            'Hotelling T2: n_folds (%d) <= n_contrasts (%d); '
            'F-test undefined. Per-contrast t-tests and Tier 2 L2-norm '
            'CIs remain valid.', n, p_dim
        )
        return {'T2': float('nan'), 'F': float('nan'), 'p_value': float('nan'),
                'df1': df1, 'df2': df2, 'ci_level': ci_level}
    x_bar = coef_matrix.mean(axis=0)
    S = np.cov(coef_matrix, rowvar=False, ddof=1)
    if p_dim == 1:
        S = np.atleast_2d(S)
    if np.linalg.matrix_rank(S) < p_dim:
        logging.warning(
            'Hotelling T2: covariance matrix is rank-deficient '
            '(rank %d < %d). Returning NaN.', np.linalg.matrix_rank(S), p_dim
        )
        return {'T2': float('nan'), 'F': float('nan'), 'p_value': float('nan'),
                'df1': df1, 'df2': df2, 'ci_level': ci_level}
    S_inv = np.linalg.inv(S)
    T2 = float(n * x_bar @ S_inv @ x_bar)
    F = (n - p_dim) / (p_dim * (n - 1)) * T2
    p_value = float(1.0 - f_dist.cdf(F, df1, df2))
    return {'T2': T2, 'F': F, 'p_value': p_value, 'df1': df1, 'df2': df2,
            'ci_level': ci_level}


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
        """No-op fit; transformer has no learnable parameters."""
        return self

    def transform(self, X):
        """Scale covariate columns by 1/penalty_weight; return X unchanged if no-op."""
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


class ModeratorScaler(BaseEstimator, TransformerMixin):
    """Scale moderator columns by 1/0.001 to protect them from regularization.

    Applies a fixed 1000x amplification to moderator feature columns so that
    the elastic net's penalty effectively ignores them (analogous to
    CovariateScaler but with a fixed, non-tuned scale factor). Operates
    independently from CovariateScaler and is a no-op when moderator_indices
    is None.

    Parameters
    ----------
    moderator_indices : list of int or None
        Column indices of moderator features. If None, transform is a no-op.
    """
    def __init__(self, moderator_indices=None):
        self.moderator_indices = moderator_indices

    def fit(self, X, y=None, **params):
        """No-op fit; transformer has no learnable parameters."""
        return self

    def transform(self, X):
        """Scale moderator columns by 1/0.001; return X unchanged if no-op."""
        if self.moderator_indices is None:
            return X
        X_t = X.copy()
        scale_factor = 1.0 / 0.001
        if isinstance(X_t, pd.DataFrame):
            X_t.iloc[:, self.moderator_indices] *= scale_factor
        else:
            X_t[:, self.moderator_indices] *= scale_factor
        return X_t


class WeightTransformer(BaseEstimator, TransformerMixin):
    """Scale X (and optionally Y) by sqrt(sample_weight) for multi-task weighted regression.

    sklearn's MultiTaskElasticNet does not accept sample_weight in fit(). Multiplying
    both X and Y by sqrt(w_i) is algebraically equivalent to weighted least squares in
    the loss term:
        sum_i w_i * ||y_i - X_i @ B||^2  ===  ||diag(sqrt(w)) @ (Y - X @ B)||^2_F

    The L1+L2 penalty term is invariant to this transformation (appropriate — regularization
    controls model complexity independently of observation representativeness). Alpha is
    re-tuned on the transformed data by RandomizedSearchCV, compensating for scale change.

    When is_multitask=False (or weights_ is None), all methods are no-ops: the transformer
    acts as a pass-through for consistent pipeline step naming.

    Parameters
    ----------
    is_multitask : bool
        If True AND weights_ is set, apply sqrt(w) transformation.
        If False, transform is a no-op regardless of weights_.
    """
    def __init__(self, is_multitask=False):
        self.is_multitask = is_multitask
        self.weights_ = None

    def set_weights(self, w):
        """Set sample weights for this transformer. w must be a 1-D non-negative array."""
        w = np.asarray(w, dtype=float)
        if np.any(w < 0):
            raise ValueError("Sample weights must be non-negative; got negative values.")
        self.weights_ = w
        return self

    def fit(self, X, y=None, **params):
        """No-op fit; weights are set externally via set_weights()."""
        return self

    def transform(self, X):
        """Scale X by sqrt(weights_) row-wise; no-op when is_multitask=False or weights_ is None."""
        if not self.is_multitask or self.weights_ is None:
            return X
        sqrt_w = np.sqrt(self.weights_)
        if isinstance(X, pd.DataFrame):
            return X.multiply(sqrt_w, axis=0)
        return X * sqrt_w[:, np.newaxis]

    def transform_y(self, Y):
        """Apply sqrt(w) scaling to the outcome matrix Y (called before fit, outside pipeline)."""
        if not self.is_multitask or self.weights_ is None:
            return Y
        sqrt_w = np.sqrt(self.weights_)
        if hasattr(Y, 'values'):
            Y_arr = Y.values
        else:
            Y_arr = np.asarray(Y)
        if Y_arr.ndim == 1:
            return Y_arr * sqrt_w
        return Y_arr * sqrt_w[:, np.newaxis]


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
        """Apply per-cluster PCA; return DataFrame of cluster PC scores (or raw singleton values)."""
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
        """Return list of output feature names: cluster labels for multi-feature clusters, raw names for singletons."""
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
        """Fit HDBSCAN clustering on training features, then fit per-cluster PCAs."""
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
        """Fit per-cluster PCAs using the externally-defined apriori cluster map."""
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
    Fit FastICA with unit-variance whitening directly on standardized features.
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
        """Fit StandardScaler and FastICA on training features; determine K via Parallel Analysis if auto."""
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
        # FastICA with unit-variance whitening acts directly on standardized
        # features. mixing_ is P x K and maps IC activations back to feature space.
        self.ica_ = FastICA(
            n_components=K,
            max_iter=ica_cfg.get('max_iter', 1000),
            random_state=ica_cfg.get('random_state', 42),
            whiten='unit-variance'
        )
        self.ica_.fit(X_std)
        # mixing_unnorm_: P x K, no column-wise normalization
        self.mixing_unnorm_ = self.ica_.mixing_
        self.ic_names_ = [f"IC_{i + 1}" for i in range(K)]
        return self

    def transform(self, X):
        """Standardize X and apply fitted ICA; return DataFrame of IC activation scores."""
        # feature_names_in_ set at fit time; DataFrame wrapping only when X arrives as ndarray
        X_df = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X
        X_std = self.scaler_.transform(X_df)
        X_ica = self.ica_.transform(X_std)
        return pd.DataFrame(X_ica, columns=self.ic_names_,
                            index=X_df.index if hasattr(X_df, 'index') else None)

    def get_feature_names_out(self):
        """Return list of IC names (IC_1, IC_2, ..., IC_K)."""
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
    moderator : dict or None
        When data_cols.moderator_col is specified: dict with keys 'series' (pd.Series of
        moderator values), 'type' (str, moderator_type from config), and 'K' (int number
        of unique levels for nominal moderators, None for continuous). None when
        moderator_col is null.

    Raises
    ------
    FileNotFoundError
        If paths.data_file does not exist.
    ValueError
        If all rows are dropped after listwise deletion, required covariates are missing,
        the apriori clustering file is missing/invalid/incomplete, moderator_col is not
        found in data, or nominal moderator K-1 >= n_outer_folds.
    """
    logging.info("--- Loading and Preprocessing Data ---")
    data_cfg = config['data_cols']
    try:
        df = pd.read_csv(config['paths']['data_file'])
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Data file not found: {config['paths']['data_file']}") from err

    # Listwise deletion with informative logging
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

    if weights is not None:
        if (weights < 0).any():
            raise ValueError(
                "CRITICAL: sample_weight_col contains negative values. "
                "All weights must be non-negative."
            )
        n_w = len(weights)
        sum_w = weights.sum()
        ess = (sum_w ** 2) / (weights ** 2).sum()
        logging.info(
            f"Sample weights: ESS={ess:.1f} / N={n_w} (ratio={ess/n_w:.3f}). "
            f"Original sum={sum_w:.4f}."
        )
        if ess / n_w < 0.5:
            logging.warning(
                f"Sample weight ESS/N={ess/n_w:.3f} < 0.5: effective sample size is "
                f"less than half the nominal N. Extreme weight heterogeneity may "
                f"compromise regularization path stability and bootstrap coverage. "
                f"Consider examining the weight distribution and trimming extreme "
                f"values as a sensitivity analysis."
            )
        weights = weights * (n_w / sum_w)
        logging.info(
            f"Sample weights normalized to sum to N={n_w} "
            f"(original sum={sum_w:.4f}; normalized sum={weights.sum():.4f})."
        )

    # Moderator
    moderator_col = data_cfg.get('moderator_col')
    moderator_type = data_cfg.get('moderator_type')
    moderator = None
    if moderator_col is not None:
        if moderator_col not in df.columns:
            raise ValueError(f"CRITICAL: moderator_col '{moderator_col}' not found in data.")
        mod_series = df[moderator_col]
        K_mod = None
        if moderator_type == 'nominal':
            K_mod = len(mod_series.unique())
            # Hotelling invertibility check
            n_outer = config['cv_params']['n_outer_folds']
            if isinstance(n_outer, str) and n_outer.lower() == 'loo':
                n_outer_int = len(df)
            else:
                n_outer_int = int(n_outer)
            if K_mod - 1 >= n_outer_int:
                raise ValueError(
                    f"CRITICAL: nominal moderator has K={K_mod} levels, requiring K-1={K_mod-1} "
                    f"for Hotelling's T-squared, but n_outer_folds={n_outer_int}. "
                    f"Increase n_outer_folds or reduce moderator levels."
                )
            P_brain_for_warn = len([c for c in df.columns if data_cfg['brain_feature_substr'] in c])
            if K_mod > 10:
                logging.warning(
                    f"Nominal moderator has K={K_mod} levels, producing {K_mod-1} interaction terms per "
                    f"brain feature (total interaction features: {P_brain_for_warn * (K_mod-1)}). "
                    f"Consider whether this moderator should be collapsed or treated as continuous."
                )
            elif K_mod >= 5:
                logging.warning(
                    f"Nominal moderator has K={K_mod} levels, producing {K_mod-1} interaction terms per "
                    f"brain feature (total interaction features: {P_brain_for_warn * (K_mod-1)}). "
                    f"Estimation stability may be reduced with high feature count."
                )
        moderator = {'series': mod_series, 'type': moderator_type, 'K': K_mod}
        logging.info(f"Moderator: '{moderator_col}' (type={moderator_type}" +
                     (f", K={K_mod}" if K_mod else "") + ")")

    # Covariates
    cov_method = config['covariate_method']
    covariate_cols = data_cfg.get('covariate_cols', [])
    if cov_method == 'none':
        logging.info("Covariate Method: 'none' (Bypass)")
        X_cov = pd.DataFrame(index=Y.index)
        active_covs = []
    else:
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing covariates: {missing}")
        X_cov = df[covariate_cols]
        active_covs = list(X_cov.columns)

    # Brain Features
    brain_substr = data_cfg['brain_feature_substr']
    drop_cols = [data_cfg['subject_id_col']] + (post_col if isinstance(post_col, list) else [post_col])
    if moderator_col:
        drop_cols.append(moderator_col)
    all_cols = df.columns.drop(drop_cols, errors='ignore')
    brain_cols = [c for c in all_cols if brain_substr in c]
    X_brain = df[brain_cols]
    logging.info(
        f"Initial Load: {len(Y)} samples. {len(active_covs)} Covariates, {len(brain_cols)} Brain Features."
    )

    # Sanity Check — top-10 absolute univariate Pearson and Spearman correlations
    Y_vec = Y if Y.ndim == 1 else Y.iloc[:, 0]
    pearson_corrs = X_brain.corrwith(Y_vec, method='pearson')
    spearman_corrs = X_brain.corrwith(Y_vec, method='spearman')
    top_10_pearson = pearson_corrs.abs().sort_values(ascending=False).head(10)
    top_10_spearman = spearman_corrs.abs().sort_values(ascending=False).head(10)
    logging.info(
        f"--- Data Sanity Check: Top 10 Absolute Univariate Pearson Correlations ---\n{top_10_pearson}"
    )
    logging.info(
        f"--- Data Sanity Check: Top 10 Absolute Univariate Spearman Correlations ---\n{top_10_spearman}"
    )

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
    return X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map, moderator


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


def create_model_and_param_dist(config, all_feature_names, active_covariate_names, Y=None, weights=None, moderator_indices=None):
    """
    Build sklearn Pipeline and RandomizedSearchCV param_dist.
    Covariates are always prepended; cov_indices = range(n_covariates).

    When analysis_type is 'regression' AND Y is multi-output AND weights are provided,
    a WeightTransformer step (is_multitask=True) is inserted between 'scaler' and
    'cov_scaler'. For all other cases WeightTransformer is inserted as a no-op
    (is_multitask=False) for consistent pipeline step naming.
    """
    mode = config['analysis_mode']
    model_cfg = config['model_params']
    seed = config['cv_params']['random_state']
    # Covariates are prepended, so indices are always the first len(active_covariate_names) positions
    n_covs = len(active_covariate_names)
    cov_indices = list(range(n_covs)) if n_covs > 0 else []

    if mode == 'predict':
        l1_min = model_cfg.get('l1_min_predict', 0.01)
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

    # Wire covariate penalty_weight loguniform distribution: only when covariates are active
    if config['covariate_method'] != 'none' and cov_indices:
        pw_min = model_cfg.get('covariate_penalty_weight_min', 0.001)
        pw_max = model_cfg.get('covariate_penalty_weight_max', 1.0)
        param_dist['cov_scaler__penalty_weight'] = loguniform(pw_min, pw_max)

    # Activate WeightTransformer only for multi-task regression with sample weights.
    # For all other cases insert as a no-op (is_multitask=False) for consistent step naming.
    is_mt = (
        config['analysis_type'] == 'regression'
        and Y is not None
        and Y.ndim > 1 and Y.shape[1] > 1
        and weights is not None
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('weight_transformer', WeightTransformer(is_multitask=is_mt)),
        ('cov_scaler', CovariateScaler(covariate_indices=cov_indices if cov_indices else None)),
        ('mod_scaler', ModeratorScaler(moderator_indices=moderator_indices)),
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


def run_nested_cv(config, X_brain, Y, weights, X_cov, active_covs, apriori_map=None, moderator=None):
    """Run nested cross-validation with fold-local feature reduction.

    Feature reduction is applied strictly inside the outer CV loop — fit on training
    folds only — to prevent information leakage from test folds.

    Parameters
    ----------
    moderator : dict or None
        If provided, a dict with keys 'series' (pd.Series of moderator values) and
        'type' (str, one of 'continuous', 'binary', 'nominal'). When not None,
        moderator main-effect columns and brain-by-moderator interaction columns are
        constructed fold-locally (post-reduction) and appended to the feature matrix.

    Returns
    -------
    score : float
        CV performance score (R² for regression, AUC-ROC for classification).
    fold_models : list of dict
        Per-fold model records for ensemble inference. Each dict contains:
        - fold_idx : int — global fold index (0-based)
        - pipeline : fitted sklearn Pipeline
        - reducer : fitted reducer (or None)
        - best_params : dict — best hyperparameters from RandomizedSearchCV
        - coef_original : ndarray — fold-specific coefficients in original brain feature space
        - coef_original_interaction : ndarray or None — fold-specific interaction coefficients
          back-projected to original brain feature space; shape (P,) for binary/continuous
          moderator, (n_moderator_cols, P) for K>2 nominal single-output, or
          (K_class, n_moderator_cols, P) for K>2 nominal multi-class
        - feat_std_map : pd.Series — StandardScaler scale_ indexed by all_feats
        - all_feats : list of str — assembled feature names (covariates + reduced brain)
        - n_covs : int — number of covariate features prepended
        - n_moderator_cols : int — number of moderator columns (0 when moderator is None)
        - n_reduced : int — number of reduced brain features for this fold
    """
    logging.info("--- Nested CV ---")
    outer = get_outer_cv(config)
    inner = get_inner_cv(config)
    n_iter = config['cv_params']['n_random_search_iter']
    inner_scoring_adj = _adjust_scoring_for_loo(SCORING_REGRESSION if config['analysis_type'] == 'regression' else SCORING_CLASSIFICATION, inner)
    is_pre = config['covariate_method'] == 'pre_regress'
    reducer_template = _make_reducer(config, apriori_map)
    original_feature_names = list(X_brain.columns)
    P_brain = len(original_feature_names)

    # Determine K (number of outer folds) for LOO vs. K-fold branching.
    n_outer_cfg = config['cv_params']['n_outer_folds']
    if isinstance(n_outer_cfg, str) and n_outer_cfg.lower() == 'loo':
        K_outer = len(Y)  # LOO: K = N
    else:
        K_outer = int(n_outer_cfg)

    y_preds, y_probs, y_trues = [], [], []
    fold_models = []

    # Use a 1-D array for CV splitting: KFold ignores Y values for regression;
    # for multi-task regression use first column; for multi-class use Y directly.
    split_Y = Y.iloc[:, 0] if hasattr(Y, 'ndim') and Y.ndim > 1 else Y
    for fold_idx, (tr, te) in enumerate(outer.split(X_brain, split_Y)):
        # Fold-local reduction: fit on training brain features only
        X_brain_tr = X_brain.iloc[tr].reset_index(drop=True)
        X_brain_te = X_brain.iloc[te].reset_index(drop=True)
        X_brain_tr_red, X_brain_te_red, fold_reducer = _apply_reducer_fold(reducer_template, X_brain_tr, X_brain_te)

        # Small-sample warning for selection frequency and bootstrap subsample viability
        if fold_reducer is not None and hasattr(fold_reducer, 'get_feature_names_out'):
            P_reduced = len(fold_reducer.get_feature_names_out())
        else:
            P_reduced = P_brain
        P_model = P_reduced
        if moderator is not None:
            n_mod_cols = moderator['K'] - 1 if moderator['type'] == 'nominal' else 1
            P_model = P_reduced + n_mod_cols + P_reduced * n_mod_cols
        half_n = len(tr) // 2
        if half_n < max(3 * P_model, 30):
            logging.warning(
                f"Fold {fold_idx}: 50%% subsample size ({half_n}) is below the recommended "
                f"minimum (max(3*P_model={3*P_model}, 30)={max(3*P_model, 30)}). "
                f"Selection frequency and bootstrap CIs may be unreliable for this fold."
            )

        # Save per-fold reducer outputs for transparency
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

        # Interaction construction (post-reduction, fold-local)
        n_moderator_cols = 0
        moderator_coded_tr = None
        moderator_coded_te = None
        interaction_tr = None
        interaction_te = None
        n_reduced = X_brain_tr_red.shape[1]

        if moderator is not None:
            mod_coded, coding_info = _code_moderator(moderator['series'], moderator['type'], tr)
            moderator_coded_tr = mod_coded.iloc[tr].reset_index(drop=True)
            moderator_coded_te = mod_coded.iloc[te].reset_index(drop=True)
            n_moderator_cols = moderator_coded_tr.shape[1]

            interaction_tr = _construct_interactions(X_brain_tr_red, moderator_coded_tr)
            interaction_te = _construct_interactions(X_brain_te_red, moderator_coded_te)

        # Recombine: [covariates, moderator_main, reduced_brain, interactions]
        parts_tr = []
        parts_te = []
        if not X_cov.empty and config['covariate_method'] != 'pre_regress':
            X_cov_tr = X_cov.iloc[tr].reset_index(drop=True)
            X_cov_te = X_cov.iloc[te].reset_index(drop=True)
            parts_tr.append(X_cov_tr)
            parts_te.append(X_cov_te)
        if moderator_coded_tr is not None:
            parts_tr.append(moderator_coded_tr)
            parts_te.append(moderator_coded_te)
        parts_tr.append(X_brain_tr_red)
        parts_te.append(X_brain_te_red)
        if interaction_tr is not None:
            parts_tr.append(interaction_tr)
            parts_te.append(interaction_te)
        X_tr = pd.concat(parts_tr, axis=1)
        X_te = pd.concat(parts_te, axis=1)

        Y_tr, Y_te = Y.iloc[tr], Y.iloc[te]
        if is_pre and not X_cov.empty:
            Y_tr, Y_te = _local_residualize(X_cov, Y, tr, te)

        all_feats = list(X_tr.columns)
        n_covs = len(active_covs) if config['covariate_method'] != 'pre_regress' else 0
        moderator_indices = list(range(n_covs, n_covs + n_moderator_cols)) if n_moderator_cols > 0 else None
        pipeline, param_dist, scoring, metric = create_model_and_param_dist(
            config, all_feats, active_covs if config['covariate_method'] != 'pre_regress' else [],
            Y=Y, weights=weights, moderator_indices=moderator_indices
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

        # WeightTransformer (multi-task regression + weights): set weights before fit,
        # transform Y before passing to search (Y never enters StandardScaler in pipeline).
        is_mt = _is_multitask(config, Y)
        Y_tr_fit = Y_tr.values if hasattr(Y_tr, 'values') else Y_tr

        fit_params = {}
        if weights is not None and not is_mt:
            fit_params['model__sample_weight'] = weights.iloc[tr].values

        if is_mt and weights is not None:
            wt = search.estimator.named_steps['weight_transformer']
            wt.set_weights(weights.iloc[tr].values)
            Y_tr_fit = wt.transform_y(Y_tr_fit)

        search.fit(X_tr, Y_tr_fit, **fit_params)

        y_preds.append(search.predict(X_te))
        y_trues.append(Y_te.values if hasattr(Y_te, 'values') else Y_te)
        if config['analysis_type'] == 'classification':
            y_probs.append(search.predict_proba(X_te))

        # Extract fold-specific coefficients and back-project to original brain feature space.
        # For multi-task + weights: search.best_estimator_.weight_transformer.weights_ is None
        # because sklearn's clone() (called by RandomizedSearchCV for each parameter combination
        # and for the refit step) preserves only __init__ parameters, dropping weights_ which is
        # set via set_weights(). Construct a correctly-weighted fold model by re-fitting a fresh
        # pipeline with best_params and correct weights. Note: hyperparameter selection used
        # unweighted data in the inner CV; the refit here produces correct fold-model
        # coefficients with the intended weight transformation.
        if is_mt and weights is not None:
            best_pipe = clone(search.estimator)
            best_pipe.set_params(**search.best_params_)
            wt_refit = best_pipe.named_steps['weight_transformer']
            wt_refit.set_weights(weights.iloc[tr].values)
            Y_tr_refit = wt_refit.transform_y(
                Y_tr.values if hasattr(Y_tr, 'values') else Y_tr
            )
            best_pipe.fit(X_tr, Y_tr_refit)
            best_estimator = best_pipe
        else:
            best_estimator = search.best_estimator_
        c = _squeeze_binary_coef(best_estimator.named_steps['model'].coef_)
        # Split off covariate, moderator, and interaction parts before back-projection.
        # _strip_protected strips covariates and moderator columns and returns the brain-main
        # and interaction coefficient blocks separately.
        c_brain_main, c_brain_interaction = _strip_protected(c, n_covs, n_moderator_cols, n_reduced)
        coef_original = _backproject_coef_original_space(c_brain_main, fold_reducer, original_feature_names)
        coef_original_interaction = None
        if c_brain_interaction is not None:
            if n_moderator_cols > 1:
                # K>2 nominal: c_brain_interaction has shape (n_reduced * n_moderator_cols,) or
                # (K_class, n_reduced * n_moderator_cols). Reshape per-contrast block and
                # back-project each contrast independently using the same fold reducer/loading matrix.
                if c_brain_interaction.ndim == 1:
                    reshaped = c_brain_interaction.reshape(n_moderator_cols, n_reduced)
                    bp_blocks = [_backproject_coef_original_space(reshaped[j], fold_reducer, original_feature_names) for j in range(n_moderator_cols)]
                    coef_original_interaction = np.stack(bp_blocks, axis=0)  # (n_moderator_cols, P)
                else:
                    K_class = c_brain_interaction.shape[0]
                    reshaped = c_brain_interaction.reshape(K_class, n_moderator_cols, n_reduced)
                    bp_blocks = [[_backproject_coef_original_space(reshaped[k, j], fold_reducer, original_feature_names) for j in range(n_moderator_cols)] for k in range(K_class)]
                    coef_original_interaction = np.array(bp_blocks)  # (K_class, n_moderator_cols, P)
            else:
                coef_original_interaction = _backproject_coef_original_space(c_brain_interaction, fold_reducer, original_feature_names)

        feat_std_map = pd.Series(best_estimator.named_steps['scaler'].scale_, index=all_feats)

        fold_models.append({
            'fold_idx': fold_idx,
            'pipeline': best_estimator,
            'reducer': fold_reducer,
            'best_params': search.best_params_,
            'coef_original': coef_original,
            'coef_original_interaction': coef_original_interaction,
            'feat_std_map': feat_std_map,
            'all_feats': all_feats,
            'n_covs': n_covs,
            'n_moderator_cols': n_moderator_cols,
            'n_reduced': n_reduced,
        })

    if moderator is not None:
        violations = []
        for fm in fold_models:
            c_fm = _squeeze_binary_coef(fm['pipeline'].named_steps['model'].coef_)
            c_main_red, c_int_red = _strip_protected(c_fm, fm['n_covs'], fm['n_moderator_cols'], fm['n_reduced'])
            if c_int_red is None:
                continue
            main_selected = np.abs(c_main_red) > 1e-10
            n_mod_cols_fm = fm['n_moderator_cols']
            n_red_fm = fm['n_reduced']
            if n_mod_cols_fm > 1:
                if c_int_red.ndim == 1:
                    int_reshaped = c_int_red.reshape(n_mod_cols_fm, n_red_fm)
                    int_any_selected = np.any(np.abs(int_reshaped) > 1e-10, axis=0)
                else:
                    # multi-class: any class, any contrast
                    int_reshaped = c_int_red.reshape(c_int_red.shape[0], n_mod_cols_fm, n_red_fm)
                    int_any_selected = np.any(np.abs(int_reshaped) > 1e-10, axis=(0, 1))
                    main_selected = np.any(np.atleast_2d(main_selected), axis=0) if main_selected.ndim > 1 else main_selected
            else:
                if c_int_red.ndim == 1:
                    int_any_selected = np.abs(c_int_red) > 1e-10
                else:
                    int_any_selected = np.any(np.abs(c_int_red) > 1e-10, axis=0)
                    main_selected = np.any(np.atleast_2d(main_selected), axis=0) if main_selected.ndim > 1 else main_selected
            for i in range(n_red_fm):
                if int_any_selected[i] and not main_selected[i]:
                    violations.append((i, fm['fold_idx']))
        if violations:
            from collections import Counter
            feat_fold_counts = Counter(i for i, _ in violations)
            K_folds_total = len(fold_models)
            logging.warning(
                f"Heredity diagnostic: {len(feat_fold_counts)} reduced-space brain features "
                f"show interaction selected without main effect in at least one fold."
            )
            pd.DataFrame([
                {'feature_index': i, 'n_folds_violation': cnt, 'n_folds_total': K_folds_total}
                for i, cnt in feat_fold_counts.items()
            ]).to_csv(os.path.join(config['paths']['output_dir'], 'report_heredity_diagnostic.csv'), index=False)
        else:
            logging.info("Heredity diagnostic: no violations detected.")

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

    # Compute expanded evaluation metrics post-hoc from outer CV predictions
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

    return score, fold_models


# --- Step 5: Tier 1 Inference (fold-wise ensemble) ---

def _write_fold_diagnostics(fold_models, out_dir):
    """Write per-fold hyperparameter diagnostics and summary statistics.

    Outputs
    -------
    report_fold_diagnostics.csv : per-fold alpha_or_C, l1_ratio, penalty_weight
    report_fold_params_summary.csv : mean ± SD, min, max across all K folds
    """
    rows = []
    for fm in fold_models:
        bp = fm['best_params']
        rows.append({
            'fold_idx': fm['fold_idx'],
            'alpha_or_C': bp.get('model__alpha', bp.get('model__C', np.nan)),
            'l1_ratio': bp.get('model__l1_ratio', np.nan),
            'penalty_weight': bp.get('cov_scaler__penalty_weight', np.nan),
        })
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(os.path.join(out_dir, 'report_fold_diagnostics.csv'), index=False)

    # Summary statistics across all folds
    summary_rows = []
    for col in ['alpha_or_C', 'l1_ratio', 'penalty_weight']:
        vals = diag_df[col].dropna()
        if len(vals) == 0:
            continue
        summary_rows.append({
            'param': col,
            'mean': float(vals.mean()),
            'sd': float(vals.std(ddof=1)) if len(vals) > 1 else float('nan'),
            'min': float(vals.min()),
            'max': float(vals.max()),
            'n_folds': len(vals),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, 'report_fold_params_summary.csv'), index=False
    )


def _write_tier1_report(coef_matrix, feature_names, out_dir, ci_level, effect_type=None, contrast=None):
    """Compute and write Tier 1 inference: one-sample t-test across K fold coefficients.

    Parameters
    ----------
    coef_matrix : ndarray, shape (K, P)
        Fold-specific coefficients in original brain feature space.
    feature_names : list of str, length P
        Original brain feature names.
    out_dir : str
        Output directory.
    ci_level : float
        Confidence level (e.g. 0.95).
    effect_type : str or None
        If not None, an 'effect_type' column is added to the output DataFrame.
        Use 'main' for main-effect rows and 'interaction' for interaction rows.
        When None (no moderator configured), behavior is identical to the
        pre-interaction codepath: no new columns, single fresh write.
    contrast : str or None
        If effect_type is not None, a 'contrast' column is added to the output
        DataFrame identifying the specific contrast (e.g. 'main', 'moderator',
        'contrast_1', 'omnibus').

    Outputs
    -------
    report_fold_ensemble_importance.csv
        fold_mean_coef, fold_sd_coef, fold_cv_coef, t_statistic, p_value_t,
        ci_low_t, ci_high_t, is_significant, is_significant_fdr
        [plus effect_type, contrast when effect_type is not None]
    """
    KR, P = coef_matrix.shape
    fold_mean = coef_matrix.mean(axis=0)
    fold_sd = coef_matrix.std(axis=0, ddof=1)
    # Coefficient of variation (unsigned): |SD / mean|; guard against division by zero
    fold_cv = np.where(np.abs(fold_mean) > 1e-30, np.abs(fold_sd / fold_mean), np.nan)

    t_stat, p_val = ttest_1samp(coef_matrix, popmean=0, axis=0)

    # t-based CI (two-sided)
    alpha_t = 1.0 - ci_level
    dof = KR - 1
    se = fold_sd / np.sqrt(KR)
    t_crit = t_dist.ppf(1.0 - alpha_t / 2, dof)
    ci_low = fold_mean - t_crit * se
    ci_high = fold_mean + t_crit * se

    is_sig = (ci_low > 0) | (ci_high < 0)
    is_sig_fdr = _bh_fdr(p_val, q=0.05)

    df_out = pd.DataFrame({
        'feature': feature_names,
        'fold_mean_coef': fold_mean,
        'fold_sd_coef': fold_sd,
        'fold_cv_coef': fold_cv,
        't_statistic': t_stat,
        'p_value_t': p_val,
        'ci_low_t': ci_low,
        'ci_high_t': ci_high,
        'is_significant': is_sig,
        'is_significant_fdr': is_sig_fdr,
    })
    if effect_type is not None:
        df_out['effect_type'] = effect_type
        df_out['contrast'] = contrast
    out_path = os.path.join(out_dir, 'report_fold_ensemble_importance.csv')
    if effect_type is not None and effect_type != 'main':
        df_out.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df_out.to_csv(out_path, index=False)


def run_tier1_inference(config, fold_models, X_brain, Y, active_covs, moderator=None):
    """Run Tier 1 inference: one-sample t-test across K fold-specific coefficient vectors.

    For each feature, tests H0: mean fold coefficient = 0 using a one-sample t-test
    across the K fold-specific coefficient vectors. This captures both sampling variance
    and hyperparameter tuning variance (different best_params per fold).

    Design intent: Tier 1 is a liberal screen. Within-fold training set overlap
    (~(K-1)/K) induces positive correlation between fold coefficient estimates,
    underestimating SE and producing anti-conservative CIs (Bengio & Grandvalet, 2004).
    This is intentional — the anti-conservative bias ensures Tier 1 casts a wider net
    than Tier 2. Tier 2 bootstrap CIs are conservative due to L1 zero-inflation.
    Features significant in both tiers are high-confidence; features significant only
    in Tier 1 are candidates warranting further investigation (see STATISTICAL NOTES).

    Outputs
    -------
    report_fold_ensemble_importance.csv (or task_{lbl}/report_fold_ensemble_importance.csv)
    report_fold_diagnostics.csv
    report_fold_params_summary.csv
    """
    logging.info("--- Tier 1 Inference (fold-wise ensemble t-test) ---")
    out_dir = config['paths']['output_dir']
    ci_level = config['stats_params']['ci_level']
    original_feature_names = list(X_brain.columns)

    logging.info(
        "Tier 1 design: liberal screen (anti-conservative by design). "
        "Within-fold overlap underestimates SE (Bengio & Grandvalet, 2004). "
        "Interpret through concordance with Tier 2 bootstrap CIs. "
        "Features significant in both tiers: high-confidence. "
        "Features significant only in Tier 1: candidates for further investigation."
    )

    # Determine if multi-output from the first fold's coef_original
    sample_coef = fold_models[0]['coef_original']
    is_multi = sample_coef.ndim == 2  # (K_tasks, P)

    if not is_multi:
        # Single-output: stack (K, P)
        coef_matrix = np.stack([fm['coef_original'] for fm in fold_models], axis=0)
        _write_tier1_report(
            coef_matrix, original_feature_names, out_dir, ci_level,
            effect_type='main' if moderator is not None else None,
            contrast=None if moderator is None else 'main',
        )

        # Interaction-effect inference (single-output only; multi-task left as documented limitation)
        if moderator is not None:
            sample_interaction = fold_models[0].get('coef_original_interaction')
            n_mod_cols = fold_models[0].get('n_moderator_cols', 0)
            if sample_interaction is not None:
                if n_mod_cols <= 1:
                    # Continuous or binary nominal: single contrast, same t-test as main effects
                    coef_matrix_interaction = np.stack(
                        [fm['coef_original_interaction'] for fm in fold_models], axis=0
                    )
                    _write_tier1_report(
                        coef_matrix_interaction, original_feature_names, out_dir, ci_level,
                        effect_type='interaction', contrast='moderator',
                    )
                else:
                    # K>2 nominal: coef_original_interaction per fold has shape (n_mod_cols, P)
                    coef_array_interaction = np.stack(
                        [fm['coef_original_interaction'] for fm in fold_models], axis=0
                    )
                    # coef_array_interaction shape: (K_folds, n_mod_cols, P)
                    K_folds_local = coef_array_interaction.shape[0]
                    P_local = coef_array_interaction.shape[2]

                    # Per-feature Hotelling's T-squared omnibus test
                    hotelling_rows = []
                    for i in range(P_local):
                        interaction_vectors = coef_array_interaction[:, :, i]  # (K_folds, n_mod_cols)
                        result = _hotelling_t2(interaction_vectors, ci_level)
                        hotelling_rows.append({
                            'feature': original_feature_names[i],
                            'effect_type': 'interaction',
                            'contrast': 'omnibus',
                            'T2_statistic': result['T2'],
                            'F_statistic': result['F'],
                            'p_value_T2': result['p_value'],
                            'is_significant': bool(result['p_value'] < (1.0 - ci_level)),
                        })
                    pd.DataFrame(hotelling_rows).to_csv(
                        os.path.join(out_dir, 'report_fold_ensemble_importance.csv'),
                        mode='a', header=False, index=False,
                    )

                    # Per-contrast univariate t-tests
                    for j in range(n_mod_cols):
                        contrast_col = coef_array_interaction[:, j, :]  # (K_folds, P)
                        _write_tier1_report(
                            contrast_col, original_feature_names, out_dir, ci_level,
                            effect_type='interaction', contrast=f'contrast_{j + 1}',
                        )
    else:
        # Multi-output: coef_original is (K_tasks, P); stack to (K, K_tasks, P)
        coef_array = np.stack([fm['coef_original'] for fm in fold_models], axis=0)
        task_labels = _get_task_labels(Y, config)
        for k, lbl in enumerate(task_labels):
            out_dir_k = os.path.join(out_dir, f'task_{lbl}')
            os.makedirs(out_dir_k, exist_ok=True)
            coef_matrix_k = coef_array[:, k, :]  # (K, P)
            _write_tier1_report(coef_matrix_k, original_feature_names, out_dir_k, ci_level,
                                effect_type='main' if moderator is not None else None,
                                contrast='main' if moderator is not None else None)

        # Interaction-effect inference (multi-output)
        if moderator is not None:
            sample_interaction = fold_models[0].get('coef_original_interaction')
            n_mod_cols = fold_models[0].get('n_moderator_cols', 0)
            if sample_interaction is not None:
                coef_array_interaction = np.stack(
                    [fm['coef_original_interaction'] for fm in fold_models], axis=0
                )

                for k, lbl in enumerate(task_labels):
                    out_dir_k = os.path.join(out_dir, f'task_{lbl}')
                    if n_mod_cols <= 1:
                        coef_matrix_interaction_k = coef_array_interaction[:, k, :]  # (K_folds, P)
                        _write_tier1_report(
                            coef_matrix_interaction_k, original_feature_names, out_dir_k, ci_level,
                            effect_type='interaction', contrast='moderator',
                        )
                    else:
                        coef_slice_k = coef_array_interaction[:, k, :, :]  # (K_folds, n_mod_cols, P)
                        K_folds_local = coef_slice_k.shape[0]
                        P_local = coef_slice_k.shape[2]

                        hotelling_rows = []
                        for i in range(P_local):
                            interaction_vectors = coef_slice_k[:, :, i]  # (K_folds, n_mod_cols)
                            result = _hotelling_t2(interaction_vectors, ci_level)
                            hotelling_rows.append({
                                'feature': original_feature_names[i],
                                'effect_type': 'interaction',
                                'contrast': 'omnibus',
                                'T2_statistic': result['T2'],
                                'F_statistic': result['F'],
                                'p_value_T2': result['p_value'],
                                'is_significant': bool(result['p_value'] < (1.0 - ci_level)),
                            })
                        pd.DataFrame(hotelling_rows).to_csv(
                            os.path.join(out_dir_k, 'report_fold_ensemble_importance.csv'),
                            mode='a', header=False, index=False,
                        )

                        for j in range(n_mod_cols):
                            contrast_col = coef_slice_k[:, j, :]  # (K_folds, P)
                            _write_tier1_report(
                                contrast_col, original_feature_names, out_dir_k, ci_level,
                                effect_type='interaction', contrast=f'contrast_{j + 1}',
                            )

    _write_fold_diagnostics(fold_models, out_dir)
    logging.info("Tier 1 inference complete.")


# --- Step 6: Permutation ---
def _run_cv_fold_loop(X_brain, Y, X_cov, active_covs, config, seed, apriori_map=None, moderator=None):
    """Run full nested CV fold loop and return R²/AUC on concatenated outer-fold predictions.

    Shared helper used by _run_perm_task and _run_block_perm_task.
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

        n_moderator_cols = 0
        moderator_coded_tr = None
        moderator_coded_te = None
        interaction_tr = None
        interaction_te = None
        n_reduced = X_brain_tr_red.shape[1]
        if moderator is not None:
            mod_coded, _ = _code_moderator(moderator['series'], moderator['type'], tr)
            moderator_coded_tr = mod_coded.iloc[tr].reset_index(drop=True)
            moderator_coded_te = mod_coded.iloc[te].reset_index(drop=True)
            n_moderator_cols = moderator_coded_tr.shape[1]
            interaction_tr = _construct_interactions(X_brain_tr_red, moderator_coded_tr)
            interaction_te = _construct_interactions(X_brain_te_red, moderator_coded_te)

        parts_tr = []
        parts_te = []
        if not X_cov.empty and not is_pre:
            X_cov_tr = X_cov.iloc[tr].reset_index(drop=True)
            X_cov_te = X_cov.iloc[te].reset_index(drop=True)
            parts_tr.append(X_cov_tr)
            parts_te.append(X_cov_te)
        if moderator_coded_tr is not None:
            parts_tr.append(moderator_coded_tr)
            parts_te.append(moderator_coded_te)
        parts_tr.append(X_brain_tr_red)
        parts_te.append(X_brain_te_red)
        if interaction_tr is not None:
            parts_tr.append(interaction_tr)
            parts_te.append(interaction_te)
        X_tr = pd.concat(parts_tr, axis=1)
        X_te = pd.concat(parts_te, axis=1)

        Y_tr, Y_te = Y.iloc[tr], Y.iloc[te]
        if is_pre and not X_cov.empty:
            Y_tr, Y_te = _local_residualize(X_cov, Y, tr, te)

        all_feats = list(X_tr.columns)
        n_covs = len(active_covs) if not is_pre else 0
        moderator_indices = list(range(n_covs, n_covs + n_moderator_cols)) if n_moderator_cols > 0 else None
        pipeline, param_dist, scoring, _ = create_model_and_param_dist(
            config, all_feats, active_covs if not is_pre else [], Y=Y, moderator_indices=moderator_indices
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


def _run_perm_task(X_brain, Y, X_cov, active_covs, config, seed, apriori_map=None, moderator=None):
    """
    Single permutation iteration: shuffle Y, run full nested CV.
    ConvergenceWarnings suppressed per-task since permuted data frequently fails to converge.
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    rs = np.random.RandomState(seed)
    Y_shuf = sklearn_shuffle(Y, random_state=rs)
    return _run_cv_fold_loop(X_brain, Y_shuf, X_cov, active_covs, config, seed, apriori_map, moderator)


def run_permutation_test(config, X_brain, Y, weights, X_cov, active_covs, actual, job_id, n_jobs, is_worker, apriori_map=None, moderator=None):
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
        delayed(_run_perm_task)(X_brain, Y, X_cov, active_covs, config, s, apriori_map, moderator)
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


# --- Step 7: Selection Frequency ---

def _get_n_fold_bootstraps(config):
    """Return the per-fold bootstrap budget from config.

    Reads n_fold_bootstraps (preferred). Falls back to n_bootstraps (deprecated)
    with a warning. Defaults to 500 if neither is present.
    """
    stats = config.get('stats_params', {})
    if 'n_fold_bootstraps' in stats:
        return int(stats['n_fold_bootstraps'])
    if 'n_bootstraps' in stats:
        logging.warning(
            "config.stats_params.n_bootstraps is deprecated. "
            "Please rename to n_fold_bootstraps (semantics: per-fold budget; "
            "total iterations = K x n_fold_bootstraps)."
        )
        return int(stats['n_bootstraps'])
    logging.warning("Neither n_fold_bootstraps nor n_bootstraps found in config; defaulting to 500.")
    return 500



def run_selection_frequency(config, X_brain, Y, weights, X_cov, active_covs, fold_models, apriori_map=None, moderator=None):
    """Fold-wise bootstrap selection frequency: a descriptive measure of feature inclusion robustness.

    Per-iteration re-reduction: each subsample iteration fits a fresh reducer clone on
    the subsampled brain features, fits the model in reduced space with fold-specific
    best_params (from each fold's RandomizedSearchCV), back-projects selection indicators
    to the invariant original feature space using _backproject_coef_original_space.

    Selection iterations are distributed evenly across all K folds (each fold gets
    max(n_fold_bootstraps, 50) iterations). This spreads sampling and tuning variance
    across all folds rather than concentrating it in a single model.
    Ensemble selection probability is the mean across all K fold-level indicators.

    Aggregation of selection indicators in original feature space is meaningful across
    iterations with different reducer fits (Meinshausen & Bühlmann, 2010, JRSS-B).
    No significance threshold is applied; selection frequency is purely descriptive.

    moderator : dict or None
        Moderator specification with keys 'series' (pd.Series) and 'type' (str).
        When not None, interaction terms are constructed per-subsample and selection
        frequency is reported separately for main and interaction effects.
    """
    logging.info('--- Selection Frequency (fold-wise, descriptive) ---')
    n_fold_bootstraps = _get_n_fold_bootstraps(config)
    n_cores = config['n_cores']
    is_pre = config['covariate_method'] == 'pre_regress'
    original_feature_names = list(X_brain.columns)
    red_method = config['feature_reduction_method']
    out_feat_names = original_feature_names  # always report in original feature space

    n_per_fold_repeat = max(n_fold_bootstraps, 50)
    logging.info(
        f"Selection frequency: {len(fold_models)} folds x {n_per_fold_repeat} iterations/fold = "
        f"{len(fold_models) * n_per_fold_repeat} total subsample iterations."
    )

    rng_master = np.random.RandomState(43)

    full_sample_levels = None
    if moderator is not None and moderator['type'] == 'nominal':
        full_sample_levels = sorted(moderator['series'].unique().tolist())

    def _subsample_iter(fm, seed):
        """Single 50% subsample iteration using fold-specific best_params."""
        rng_sub = np.random.RandomState(seed)
        idx = rng_sub.choice(len(Y), len(Y) // 2, replace=False)

        # Use fold-specific reducer template (clone from fitted reducer or from factory)
        if fm['reducer'] is not None:
            reducer_sub = clone(fm['reducer'])
            reducer_sub.feature_names_in_ = original_feature_names
        else:
            reducer_sub = None

        # Resample brain features, fit fresh reducer clone (per-iteration re-reduction)
        X_brain_sub = X_brain.iloc[idx].reset_index(drop=True)
        if reducer_sub is not None:
            reducer_sub.fit(X_brain_sub)
            X_brain_sub_red = reducer_sub.transform(X_brain_sub)
        else:
            X_brain_sub_red = X_brain_sub

        n_moderator_cols_sub = 0
        moderator_coded_sub = None
        interaction_sub = None
        n_reduced_sub = X_brain_sub_red.shape[1]
        if moderator is not None:
            mod_coded_sub, _ = _code_moderator(moderator['series'], moderator['type'], idx,
                                                levels_override=full_sample_levels)
            moderator_coded_sub = mod_coded_sub.iloc[idx].reset_index(drop=True)
            n_moderator_cols_sub = moderator_coded_sub.shape[1]
            interaction_sub = _construct_interactions(X_brain_sub_red, moderator_coded_sub)

        # Recombine with covariates (incorporate method)
        parts_sub = []
        if not X_cov.empty and not is_pre:
            X_cov_sub = X_cov.iloc[idx].reset_index(drop=True)
            parts_sub.append(X_cov_sub)
        if moderator_coded_sub is not None:
            parts_sub.append(moderator_coded_sub)
        parts_sub.append(X_brain_sub_red)
        if interaction_sub is not None:
            parts_sub.append(interaction_sub)
        X_sub = pd.concat(parts_sub, axis=1)

        Y_arr = Y.values
        Y_sub = Y_arr[idx]
        if is_pre and not X_cov.empty:
            X_cov_sub_pr = X_cov.iloc[idx].reset_index(drop=True)
            lr_sub = LinearRegression().fit(X_cov_sub_pr, Y_sub)
            Y_sub = Y_sub - lr_sub.predict(X_cov_sub_pr)

        all_feats_sub = list(X_sub.columns)
        is_mt = _is_multitask(config, Y)
        n_covs_sub = len(active_covs) if not is_pre else 0
        moderator_indices_sub = list(range(n_covs_sub, n_covs_sub + n_moderator_cols_sub)) if n_moderator_cols_sub > 0 else None
        m, _, _, _ = create_model_and_param_dist(
            config, all_feats_sub,
            active_covs if not is_pre else [],
            Y=Y, weights=weights if is_mt else None,
            moderator_indices=moderator_indices_sub
        )
        m.set_params(**fm['best_params'])

        if is_mt and weights is not None:
            wt = m.named_steps['weight_transformer']
            wt.set_weights(weights.values[idx])
            Y_sub = wt.transform_y(Y_sub)
        elif weights is not None and not isinstance(m.named_steps['model'], MultiTaskElasticNet):
            m.fit(X_sub, Y_sub, model__sample_weight=weights.values[idx])
            c = _squeeze_binary_coef(m.named_steps['model'].coef_)
            c_main, c_interaction = _strip_protected(c, n_covs_sub, n_moderator_cols_sub, n_reduced_sub)
            c_orig_main = _backproject_coef_original_space(c_main, reducer_sub, original_feature_names)
            main_indicator = (c_orig_main != 0).astype(int)
            interaction_indicator = None
            if c_interaction is not None:
                if n_moderator_cols_sub > 1:
                    if c_interaction.ndim == 1:
                        reshaped_sub = c_interaction.reshape(n_moderator_cols_sub, n_reduced_sub)
                        interaction_indicator = np.stack([
                            (_backproject_coef_original_space(reshaped_sub[j], reducer_sub, original_feature_names) != 0).astype(int)
                            for j in range(n_moderator_cols_sub)
                        ], axis=0)
                    else:
                        K_class_sub = c_interaction.shape[0]
                        reshaped_sub = c_interaction.reshape(K_class_sub, n_moderator_cols_sub, n_reduced_sub)
                        interaction_indicator = np.array([[
                            (_backproject_coef_original_space(reshaped_sub[k, j], reducer_sub, original_feature_names) != 0).astype(int)
                            for j in range(n_moderator_cols_sub)
                        ] for k in range(K_class_sub)])
                else:
                    interaction_indicator = (_backproject_coef_original_space(c_interaction, reducer_sub, original_feature_names) != 0).astype(int)
            return main_indicator, interaction_indicator

        m.fit(X_sub, Y_sub)

        c = _squeeze_binary_coef(m.named_steps['model'].coef_)
        c_main, c_interaction = _strip_protected(c, n_covs_sub, n_moderator_cols_sub, n_reduced_sub)
        c_orig_main = _backproject_coef_original_space(c_main, reducer_sub, original_feature_names)
        main_indicator = (c_orig_main != 0).astype(int)
        interaction_indicator = None
        if c_interaction is not None:
            if n_moderator_cols_sub > 1:
                if c_interaction.ndim == 1:
                    reshaped_sub = c_interaction.reshape(n_moderator_cols_sub, n_reduced_sub)
                    interaction_indicator = np.stack([
                        (_backproject_coef_original_space(reshaped_sub[j], reducer_sub, original_feature_names) != 0).astype(int)
                        for j in range(n_moderator_cols_sub)
                    ], axis=0)
                else:
                    K_class_sub = c_interaction.shape[0]
                    reshaped_sub = c_interaction.reshape(K_class_sub, n_moderator_cols_sub, n_reduced_sub)
                    interaction_indicator = np.array([[
                        (_backproject_coef_original_space(reshaped_sub[k, j], reducer_sub, original_feature_names) != 0).astype(int)
                        for j in range(n_moderator_cols_sub)
                    ] for k in range(K_class_sub)])
            else:
                interaction_indicator = (_backproject_coef_original_space(c_interaction, reducer_sub, original_feature_names) != 0).astype(int)
        return main_indicator, interaction_indicator

    # Pre-generate all (fold_model, seed) tasks in fold-sequential order to preserve
    # reproducibility. A single flat Parallel dispatch reduces joblib pool creation
    # from K launches to 1, yielding ~5-15% wall-time reduction for typical configs.
    task_list = [
        (fm, s)
        for fm in fold_models
        for s in rng_master.randint(0, int(1e9), n_per_fold_repeat)
    ]
    flat_results = Parallel(n_jobs=n_cores)(
        delayed(_subsample_iter)(fm, s) for fm, s in task_list
    )
    all_results = [r for r in flat_results if r is not None]
    if not all_results:
        logging.warning("Selection frequency: all iterations failed. Skipping output.")
        return
    all_main_results = [r[0] for r in all_results]
    sample_res = all_main_results[0]
    is_multi = sample_res.ndim == 2  # True for multi-task regression or multi-class classification
    out_dir = config['paths']['output_dir']

    if not is_multi:
        # Single-output: results is a list of (P,) arrays; ensemble mean = selection probability
        main_df = pd.DataFrame({
            'feature': out_feat_names,
            'selection_probability': np.mean(all_main_results, axis=0)
        })
        if moderator is not None:
            main_df['effect_type'] = 'main'
            main_df['contrast'] = 'main'
        main_df.to_csv(
            os.path.join(out_dir, 'report_selection_frequency.csv'), index=False
        )
        if moderator is not None:
            interaction_results = [r[1] for r in all_results if r[1] is not None]
            if interaction_results:
                n_mod_cols_report = fold_models[0]['n_moderator_cols']
                out_path_sel = os.path.join(out_dir, 'report_selection_frequency.csv')
                if n_mod_cols_report <= 1:
                    interaction_mean = np.mean(interaction_results, axis=0)  # (P,)
                    interaction_df = pd.DataFrame({
                        'feature': out_feat_names,
                        'selection_probability': interaction_mean,
                        'effect_type': 'interaction',
                        'contrast': 'moderator',
                    })
                    interaction_df.to_csv(out_path_sel, mode='a', header=False, index=False)
                else:
                    interaction_stack = np.stack(interaction_results, axis=0)  # (n_iter, n_mod_cols, P)
                    interaction_mean_stack = interaction_stack.mean(axis=0)  # (n_mod_cols, P)
                    for j in range(n_mod_cols_report):
                        contrast_df = pd.DataFrame({
                            'feature': out_feat_names,
                            'selection_probability': interaction_mean_stack[j],
                            'effect_type': 'interaction',
                            'contrast': f'contrast_{j+1}',
                        })
                        contrast_df.to_csv(out_path_sel, mode='a', header=False, index=False)
    else:
        sel_array = np.stack(all_main_results, axis=0)  # (n_iter, K, P)
        task_labels = _get_task_labels(Y, config)
        for k, lbl in enumerate(task_labels):
            out_dir_k = os.path.join(out_dir, f'task_{lbl}')
            os.makedirs(out_dir_k, exist_ok=True)
            main_df_k = pd.DataFrame({
                'feature': out_feat_names,
                'selection_probability': sel_array[:, k, :].mean(axis=0)
            })
            if moderator is not None:
                main_df_k['effect_type'] = 'main'
                main_df_k['contrast'] = 'main'
            main_df_k.to_csv(os.path.join(out_dir_k, 'report_selection_frequency.csv'), index=False)

        union_sel = (sel_array.sum(axis=1) > 0).astype(int)
        main_union_df = pd.DataFrame({
            'feature': out_feat_names,
            'selection_probability': union_sel.mean(axis=0)
        })
        if moderator is not None:
            main_union_df['effect_type'] = 'main'
            main_union_df['contrast'] = 'main'
        main_union_df.to_csv(os.path.join(out_dir, 'report_selection_frequency.csv'), index=False)

        if moderator is not None:
            interaction_results = [r[1] for r in all_results if r[1] is not None]
            if interaction_results:
                n_mod_cols_report = fold_models[0]['n_moderator_cols']

                if n_mod_cols_report <= 1:
                    interaction_array = np.stack(interaction_results, axis=0)  # (n_iter, T, P)
                    for k, lbl in enumerate(task_labels):
                        out_dir_k = os.path.join(out_dir, f'task_{lbl}')
                        interaction_df_k = pd.DataFrame({
                            'feature': out_feat_names,
                            'selection_probability': interaction_array[:, k, :].mean(axis=0),
                            'effect_type': 'interaction',
                            'contrast': 'moderator',
                        })
                        interaction_df_k.to_csv(
                            os.path.join(out_dir_k, 'report_selection_frequency.csv'),
                            mode='a', header=False, index=False
                        )
                    interaction_union = (interaction_array.sum(axis=1) > 0).astype(int)  # (n_iter, P)
                    pd.DataFrame({
                        'feature': out_feat_names,
                        'selection_probability': interaction_union.mean(axis=0),
                        'effect_type': 'interaction',
                        'contrast': 'union',
                    }).to_csv(
                        os.path.join(out_dir, 'report_selection_frequency.csv'),
                        mode='a', header=False, index=False
                    )
                else:
                    interaction_array = np.stack(interaction_results, axis=0)  # (n_iter, T, n_mod_cols, P)
                    for k, lbl in enumerate(task_labels):
                        out_dir_k = os.path.join(out_dir, f'task_{lbl}')
                        interaction_k = interaction_array[:, k, :, :]  # (n_iter, n_mod_cols, P)
                        interaction_mean_k = interaction_k.mean(axis=0)  # (n_mod_cols, P)
                        for j in range(n_mod_cols_report):
                            contrast_df = pd.DataFrame({
                                'feature': out_feat_names,
                                'selection_probability': interaction_mean_k[j],
                                'effect_type': 'interaction',
                                'contrast': f'contrast_{j+1}',
                            })
                            contrast_df.to_csv(
                                os.path.join(out_dir_k, 'report_selection_frequency.csv'),
                                mode='a', header=False, index=False
                            )
                    n_iter = interaction_array.shape[0]
                    T_tasks = interaction_array.shape[1]
                    flat = interaction_array.reshape(n_iter, T_tasks * n_mod_cols_report, -1)  # (n_iter, T*n_mod_cols, P)
                    interaction_union = (flat.sum(axis=1) > 0).astype(int)  # (n_iter, P)
                    pd.DataFrame({
                        'feature': out_feat_names,
                        'selection_probability': interaction_union.mean(axis=0),
                        'effect_type': 'interaction',
                        'contrast': 'union',
                    }).to_csv(
                        os.path.join(out_dir, 'report_selection_frequency.csv'),
                        mode='a', header=False, index=False
                    )


# --- Step 8: Bootstrap & Reporting ---
def _boot_task(X_brain, Y, weights, seed, config, best_params, reducer_template,
               apriori_map=None, X_cov=None, active_covs=None, moderator=None,
               full_sample_levels=None):
    """Single bootstrap iteration (per-iteration re-reduction + back-projection).

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
        Sample weights for weight-aware resampling.
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
    moderator : dict or None
        Moderator specification with keys 'series' (pd.Series) and 'type' (str).
        When not None, interaction terms are constructed post-reduction and
        appended to the feature matrix; moderator main-effect columns are also
        included. Passed through from run_bootstrap.

    Returns
    -------
    (coef_original_main, c_original_interaction, converged, firth_used) : tuple
        coef_original_main : ndarray, shape (P,) or (K_tasks, P) — main brain-feature
            coefficients in original feature space after back-projection.
        c_original_interaction : ndarray or None — interaction coefficients back-projected
            to original brain-feature space. Shape (P,) for a single moderator contrast,
            (n_moderator_cols, P) for K>2 nominal moderators (regression/binary), or
            (K_class, n_moderator_cols, P) for multi-class with K>2 nominal moderators.
            None when no moderator is specified.
        converged : bool — False if ConvergenceWarning was raised.
        firth_used : bool — True if Firth-penalized logistic regression was invoked
            as fallback within _partial_ridge_refit.
    None on failure.
    """
    rng_boot = np.random.RandomState(seed)
    is_pre = config['covariate_method'] == 'pre_regress'
    active_covs = active_covs or []
    original_feature_names = list(X_brain.columns)

    try:
        # Weight-aware bootstrap resampling
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

        # Interaction construction (post-reduction, per-iteration)
        n_moderator_cols = 0
        moderator_coded_boot = None
        interaction_boot = None
        n_reduced = X_brain_boot_red.shape[1]
        if moderator is not None:
            mod_coded, coding_info = _code_moderator(moderator['series'], moderator['type'], idx,
                                                      levels_override=full_sample_levels)
            moderator_coded_boot = mod_coded.iloc[idx].reset_index(drop=True)
            n_moderator_cols = moderator_coded_boot.shape[1]
            interaction_boot = _construct_interactions(X_brain_boot_red, moderator_coded_boot)

        # Recombine with covariates, moderator main effects, and interactions
        parts_boot = []
        if X_cov is not None and not X_cov.empty and not is_pre:
            X_cov_boot = X_cov.iloc[idx].reset_index(drop=True)
            parts_boot.append(X_cov_boot)
        if moderator_coded_boot is not None:
            parts_boot.append(moderator_coded_boot)
        parts_boot.append(X_brain_boot_red)
        if interaction_boot is not None:
            parts_boot.append(interaction_boot)
        X_boot = pd.concat(parts_boot, axis=1)

        # Outcome: residualize for pre_regress
        Y_arr = Y.values
        Y_boot = Y_arr[idx]
        if is_pre and X_cov is not None and not X_cov.empty:
            X_cov_boot_pr = X_cov.iloc[idx].reset_index(drop=True)
            lr_boot = LinearRegression().fit(X_cov_boot_pr, Y_boot)
            Y_boot = Y_boot - lr_boot.predict(X_cov_boot_pr)

        # Build pipeline from best_params (no re-tuning)
        all_feats_boot = list(X_boot.columns)
        n_covs_boot = len(active_covs) if not is_pre else 0
        is_mt = _is_multitask(config, Y)

        moderator_indices_boot = list(range(n_covs_boot, n_covs_boot + n_moderator_cols)) if n_moderator_cols > 0 else None
        pipeline_boot, _, _, _ = create_model_and_param_dist(
            config, all_feats_boot,
            active_covs if not is_pre else [],
            Y=Y if not hasattr(Y, 'values') else (pd.DataFrame(Y_boot, columns=Y.columns) if Y.ndim > 1 else pd.Series(Y_boot)),
            weights=weights if is_mt else None,
            moderator_indices=moderator_indices_boot
        )
        # Set best hyperparameters directly (no RandomizedSearchCV)
        pipeline_boot.set_params(**best_params)

        # WeightTransformer: set weights and transform Y for multi-task + weights case
        Y_fit = Y_boot
        if is_mt and weights is not None:
            wt = pipeline_boot.named_steps['weight_transformer']
            wt.set_weights(weights.values[idx])
            Y_fit = wt.transform_y(Y_boot)

        converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if weights is not None and not is_mt:
                pipeline_boot.fit(X_boot, Y_fit,
                                  model__sample_weight=weights.values[idx])
            else:
                pipeline_boot.fit(X_boot, Y_fit)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

        # Binary classification: coef_ is (1, P) — squeeze to (P,) for single-output path.
        # Multi-class (K>=2): coef_ is (K, P) — preserve for per-class reporting.
        # Regression single-output: coef_ is (P,) — unchanged.
        # Multi-task regression: coef_ is (K, P) — preserved.
        firth_used = False
        bootstrap_ci_method = config.get('stats_params', {}).get('bootstrap_ci_method', 'partial_ridge')
        if bootstrap_ci_method == 'partial_ridge':
            c_raw = _squeeze_binary_coef(pipeline_boot.named_steps['model'].coef_)
            selected_mask = np.any(np.abs(c_raw) > 1e-10, axis=0) if c_raw.ndim == 2 else np.abs(c_raw) > 1e-10
            X_boot_transformed = X_boot
            for _, step in pipeline_boot.steps[:-1]:
                X_boot_transformed = step.transform(X_boot_transformed)
            X_boot_arr = X_boot_transformed.values if hasattr(X_boot_transformed, 'values') else np.asarray(X_boot_transformed)
            w_boot_arr = weights.values[idx] if (weights is not None and not is_mt) else None
            Y_for_pr = Y_fit if is_mt else Y_boot
            c, firth_used = _partial_ridge_refit(
                X_boot_arr, Y_for_pr, selected_mask, config,
                weights=w_boot_arr, n_covs=n_covs_boot, n_moderator_cols=n_moderator_cols
            )
        else:
            c = _squeeze_binary_coef(pipeline_boot.named_steps['model'].coef_)

        # Back-project main and interaction coefficient blocks to original feature space.
        # _strip_protected removes covariate and moderator main-effect columns, then
        # separates the main brain-feature block from the interaction block.
        c_main, c_interaction = _strip_protected(c, n_covs_boot, n_moderator_cols, n_reduced)
        c_original_main = _backproject_coef_original_space(c_main, reducer_boot, original_feature_names)
        c_original_interaction = None
        if c_interaction is not None:
            if n_moderator_cols > 1:
                if c_interaction.ndim == 1:
                    reshaped = c_interaction.reshape(n_moderator_cols, n_reduced)
                    bp_blocks = [_backproject_coef_original_space(reshaped[j], reducer_boot, original_feature_names) for j in range(n_moderator_cols)]
                    c_original_interaction = np.stack(bp_blocks, axis=0)
                else:
                    K_class = c_interaction.shape[0]
                    reshaped = c_interaction.reshape(K_class, n_moderator_cols, n_reduced)
                    bp_blocks = [[_backproject_coef_original_space(reshaped[k, j], reducer_boot, original_feature_names) for j in range(n_moderator_cols)] for k in range(K_class)]
                    c_original_interaction = np.array(bp_blocks)
            else:
                c_original_interaction = _backproject_coef_original_space(c_interaction, reducer_boot, original_feature_names)

        return c_original_main, c_original_interaction, converged, firth_used
    except Exception as exc:
        logging.debug("Bootstrap iteration failed: %s", exc, exc_info=True)
        return None


def calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model, report_df, level, X_brain_raw=None,
                                 moderator=None, effect_type=None):
    """Compute subject-level feature-vs-outcome data for visualization and write to CSV.

    For each significant feature/component in report_df, computes the partial
    association between that feature and the outcome after partialling out the
    linear contribution of all other features. Results are written to
    report_{level}_plotting.csv (main effects) or
    report_{level}_interaction_plotting.csv (interaction effects).

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    X_full : pd.DataFrame
        Full feature matrix (covariates + moderator + reduced brain + interactions).
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
    moderator : dict or None
        When provided, dict with 'series' (pd.Series), 'type' (str), 'K' (int).
    effect_type : str or None
        None (no moderator), 'main' (main-effect partialling), or 'interaction'
        (interaction-effect partialling).
    """
    analysis_type = config['analysis_type']
    out_dir = config['paths']['output_dir']
    mask = report_df['is_significant']
    candidates = report_df[mask]
    if candidates.empty:
        return
    coeffs = best_model.named_steps['model'].coef_
    intercept = getattr(best_model.named_steps['model'], 'intercept_', 0.0)
    if coeffs.ndim > 1:
        coeffs = coeffs.mean(axis=0)
    X_scaled = pd.DataFrame(best_model.named_steps['scaler'].transform(X_full), columns=X_full.columns, index=X_full.index)
    linear_pred_full = X_scaled.dot(coeffs) + (intercept.mean() if isinstance(intercept, np.ndarray) else intercept)
    feat_col = 'component_id' if 'component_id' in report_df.columns else (
        'cluster_id' if 'cluster_id' in report_df.columns else 'feature'
    )
    if Y.ndim > 1:
        logging.info(
            "calculate_visualization_data: skipped for multi-output Y "
            f"(shape {Y.shape}); visualization is single-output only."
        )
        return
    if effect_type == 'interaction' and moderator is not None:
        mod_coded, _ = _code_moderator(
            moderator['series'], moderator['type'],
            np.arange(len(moderator['series']))
        )
        if mod_coded.shape[1] > 1:
            logging.info(
                "calculate_visualization_data: skipped interaction "
                "visualization for K>2 nominal moderator "
                f"({mod_coded.shape[1]+1} levels). Per-contrast "
                "interaction effects are reported in "
                "report_fold_bootstrap_ci.csv."
            )
            return
    mod_raw = moderator['series'].values if moderator is not None else None
    mod_coded_vals = None
    if effect_type == 'interaction' and moderator is not None:
        mod_coded, _ = _code_moderator(
            moderator['series'], moderator['type'],
            np.arange(len(moderator['series']))
        )
        mod_coded_vals = mod_coded.values.flatten()
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
        if effect_type == 'interaction' and mod_coded_vals is not None:
            lin_contrib = f_weight * f_val_scaled * mod_coded_vals
        else:
            lin_contrib = f_weight * f_val_scaled
        cov_score = linear_pred_full - lin_contrib
        y_val = (Y_arr - cov_score.values) if analysis_type == 'regression' else expit(lin_contrib + cov_score.mean())
        for i in range(len(Y_arr)):
            record = {
                'subject_id': subject_ids_arr[i],
                'outcome_raw': Y_arr[i],
                'feature_name': f_name,
                'y_axis_value': y_val[i] if hasattr(y_val, '__getitem__') else y_val.iloc[i]
            }
            if mod_raw is not None:
                record['moderator_value'] = mod_raw[i]
            plot_data.append(record)
    if plot_data:
        suffix = 'interaction_plotting' if effect_type == 'interaction' else 'plotting'
        pd.DataFrame(plot_data).to_csv(os.path.join(out_dir, f'report_{level}_{suffix}.csv'), index=False)


def _build_individual_report_df(all_feats, stats, feat_col='feature', effect_type=None, contrast=None):
    """Build the standard individual-level importance DataFrame from shared statistics.

    Used by all four report branches to avoid duplicating the 8-column construction.
    Returns the DataFrame with FDR columns already applied.

    Parameters
    ----------
    all_feats : list of str
        Feature names.
    stats : dict
        Statistics dict from _compute_importance_preamble.
    feat_col : str
        Column name for the feature column.
    effect_type : str or None
        When moderator is active, the effect type label ('main' or 'interaction').
        If None, backward-compatible behavior: no column added.
    contrast : str or None
        When moderator is active, the contrast label. If None, no column added.
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
    if effect_type is not None:
        df['effect_type'] = effect_type
        df['contrast'] = contrast
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



def _compute_importance_preamble(df_coef, all_feats, feat_std_map, config):
    """
    Compute shared statistics for all reduction methods: std/raw means, CIs, pd, is_significant.

    Notes
    -----
    With per-iteration re-reduction (conditional bootstrap, Efron & Tibshirani, 1993,
    Ch. 13), df_coef reaching this function is always brain-only: covariate coefficients
    are stripped upstream in run_nested_cv (coef_original), _boot_task (bootstrap
    coefficients), and _subsample_iter (selection-frequency indicators). No penalty-weight
    adjustment is therefore applied here; raw coefficients are recovered by dividing
    standardised coefficients by the aligned feature standard deviations directly.
    """
    out_dir = config['paths']['output_dir']
    alpha = 1.0 - config['stats_params']['ci_level']
    std_means = df_coef.mean()
    std_ci_low = df_coef.quantile(alpha / 2)
    std_ci_high = df_coef.quantile(1 - alpha / 2)

    # Align feat_std_map to df_coef columns (may differ after per-iteration back-projection)
    feat_std_map_aligned = feat_std_map.reindex(df_coef.columns).fillna(1.0)

    raw_means = np.divide(std_means, feat_std_map_aligned, out=np.zeros_like(std_means), where=feat_std_map_aligned != 0)
    raw_ci_low = np.divide(std_ci_low, feat_std_map_aligned, out=np.zeros_like(std_ci_low), where=feat_std_map_aligned != 0)
    raw_ci_high = np.divide(std_ci_high, feat_std_map_aligned, out=np.zeros_like(std_ci_high), where=feat_std_map_aligned != 0)

    is_sig = (std_ci_low > 0) | (std_ci_high < 0)
    pd_val = pd.concat([(df_coef > 0).mean(), (df_coef < 0).mean()], axis=1).max(axis=1)

    return dict(
        std_means=std_means, std_ci_low=std_ci_low, std_ci_high=std_ci_high,
        raw_means=raw_means, raw_ci_low=raw_ci_low, raw_ci_high=raw_ci_high,
        is_sig=is_sig, pd_val=pd_val, out_dir=out_dir, alpha=alpha
    )


def _report_apriori(df_coef, all_feats, stats, config, active_covs,
                    reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model,
                    moderator=None, interaction_report_df=None):
    """Apriori: feature-level report (report_feature_importance.csv) + network-level aggregation via apriori cluster map.

    With per-iteration re-reduction, df_coef is in original brain feature space. Network-level
    statistics are produced by aggregating individual feature bootstrap distributions
    to cluster level using the apriori map from reducer_full (mean coefficient per cluster).
    """
    out_dir = stats['out_dir']
    alpha = stats['alpha']

    indiv_df = _build_individual_report_df(all_feats, stats)
    indiv_df.to_csv(os.path.join(out_dir, 'report_feature_importance.csv'), index=False)

    eff_type = 'main' if moderator is not None else None
    try:
        if reducer_full is not None and hasattr(reducer_full, 'get_loadings'):
            loadings = pd.DataFrame(reducer_full.get_loadings())
            cluster_rows = []
            for c_name in loadings['cluster'].unique():
                cluster_feats = loadings[loadings['cluster'] == c_name]['feature'].tolist()
                cluster_feats_in = [f for f in cluster_feats if f in df_coef.columns]
                if not cluster_feats_in:
                    continue
                cluster_boot = df_coef[cluster_feats_in].mean(axis=1)
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
                                             net_rep, 'cluster', X_brain,
                                             moderator=moderator, effect_type=eff_type)
    except (KeyError, ValueError, IndexError) as exc:
        logging.warning(f"Apriori network-level aggregation failed: {exc}")
        logging.debug("Back-projection traceback:", exc_info=True)

    if interaction_report_df is not None:
        interaction_report_df.to_csv(
            os.path.join(out_dir, 'report_interaction_importance.csv'), index=False
        )


def _report_standard(df_coef, all_feats, stats, config, active_covs,
                     reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model,
                     moderator=None, interaction_report_df=None):
    """Standard importance report for none, cluster_pca, and ica reduction methods.

    Writes report_feature_importance.csv for all reduction methods and calls
    calculate_visualization_data. For ica, also saves the full-data mixing matrix
    for transparency. For cluster_pca, renames 'feature' to 'cluster_id' in the
    visualization call to match the expected column name.

    With per-iteration re-reduction, df_coef is already in original brain feature space
    after back-projection in each bootstrap iteration, so report directly.
    """
    out_dir = stats['out_dir']
    red_method = config['feature_reduction_method']

    if red_method == 'ica' and reducer_full is not None and hasattr(reducer_full, 'mixing_unnorm_'):
        mixing_df = pd.DataFrame(
            reducer_full.mixing_unnorm_,
            index=reducer_full.feature_names_in_,
            columns=reducer_full.ic_names_
        )
        mixing_df.to_csv(os.path.join(out_dir, 'ica_mixing_matrix.csv'))

    indiv_df = _build_individual_report_df(all_feats, stats)
    indiv_df.to_csv(os.path.join(out_dir, 'report_feature_importance.csv'), index=False)

    eff_type = 'main' if moderator is not None else None
    if red_method == 'none':
        calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model,
                                     indiv_df, 'individual', None,
                                     moderator=moderator, effect_type=eff_type)
    else:
        vis_df = indiv_df.rename(columns={'feature': 'cluster_id'}) if red_method == 'cluster_pca' else indiv_df
        calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model,
                                     vis_df, 'individual', X_brain,
                                     moderator=moderator, effect_type=eff_type)

    if interaction_report_df is not None:
        interaction_report_df.to_csv(
            os.path.join(out_dir, 'report_interaction_importance.csv'), index=False
        )
        calculate_visualization_data(
            config, X_full, Y, weights, subject_ids, best_model,
            interaction_report_df, 'individual',
            X_brain if red_method != 'none' else None,
            moderator=moderator, effect_type='interaction'
        )


def _compute_importance_report(df_coef, all_feats, feat_std_map, config, active_covs, reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model,
                               moderator=None, interaction_report_df=None):
    """
    Dispatcher: compute shared statistics, then delegate to reduction-specific branch.
    Applies BH-FDR correction at q=0.05.
    """
    stats = _compute_importance_preamble(df_coef, all_feats, feat_std_map, config)
    red_method = config['feature_reduction_method']
    branch_args = (df_coef, all_feats, stats, config, active_covs,
                   reducer_full, X_brain, X_full, Y, weights, subject_ids, best_model)
    if red_method == 'apriori':
        _report_apriori(*branch_args, moderator=moderator,
                        interaction_report_df=interaction_report_df)
    else:
        _report_standard(*branch_args, moderator=moderator,
                         interaction_report_df=interaction_report_df)


def _reconstruct_x_full(fold_models, X_brain, X_cov, config, active_covs, moderator=None):
    """Reconstruct a representative full-data feature matrix for visualization.

    Uses the fold-0 model's reducer (if any) to transform the full X_brain, then
    assembles the canonical column layout [covariates, moderator_main, brain_reduced,
    interactions]. This is a descriptive approximation — the fold-0 reducer was fit
    on the fold-0 training set only, not the full dataset. When moderator is provided,
    full-sample indices are used for coding (consistent with the approximation).

    Parameters
    ----------
    moderator : dict or None
        If provided, a dict with keys 'series' (pd.Series of moderator values) and
        'type' (str, one of 'continuous', 'nominal'). When not None, moderator
        main-effect columns and brain-by-moderator interaction columns are added.

    Returns
    -------
    X_full_repr : pd.DataFrame
        Representative assembled feature matrix in canonical column order.
    fm0 : dict
        fold_models[0] record, for access to representative pipeline and feat_std_map.
    """
    fm0 = fold_models[0]
    reducer0 = fm0['reducer']
    is_pre = config['covariate_method'] == 'pre_regress'

    if reducer0 is not None:
        X_brain_red = reducer0.transform(X_brain)
    else:
        X_brain_red = X_brain

    moderator_coded = None
    interactions = None
    if moderator is not None:
        mod_coded, _ = _code_moderator(
            moderator['series'], moderator['type'],
            np.arange(len(moderator['series']))
        )
        moderator_coded = mod_coded.reset_index(drop=True)
        interactions = _construct_interactions(
            X_brain_red.reset_index(drop=True), moderator_coded
        )

    parts = []
    if not X_cov.empty and not is_pre:
        parts.append(X_cov.reset_index(drop=True))
    if moderator_coded is not None:
        parts.append(moderator_coded)
    parts.append(X_brain_red.reset_index(drop=True))
    if interactions is not None:
        parts.append(interactions)
    X_full_repr = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    return X_full_repr, fm0


def _write_tier2_single(coef_pool, feature_names, out_dir, ci_level, effect_type=None, contrast=None):
    """Compute and write Tier 2 inference: pooled fold-wise bootstrap percentile CIs.

    Parameters
    ----------
    coef_pool : ndarray, shape (B_total, P)
        Pooled bootstrap coefficients across all K folds.
    feature_names : list of str, length P
        Original brain feature names.
    out_dir : str
        Output directory.
    ci_level : float
        Confidence level (e.g. 0.95).
    effect_type : str or None
        When moderator is active, the effect type label ('main' or 'interaction').
        If None, backward-compatible behavior: no column added, single fresh write.
    contrast : str or None
        When moderator is active, the contrast label (e.g. 'main', 'moderator',
        'L2_norm', 'contrast_1'). If None, backward-compatible behavior.

    Outputs
    -------
    report_fold_bootstrap_ci.csv
        boot_mean_coef, boot_ci_low, boot_ci_high, pd, p_value,
        is_significant, is_significant_fdr. When moderator is active, also
        effect_type and contrast columns.
    """
    alpha_b = 1.0 - ci_level
    boot_mean = coef_pool.mean(axis=0)
    boot_ci_low = np.percentile(coef_pool, 100 * alpha_b / 2, axis=0)
    boot_ci_high = np.percentile(coef_pool, 100 * (1 - alpha_b / 2), axis=0)
    B = coef_pool.shape[0]
    pd_val = np.maximum(
        (coef_pool > 0).sum(axis=0) / B,
        (coef_pool < 0).sum(axis=0) / B
    )
    p_value = np.clip(2 * (1 - pd_val), 0.0, 1.0)
    is_sig = (boot_ci_low > 0) | (boot_ci_high < 0)
    is_sig_fdr = _bh_fdr(p_value, q=0.05)

    df_result = pd.DataFrame({
        'feature': feature_names,
        'boot_mean_coef': boot_mean,
        'boot_ci_low': boot_ci_low,
        'boot_ci_high': boot_ci_high,
        'pd': pd_val,
        'p_value': p_value,
        'is_significant': is_sig,
        'is_significant_fdr': is_sig_fdr,
    })
    if effect_type is not None:
        df_result['effect_type'] = effect_type
        df_result['contrast'] = contrast

    out_path = os.path.join(out_dir, 'report_fold_bootstrap_ci.csv')
    if effect_type is not None and effect_type != 'main':
        df_result.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df_result.to_csv(out_path, index=False)


def run_bootstrap(config, X_brain, Y, weights, subject_ids, X_cov, active_covs, fold_models, apriori_map=None, moderator=None):
    """Run fold-wise pooled bootstrap importance estimation (Tier 2 inference).

    Bootstrap iterations are distributed evenly across all K folds (each fold gets
    max(n_fold_bootstraps, 50) iterations). Each iteration uses the fold-specific
    best_params, ensuring tuning variance propagates to the bootstrap distribution.
    Per-iteration re-reduction (conditional bootstrap, Efron & Tibshirani, 1993, Ch. 13):
    each iteration clones and refits the reducer on resampled brain features, fits the
    model in reduced space with fixed best_params, and back-projects coefficients to the
    invariant original feature space.

    All valid results are pooled across folds for Tier 2 percentile CIs
    (report_fold_bootstrap_ci.csv). The full pooled distribution is also passed to
    _compute_importance_report for the standard importance report (report_*_importance.csv).

    moderator : dict or None
        Moderator specification with keys 'series' (pd.Series) and 'type' (str).
        When not None and 'type' == 'nominal', full-sample levels are computed once
        and passed to _boot_task as levels_override to ensure consistent coding
        dimensions across bootstrap resamples (see _code_moderator).
    """
    logging.info("--- Bootstrap Importance (fold-wise, Tier 2) ---")
    n_fold_bootstraps = _get_n_fold_bootstraps(config)
    n_cores = config['n_cores']
    original_feature_names = list(X_brain.columns)
    red_method = config['feature_reduction_method']
    cov_method = config['covariate_method']

    n_per_fold_repeat = max(n_fold_bootstraps, 50)
    logging.info(
        f"Bootstrap: {len(fold_models)} folds x {n_per_fold_repeat} iterations/fold = "
        f"{len(fold_models) * n_per_fold_repeat} total bootstrap iterations."
    )

    X_cov_for_boot = X_cov if (cov_method != 'none' and X_cov is not None and not X_cov.empty) else None

    full_sample_levels = None
    if moderator is not None and moderator['type'] == 'nominal':
        full_sample_levels = sorted(moderator['series'].unique().tolist())

    rng_master = np.random.RandomState(42)

    # Pre-generate all (best_params, reducer_template, seed) tasks in fold-sequential order to
    # preserve reproducibility. A single flat Parallel dispatch reduces joblib pool creation
    # from K launches to 1, yielding ~5-15% wall-time reduction for typical configs.
    task_list = []
    for fm in fold_models:
        # Use fold-specific reducer template: clone from fitted reducer for type fidelity
        reducer_tmpl = clone(fm['reducer']) if fm['reducer'] is not None else None
        for s in rng_master.randint(0, int(1e9), n_per_fold_repeat):
            task_list.append((fm['best_params'], reducer_tmpl, s))

    all_res = Parallel(n_jobs=n_cores)(
        delayed(_boot_task)(
            X_brain, Y, weights, s, config, bp, reducer_tmpl,
            apriori_map=apriori_map,
            X_cov=X_cov_for_boot,
            active_covs=active_covs,
            moderator=moderator,
            full_sample_levels=full_sample_levels
        )
        for bp, reducer_tmpl, s in task_list
    )

    valid_res = [r for r in all_res if r is not None]
    n_failed = len(all_res) - len(valid_res)
    n_total = len(all_res)
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

    n_conv_warn = sum(1 for r in valid_res if not r[2])
    if n_conv_warn > 0:
        logging.warning(
            f"ConvergenceWarning in {n_conv_warn}/{len(valid_res)} bootstrap iterations."
        )

    n_firth = sum(1 for r in valid_res if r[3])
    if n_firth > 0:
        firth_pct = 100 * n_firth / len(valid_res)
        if firth_pct > 20.0:
            logging.warning(
                f"Firth fallback used in {n_firth}/{len(valid_res)} bootstrap iterations "
                f"({firth_pct:.1f}%). High fallback rate suggests data characteristics "
                f"(small n, high-dimensional features, class imbalance) cause frequent "
                f"near-separation in bootstrap resamples."
            )
        else:
            logging.info(f"Firth fallback used in {n_firth}/{len(valid_res)} iterations ({firth_pct:.1f}%).")

    # Representative pipeline and X_full for visualization (fold-0 approximation)
    # Documented as a descriptive approximation: fold-0 reducer fit on fold-0 training data.
    X_full_repr, fm0 = _reconstruct_x_full(fold_models, X_brain, X_cov, config, active_covs, moderator=moderator)
    best_model_repr = fm0['pipeline']
    reducer_full_repr = fm0['reducer']

    # feat_std_map for importance reporting (brain-only; _boot_task strips
    # protected columns before returning, so coef_matrix is always brain-space)
    brain_std = X_brain.std(ddof=1).replace(0, 1.0)
    feat_std_map_report = brain_std
    all_feats_report = original_feature_names
    df_coef_cols = original_feature_names

    # Save cluster/ICA descriptive outputs from fold-0 reducer (representative)
    if reducer_full_repr is not None and hasattr(reducer_full_repr, 'get_loadings'):
        pd.DataFrame(reducer_full_repr.get_loadings()).to_csv(
            os.path.join(config['paths']['output_dir'], 'cluster_loadings.csv'), index=False
        )

    # Detect multi-output from the first valid result
    sample_coef = valid_res[0][0]
    is_multi = sample_coef.ndim == 2  # True for multi-task regression or multi-class

    if not is_multi:
        coef_matrix = np.stack([r[0] for r in valid_res], axis=0)  # (B, P)
        df_coef = pd.DataFrame(coef_matrix, columns=df_coef_cols)

        has_moderator = moderator is not None and fold_models[0].get('coef_original_interaction') is not None
        coef_matrix_interaction = None
        n_mod_cols = 0
        if has_moderator:
            coef_matrix_interaction = np.stack([r[1] for r in valid_res], axis=0)
            n_mod_cols = fold_models[0]['n_moderator_cols']

        if config['stats_params'].get('save_distributions', True):
            save_kwargs = dict(
                coef_dist=coef_matrix,
                feature_names=np.array(df_coef_cols)
            )
            if has_moderator:
                save_kwargs['coef_dist_interaction'] = coef_matrix_interaction
                if n_mod_cols > 1:
                    save_kwargs['moderator_contrasts'] = np.array([f'contrast_{j+1}' for j in range(n_mod_cols)])
            np.savez_compressed(
                os.path.join(config['paths']['output_dir'], 'bootstrap_coef_distribution.npz'),
                **save_kwargs
            )

        interaction_report_df = None
        if has_moderator:
            if n_mod_cols <= 1:
                df_coef_interaction = pd.DataFrame(
                    coef_matrix_interaction, columns=df_coef_cols
                )
            else:
                l2_norms_int = np.sqrt(np.sum(
                    coef_matrix_interaction ** 2, axis=1
                ))
                df_coef_interaction = pd.DataFrame(
                    l2_norms_int, columns=df_coef_cols
                )
            int_stats = _compute_importance_preamble(
                df_coef_interaction, all_feats_report,
                feat_std_map_report, config
            )
            interaction_report_df = _build_individual_report_df(
                all_feats_report, int_stats
            )

        _compute_importance_report(
            df_coef, all_feats_report, feat_std_map_report, config, active_covs,
            reducer_full_repr, X_brain, X_full_repr, Y, weights, subject_ids, best_model_repr,
            moderator=moderator,
            interaction_report_df=interaction_report_df
        )

        _write_tier2_single(
            coef_matrix,
            all_feats_report,
            config['paths']['output_dir'],
            config['stats_params']['ci_level'],
            effect_type='main' if moderator is not None else None,
            contrast='main' if moderator is not None else None,
        )

        if has_moderator:
            if n_mod_cols <= 1:
                _write_tier2_single(
                    coef_matrix_interaction, all_feats_report, config['paths']['output_dir'], config['stats_params']['ci_level'],
                    effect_type='interaction', contrast='moderator',
                )
            else:
                l2_norms = np.sqrt(np.sum(coef_matrix_interaction ** 2, axis=1))  # (B, P)
                _write_tier2_single(
                    l2_norms, all_feats_report, config['paths']['output_dir'], config['stats_params']['ci_level'],
                    effect_type='interaction', contrast='L2_norm',
                )
                for j in range(n_mod_cols):
                    per_contrast = coef_matrix_interaction[:, j, :]  # (B, P)
                    _write_tier2_single(
                        per_contrast, all_feats_report, config['paths']['output_dir'], config['stats_params']['ci_level'],
                        effect_type='interaction', contrast=f'contrast_{j+1}',
                    )
    else:
        coef_array = np.stack([r[0] for r in valid_res], axis=0)  # (B, K, P)
        task_labels = _get_task_labels(Y, config)

        has_moderator = moderator is not None and fold_models[0].get('coef_original_interaction') is not None
        coef_array_interaction = None
        n_mod_cols = 0
        if has_moderator:
            coef_array_interaction = np.stack([r[1] for r in valid_res], axis=0)
            n_mod_cols = fold_models[0]['n_moderator_cols']

        if config['stats_params'].get('save_distributions', True):
            save_kwargs = dict(
                coef_dist=coef_array,
                feature_names=np.array(df_coef_cols),
                task_labels=np.array(task_labels)
            )
            if has_moderator:
                save_kwargs['coef_dist_interaction'] = coef_array_interaction
                if n_mod_cols > 1:
                    save_kwargs['moderator_contrasts'] = np.array([f'contrast_{j+1}' for j in range(n_mod_cols)])
            np.savez_compressed(
                os.path.join(config['paths']['output_dir'], 'bootstrap_coef_distribution.npz'),
                **save_kwargs
            )

        for k, lbl in enumerate(task_labels):
            out_dir_k = os.path.join(config['paths']['output_dir'], f'task_{lbl}')
            os.makedirs(out_dir_k, exist_ok=True)
            config_k = {**config, 'paths': {**config['paths'], 'output_dir': out_dir_k}}
            df_coef_k = pd.DataFrame(coef_array[:, k, :], columns=df_coef_cols)
            Y_k = Y.iloc[:, k] if hasattr(Y, 'iloc') and Y.ndim > 1 else Y

            interaction_report_df_k = None
            if has_moderator:
                if n_mod_cols <= 1:
                    df_coef_int_k = pd.DataFrame(
                        coef_array_interaction[:, k, :], columns=df_coef_cols
                    )
                else:
                    int_k = coef_array_interaction[:, k, :, :]
                    l2_k = np.sqrt(np.sum(int_k ** 2, axis=1))
                    df_coef_int_k = pd.DataFrame(l2_k, columns=df_coef_cols)
                int_stats_k = _compute_importance_preamble(
                    df_coef_int_k, all_feats_report,
                    feat_std_map_report, config_k
                )
                interaction_report_df_k = _build_individual_report_df(
                    all_feats_report, int_stats_k
                )

            _compute_importance_report(
                df_coef_k, all_feats_report, feat_std_map_report, config_k, active_covs,
                reducer_full_repr, X_brain, X_full_repr, Y_k, weights, subject_ids, best_model_repr,
                moderator=moderator,
                interaction_report_df=interaction_report_df_k
            )
            _write_tier2_single(
                coef_array[:, k, :],
                all_feats_report,
                out_dir_k,
                config['stats_params']['ci_level'],
                effect_type='main' if moderator is not None else None,
                contrast='main' if moderator is not None else None,
            )

        if has_moderator:
            for k, lbl in enumerate(task_labels):
                out_dir_k = os.path.join(config['paths']['output_dir'], f'task_{lbl}')
                if n_mod_cols <= 1:
                    _write_tier2_single(
                        coef_array_interaction[:, k, :],
                        all_feats_report,
                        out_dir_k,
                        config['stats_params']['ci_level'],
                        effect_type='interaction', contrast='moderator',
                    )
                else:
                    interaction_k = coef_array_interaction[:, k, :, :]  # (B, n_mod_cols, P)
                    l2_norms_k = np.sqrt(np.sum(interaction_k ** 2, axis=1))  # (B, P)
                    _write_tier2_single(
                        l2_norms_k, all_feats_report, out_dir_k, config['stats_params']['ci_level'],
                        effect_type='interaction', contrast='L2_norm',
                    )
                    for j in range(n_mod_cols):
                        per_contrast_k = interaction_k[:, j, :]  # (B, P)
                        _write_tier2_single(
                            per_contrast_k, all_feats_report, out_dir_k, config['stats_params']['ci_level'],
                            effect_type='interaction', contrast=f'contrast_{j+1}',
                        )


# --- Ensemble Prediction Utility ---

def predict_ensemble(fold_models, X_brain_new, X_cov_new, config, active_covs, moderator=None):
    """Predict on new data by averaging predictions across all K fold submodels.

    Each fold's reducer is applied to X_brain_new, the assembled feature matrix is
    passed through the fold's fitted pipeline, and predictions are averaged uniformly
    across all K folds. Uniform weighting is used: performance weighting is unstable
    at typical fMRI sample sizes where per-fold R² estimates have high variance
    (SE(R²) > signal for N/K ~ 20 test observations).

    This utility is not called from main(). It is exposed for downstream use after
    pipeline execution (e.g., prediction on held-out cohorts).

    Parameters
    ----------
    fold_models : list of dict
        fold_models from run_nested_cv.
    X_brain_new : pd.DataFrame
        New brain features, shape (N_new, P).
    X_cov_new : pd.DataFrame or None
        New covariate features, shape (N_new, P_cov).
    config : dict
        Pipeline configuration.
    active_covs : list of str
        Active covariate column names.
    moderator : dict or None
        When provided, must contain keys 'series' (pd.Series of moderator values for
        the new data, length N_new), 'type' (str), and 'K' (int). Required when the
        fold models were trained with a moderator.

    Returns
    -------
    y_pred_mean : ndarray, shape (N_new,) or (N_new, K_tasks)
        Mean prediction across all K fold submodels.
    y_pred_std : ndarray, shape (N_new,) or (N_new, K_tasks)
        Standard deviation of predictions across submodels (uncertainty estimate).

    Raises
    ------
    ValueError
        If fold models were trained with a moderator but moderator is None, or vice versa.
    """
    fm0 = fold_models[0]
    has_mod_in_model = fm0.get('n_moderator_cols', 0) > 0
    if has_mod_in_model and moderator is None:
        raise ValueError(
            "Fold models were trained with a moderator but no moderator was provided "
            "to predict_ensemble. Pass a moderator dict with keys 'series', 'type', 'K'."
        )
    if not has_mod_in_model and moderator is not None:
        raise ValueError(
            "Fold models were trained without a moderator but a moderator was provided "
            "to predict_ensemble."
        )

    is_pre = config['covariate_method'] == 'pre_regress'
    moderator_coded = None
    if moderator is not None:
        mod_coded, _ = _code_moderator(
            moderator['series'], moderator['type'],
            np.arange(len(moderator['series']))
        )
        moderator_coded = mod_coded.reset_index(drop=True)

    preds = []
    for fm in fold_models:
        reducer = fm['reducer']
        pipe = fm['pipeline']
        if reducer is not None:
            X_br_red = reducer.transform(X_brain_new)
        else:
            X_br_red = X_brain_new

        parts = []
        if (X_cov_new is not None and not X_cov_new.empty
                and not is_pre):
            parts.append(X_cov_new.reset_index(drop=True))
        if moderator_coded is not None:
            parts.append(moderator_coded)
        parts.append(X_br_red.reset_index(drop=True))
        if moderator_coded is not None:
            interactions = _construct_interactions(X_br_red.reset_index(drop=True), moderator_coded)
            parts.append(interactions)
        X_new = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
        preds.append(pipe.predict(X_new))
    preds_arr = np.array(preds)  # shape (K, N_new) or (K, N_new, K_tasks)
    y_pred_mean = preds_arr.mean(axis=0)
    y_pred_std = preds_arr.std(axis=0, ddof=1)
    return y_pred_mean, y_pred_std


# --- Step 9: Block Permutation ---
def _run_block_perm_task(X_brain, X_block_cols, Y, X_cov, active_covs, config, seed, apriori_map=None, moderator=None):
    """Single block permutation iteration: shuffle block column rows, run full nested CV.

    The null distribution uses the full nested CV score computed on the permuted
    brain feature matrix, consistent with the observed score from run_nested_cv.
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    rs = np.random.RandomState(seed)
    X_brain_perm = X_brain.copy()
    block_data = X_brain_perm[X_block_cols].values
    shuffled_idx = rs.permutation(len(block_data))
    X_brain_perm[X_block_cols] = block_data[shuffled_idx]
    return _run_cv_fold_loop(X_brain_perm, Y, X_cov, active_covs, config, seed, apriori_map, moderator)


def run_block_perms(config, X_brain, Y, weights, X_cov, active_covs, actual_score, apriori_map=None, moderator=None):
    """Run block-specific permutation tests to assess the incremental contribution of feature subsets.

    The observed score is the nested CV score passed as actual_score. The null distribution
    is constructed by running the full nested CV on versions of X_brain where the block
    columns are row-permuted. Number of permutations is set by n_block_permutations in config.
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
                X_brain, b_cols, Y, X_cov, active_covs, config, s, apriori_map, moderator
            )
            for s in seeds
        )
        p_value = (np.sum(np.array(null_scores) >= actual_score) + 1) / (len(null_scores) + 1)
        results.append({'block': label, 'observed_score': actual_score, 'p_value': p_value})
        logging.info(f"Block '{label}': observed={actual_score:.4f}, p={p_value:.4f}")

        # Save block permutation null distribution
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
    """Entry point for the fmri-elastic-net pipeline.

    Supports three execution modes controlled by --mode:
    - 'main' (default): runs nested CV, Tier 1 fold-wise ensemble inference,
      fold-wise selection frequency, fold-wise bootstrap importance (Tier 2),
      optional label-permutation test, and optional block permutation tests.
      New output files (relative to output_dir):
        report_fold_ensemble_importance.csv — Tier 1: fold-wise t-test CIs per feature
        report_fold_diagnostics.csv         — per-fold hyperparameter records
        report_fold_params_summary.csv      — mean ± SD across K folds
        report_fold_bootstrap_ci.csv        — Tier 2: pooled bootstrap percentile CIs
    - 'perm_worker': runs a SLURM array worker subset of permutations and writes
      perm_chunk_{job_id}.csv to the output directory.
    - 'aggregate': collects all perm_chunk_*.csv files, computes the final
      permutation p-value, and writes permutation_result.csv.

    Required argument: --config (path to YAML configuration file).
    """
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

    # Suppress known-harmless warnings that clutter stderr during normal operation.
    # The _boot_task catch_warnings(record=True) block still captures convergence
    # state: its inner simplefilter("always") overrides this filter within that scope.
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', message='overflow encountered in exp',
                            category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='Ill-conditioned matrix')

    # Validate required config parameters
    if 'n_random_search_iter' not in config.get('cv_params', {}):
        sys.exit("CRITICAL: config.cv_params.n_random_search_iter is required but absent. "
                 "Recommended range: 10-100. For a 2-parameter space (alpha, l1_ratio), 20-50 is typically sufficient.")

    data_cfg_check = config.get('data_cols', {})
    mod_col_check = data_cfg_check.get('moderator_col')
    mod_type_check = data_cfg_check.get('moderator_type')
    if (mod_col_check is None) != (mod_type_check is None):
        sys.exit(
            "CRITICAL: config.data_cols.moderator_col and moderator_type must both be null "
            "or both be specified; one is set while the other is null."
        )
    if mod_type_check is not None and mod_type_check not in ('continuous', 'nominal'):
        sys.exit(
            f"CRITICAL: config.data_cols.moderator_type must be 'continuous' or 'nominal', "
            f"got '{mod_type_check}'."
        )
    bootstrap_ci_method_check = config.get('stats_params', {}).get('bootstrap_ci_method', 'partial_ridge')
    if bootstrap_ci_method_check not in ('partial_ridge', 'percentile'):
        sys.exit(
            f"CRITICAL: config.stats_params.bootstrap_ci_method must be 'partial_ridge' or "
            f"'percentile', got '{bootstrap_ci_method_check}'."
        )
    if bootstrap_ci_method_check == 'percentile' and config.get('analysis_mode') == 'predict':
        logging.warning(
            "bootstrap_ci_method='percentile' with analysis_mode='predict': percentile CIs "
            "are known to be anti-conservative under L1 zero-inflation in high-sparsity regimes. "
            "'partial_ridge' (the default) is recommended for predict mode."
        )

    try:
        X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map, moderator = load_and_prep_data(config, out_dir)

        if args.mode == 'perm_worker':
            run_permutation_test(
                config, X_brain, Y, weights, X_cov, active_covs,
                0.0, args.job_id, args.n_jobs, True, apriori_map, moderator
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
            actual, fold_models = run_nested_cv(config, X_brain, Y, weights, X_cov, active_covs, apriori_map, moderator)
            # Store metric for perm worker label
            _, _, _, metric = create_model_and_param_dist(config, ['dummy'], [], Y=Y)
            config.setdefault('_runtime', {})['metric'] = metric

            # Tier 1 inference: fold-wise ensemble t-test (new Step 5)
            run_tier1_inference(config, fold_models, X_brain, Y, active_covs, moderator)

            run_selection_frequency(config, X_brain, Y, weights, X_cov, active_covs, fold_models, apriori_map, moderator)
            run_bootstrap(config, X_brain, Y, weights, subj_ids, X_cov, active_covs, fold_models, apriori_map, moderator)
            if not args.skip_main_perm:
                run_permutation_test(
                    config, X_brain, Y, weights, X_cov, active_covs,
                    actual, 0, 1, False, apriori_map, moderator
                )
            run_block_perms(config, X_brain, Y, weights, X_cov, active_covs, actual, apriori_map, moderator)

    except Exception:
        logging.error("PIPELINE FAILED", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
