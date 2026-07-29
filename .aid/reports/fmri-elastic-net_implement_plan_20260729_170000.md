<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-07-29T17:00:00-04:00" />
  <input_reports>
    <report path="fmri-elastic-net_cr_20260729_154100.md" mode="cr" key_items="6" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="F1">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Fix logistic Partial Ridge back-transformation direction. Change /= to *= at line 522.</description>
      <spec>
In _partial_ridge_refit, inside _refit_binary, line 522:

Change:
    c_row[s_idx] /= sqrt_n

To:
    c_row[s_idx] *= sqrt_n

No other changes. The Firth threshold at line 525 (|coef| > 10.0) remains unchanged per brainstorm T1 validation.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single operator change with mathematical proof; significance determinations unaffected (scale-invariant zero-crossing)</risk>
      <rollback>Revert *= back to /=</rollback>
    </change>
    <change id="C2" priority="P0" source_item="F2">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Fix multi-output selected_mask dimensionality in _boot_task. Collapse 2D mask to 1D via union across outputs.</description>
      <spec>
In _boot_task, line 2767:

Change:
    selected_mask = np.abs(c_raw) > 1e-10

To:
    selected_mask = np.any(np.abs(c_raw) > 1e-10, axis=0) if c_raw.ndim == 2 else np.abs(c_raw) > 1e-10

This produces a 1D boolean mask of shape (P,) regardless of whether c_raw is 1D (single-output) or 2D (multi-output). The union approach is validated by brainstorm T2.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single line replacement; union is exactly correct for multi-task (group sparsity) and conservatively defensible for multi-class</risk>
      <rollback>Revert to the original single expression</rollback>
    </change>
    <change id="C3" priority="P0" source_item="F3">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Add df2 guard and covariance rank check to _hotelling_t2. Return NaN when the F-test is undefined.</description>
      <spec>
In _hotelling_t2, replace lines 564-577 (from the scipy import through the return statement) with:

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

The warning message directs users to per-contrast t-tests and Tier 2 L2-norm CIs per brainstorm T3 validation.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - defensive guard added before existing logic; existing behavior unchanged when df2 > 0 and S is full-rank</risk>
      <rollback>Remove the two guard blocks (df2 check and rank check)</rollback>
    </change>
    <change id="C4" priority="P0" source_item="F5">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Add levels_override parameter to _code_moderator. Pass full-sample levels from the bootstrap and selection frequency paths to ensure consistent coding dimensions across resampled iterations.</description>
      <spec>
Part 1: Modify _code_moderator signature and nominal path (lines 264-320).

Change signature from:
    def _code_moderator(moderator_series, moderator_type, train_idx):

To:
    def _code_moderator(moderator_series, moderator_type, train_idx, levels_override=None):

In the nominal path, replace lines 296-297:
    train_vals = moderator_series.iloc[train_idx]
    levels = sorted(train_vals.unique().tolist())

With:
    if levels_override is not None:
        levels = levels_override
    else:
        train_vals = moderator_series.iloc[train_idx]
        levels = sorted(train_vals.unique().tolist())

Update the docstring Parameters section to add:
    levels_override : list or None, optional
        Pre-computed sorted level list for nominal moderators. When provided,
        overrides training-set-derived levels. Used by bootstrap and selection
        frequency paths to ensure consistent coding dimensions across
        resampled iterations. Default: None (derive from train_idx).

Part 2: Pass full-sample levels from bootstrap path (line 2700 in _boot_task).

_boot_task receives `moderator` dict. Before the Parallel dispatch in run_bootstrap (after line 3198), add full-sample levels to the moderator dict:

After line 3198 (X_cov_for_boot = ...):
    full_sample_levels = None
    if moderator is not None and moderator['type'] == 'nominal':
        full_sample_levels = sorted(moderator['series'].unique().tolist())

Then modify the Parallel call (lines 3212-3221) to pass full_sample_levels:
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

Add full_sample_levels parameter to _boot_task signature (line 2613):
    def _boot_task(X_brain, Y, weights, seed, config, best_params, reducer_template,
                   apriori_map=None, X_cov=None, active_covs=None, moderator=None,
                   full_sample_levels=None):

In _boot_task at line 2700, pass levels_override:
    mod_coded, coding_info = _code_moderator(
        moderator['series'], moderator['type'], idx,
        levels_override=full_sample_levels
    )

Part 3: Apply same fix to selection frequency path.

In run_selection_frequency, compute full_sample_levels before the subsampling loop (same pattern as Part 2). Pass it to the _code_moderator call at line 2377. This requires reading the run_selection_frequency function to identify the exact insertion point.

The CV fold paths (lines 1675, 2190) do NOT receive levels_override: training-set-derived levels are correct for fold-local coding where unseen test levels should be coded as zero.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive parameter with None default; all existing call sites unaffected; only bootstrap and selection frequency paths pass the override</risk>
      <rollback>Remove levels_override parameter and revert all call sites</rollback>
    </change>
    <change id="C5" priority="P1" source_item="F4">
      <file path="README.md" action="modify" />
      <description>Document Tier 1 fold-dependence limitation in Known Limitations section.</description>
      <spec>
Append the following bullet to the Known Limitations section (after line 302):

- Tier 1 fold-ensemble p-values (report_fold_ensemble_importance.csv) treat K
  fold-level coefficient estimates as independent observations in a one-sample
  t-test. Because adjacent folds share overlapping training data, the naive
  variance estimator is downward biased (Bengio and Grandvalet, 2004),
  producing anti-conservative p-values whose Type I error exceeds the nominal
  alpha by an algorithm-dependent amount. Tier 1 is designed as a liberal
  sensitivity screen; Tier 2 bootstrap CIs (report_fold_bootstrap_ci.csv)
  provide the confirmatory inference and are not affected by this bias.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - documentation only</risk>
      <rollback>Remove the added bullet</rollback>
    </change>
    <change id="C6" priority="P1" source_item="F6">
      <file path="README.md" action="modify" />
      <description>Document logistic Partial Ridge adaptation and percentile CI coverage note in Known Limitations and Feature importance sections.</description>
      <spec>
1. Append the following bullet to the Known Limitations section (after the C5 bullet):

- For classification, the Partial Ridge bootstrap CI method (Liu et al., 2020)
  is adapted from its original linear-model formulation to logistic regression
  via column scaling (selected features scaled by sqrt(n) with fixed L2
  penalty C=1). This adaptation is not prescribed by Liu et al. (2020) and
  should be considered a project-specific extension. Percentile bootstrap CI
  coverage for this configuration (fold-wise-pooled, Partial-Ridge-refitted,
  elastic net) has not been directly benchmarked in the literature; however,
  percentile CIs in regularized settings tend toward conservative overcoverage
  (wider intervals), which is favorable for the pipeline's zero-crossing
  thresholding use case.

2. In the Feature importance subsection (line 213), after the existing paragraph about bootstrap CIs (ending at line 218), append:

For classification, the Partial Ridge method adapts Liu et al. (2020) from
linear to logistic regression via differential L2 penalization (selected
features scaled by sqrt(n), C=1). This is a project-specific extension; see
Known Limitations.
      </spec>
      <dependencies>C5 (C6 appends after C5's bullet in Known Limitations)</dependencies>
      <risk>low - documentation only</risk>
      <rollback>Remove the added text</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3, C4, C5, C6</execution_order>
</implement_plan>
