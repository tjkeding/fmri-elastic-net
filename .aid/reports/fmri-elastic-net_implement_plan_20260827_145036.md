<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-08-27T18:50:40+00:00" />
  <input_reports>
    <report path="fmri-elastic-net_brainstorm_20260827_183754.md" mode="brainstorm" key_items="7" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="T1: config schema and validation">
      <file path="config_template.yaml" action="modify" />
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Add reference_class config parameter (mandatory for multi-class classification, hard halt if absent) and validate in main().</description>
      <spec>
        config_template.yaml:
        - After the moderator_type entry (~line 74), add:

            reference_class: null   # Required when analysis_type is "classification" AND the outcome
                                    #   has more than 2 classes (multi-class). Specifies the reference
                                    #   class for fully standardized coefficient computation.
                                    #   Coefficients are reported as class-vs-reference contrasts.
                                    #   Pipeline halts with a diagnostic message if absent for
                                    #   multi-class classification. Ignored for regression and
                                    #   binary classification.

        fmri-elastic-net.py, main() (~line 3806, after bootstrap_ci_method validation):
        - Add validation block:
          1. Read reference_class from config: `ref_class_check = data_cfg_check.get('reference_class')`
          2. This validation runs BEFORE load_and_prep_data (pre-data-load static check is limited).
             Full validation with class-label existence check happens post-data-load (see below).

        fmri-elastic-net.py, main() (~line 3814, after load_and_prep_data returns):
        - Add post-data-load validation:
          1. Detect multi-class: `is_multiclass = config['analysis_type'] == 'classification' and len(np.unique(Y)) > 2`
          2. If is_multiclass:
             a. ref_class = config.get('data_cols', {}).get('reference_class')
             b. If ref_class is None: `sys.exit("CRITICAL: config.data_cols.reference_class is required for multi-class classification. Specify one of the class labels as the reference.")`
             c. classes = sorted(np.unique(Y).tolist())
             d. If ref_class not in classes: `sys.exit(f"CRITICAL: reference_class '{ref_class}' not found in outcome classes {classes}.")`
             e. Store in runtime config: `config.setdefault('_runtime', {})['reference_class'] = ref_class`
             f. `config['_runtime']['class_labels'] = classes`
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive config parameter with validation-only logic; no change to existing behavior when reference_class is null and classification is binary or regression</risk>
      <rollback>Remove reference_class from config_template.yaml; remove validation block from main().</rollback>
    </change>

    <change id="C2" priority="P0" source_item="T1: cross-validated predictions for SD(Y*) computation">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Modify run_nested_cv to return cross-validated probabilities (for classification) alongside score and fold_models, enabling SD(Y*) computation in main().</description>
      <spec>
        run_nested_cv (lines 1596-1902):
        - Current return (line 1902): `return score, fold_models`
        - New return: `return score, fold_models, cv_data`
        - cv_data construction (insert after line 1883, before the logging.info):

            cv_data = {}
            if config['analysis_type'] == 'classification':
                cv_data['cv_probs'] = Y_prob    # shape (N, K_classes)
                cv_data['cv_true'] = Y_true     # shape (N,)
            else:
                cv_data['cv_pred'] = Y_pred     # shape (N,) or (N, K_tasks)
                cv_data['cv_true'] = Y_true

        - Return: `return score, fold_models, cv_data`

        main() (line 3850):
        - Update call: `actual, fold_models, cv_data = run_nested_cv(config, X_brain, Y, weights, X_cov, active_covs, apriori_map, moderator)`

        Note: _run_cv_fold_loop (used by permutation/block permutation) is a separate function
        that returns only `score`. No change needed there.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - adds a return value; the only call site that unpacks fold_models is main(), updated in this change</risk>
      <rollback>Revert run_nested_cv return to (score, fold_models); revert main() call site.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="T1: SD(Y/Y*) divisor computation">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>New helper function to compute the fully standardized coefficient divisor: SD(Y) for regression, SD(Y*) = sqrt(Var(CV_logits) + pi^2/3) for classification. Invoked in main() after run_nested_cv.</description>
      <spec>
        New function _compute_sd_y_divisor (insert after _get_task_labels, ~line 177):

        def _compute_sd_y_divisor(config, Y, cv_data):
            """Compute the SD(Y) or SD(Y*) divisor for fully standardized coefficients.

            Parameters
            ----------
            config : dict
                Pipeline configuration.
            Y : pd.Series or pd.DataFrame
                Outcome variable(s).
            cv_data : dict
                Cross-validated predictions from run_nested_cv. For classification,
                contains 'cv_probs' (N, K_classes). For regression, contains 'cv_pred'.

            Returns
            -------
            float or ndarray
                Scalar for single-output (regression, binary classification).
                Array of length K_tasks for multi-task regression.
                Array of length K-1 for multi-class classification (per reference-coded contrast).
            """

        Implementation logic:

        1. Regression (single-output):
           `return np.std(Y.values, ddof=1)`

        2. Multi-task regression:
           `return np.array([np.std(Y.iloc[:, k].values, ddof=1) for k in range(Y.shape[1])])`

        3. Binary classification:
           cv_probs = cv_data['cv_probs']  # (N, 2)
           p1 = np.clip(cv_probs[:, 1], 1e-15, 1 - 1e-15)
           cv_logits = np.log(p1 / (1 - p1))
           return np.sqrt(np.var(cv_logits, ddof=1) + np.pi**2 / 3)

        4. Multi-class classification:
           cv_probs = cv_data['cv_probs']  # (N, K)
           class_labels = sorted(np.unique(Y))
           ref_class = config['_runtime']['reference_class']
           ref_idx = class_labels.index(ref_class)
           sd_contrasts = []
           for k in range(len(class_labels)):
               if k == ref_idx:
                   continue
               p_k = np.clip(cv_probs[:, k], 1e-15, 1 - 1e-15)
               p_ref = np.clip(cv_probs[:, ref_idx], 1e-15, 1 - 1e-15)
               logits_k = np.log(p_k / p_ref)
               sd_contrasts.append(np.sqrt(np.var(logits_k, ddof=1) + np.pi**2 / 3))
           return np.array(sd_contrasts)

        Branching: use _is_multitask(config, Y) for multi-task check;
        config['analysis_type'] == 'classification' for classification;
        len(class_labels) > 2 for multi-class.

        main() invocation (after the C1 validation block, ~line 3852):
        sd_y_divisor = _compute_sd_y_divisor(config, Y, cv_data)
        logging.info(f"SD(Y/Y*) divisor for fully standardized coefficients: {sd_y_divisor}")
      </spec>
      <dependencies>C1, C2</dependencies>
      <risk>medium - classification path uses log-odds which require numerical guards (np.clip to prevent log(0)). Cross-validated probabilities from sklearn should not contain exact 0.0 or 1.0 in practice, but edge cases with extreme class imbalance or LOO with single test observations could approach boundaries. The 1e-15 clip provides 15 orders of magnitude of numerical safety.</risk>
      <rollback>Remove _compute_sd_y_divisor function; remove invocation from main().</rollback>
    </change>

    <change id="C4" priority="P0" source_item="T1: reference-differencing for multi-class coefficients">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>New helper function for multi-class reference-differencing (beta_k - beta_ref). Applied in run_bootstrap (Tier 2) and run_tier1_inference (Tier 1) to transform K-class softmax coefficients to K-1 reference-coded contrasts before reporting.</description>
      <spec>
        New function _reference_difference (insert after _compute_sd_y_divisor):

        def _reference_difference(coef_array, class_labels, reference_class):
            """Reference-difference multi-class coefficients: beta_k - beta_ref per contrast.

            Parameters
            ----------
            coef_array : ndarray
                Shape (..., K, P) where K is the number of classes and P is the number
                of features. The K dimension is axis=-2.
            class_labels : list
                Sorted class labels of length K.
            reference_class : str or int or float
                The reference class label.

            Returns
            -------
            contrasts : ndarray
                Shape (..., K-1, P). Each slice along K-1 is beta_k - beta_ref.
            contrast_labels : list of str
                Labels for K-1 contrasts: ['{class_k}_vs_{reference_class}', ...].
            """
            ref_idx = class_labels.index(reference_class)
            K = len(class_labels)
            non_ref = [i for i in range(K) if i != ref_idx]
            # Extract reference slice and subtract
            ref_slice = np.take(coef_array, [ref_idx], axis=-2)  # (..., 1, P)
            non_ref_slices = np.take(coef_array, non_ref, axis=-2)  # (..., K-1, P)
            contrasts = non_ref_slices - ref_slice  # broadcast: (..., K-1, P)
            contrast_labels = [f'{class_labels[i]}_vs_{reference_class}' for i in non_ref]
            return contrasts, contrast_labels

        Application sites:

        1. run_bootstrap (~line 3488-3577, the `is_multi` branch):
           - After stacking coef_array (B, K, P), detect multi-class:
             `is_multiclass = config['analysis_type'] == 'classification' and coef_array.shape[1] > 2`
           - If is_multiclass:
             a. class_labels = config['_runtime']['class_labels']
             b. ref_class = config['_runtime']['reference_class']
             c. coef_array, task_labels = _reference_difference(coef_array, class_labels, ref_class)
             d. Also reference-difference coef_array_interaction if has_moderator
           - The per-task loop then iterates over K-1 contrasts instead of K classes
           - Output directories: task_{contrast_label}/ instead of task_{class_label}/

        2. run_tier1_inference (~line 2113-2170, the `is_multi` branch):
           - After stacking coef_array (K_folds, K_tasks, P), detect multi-class:
             `is_multiclass = config['analysis_type'] == 'classification' and coef_array.shape[1] > 2`
           - If is_multiclass:
             a. coef_array, task_labels = _reference_difference(coef_array, class_labels, ref_class)
           - Per-task loop iterates over K-1 contrasts

        3. Selection frequency (run_selection_frequency): NO CHANGE.
           Selection indicators are binary (feature selected/not selected by the model).
           Reference-differencing affects coefficient magnitude, not selection status.
           Multi-class selection frequency continues to report per-class indicators.

        4. Saved bootstrap distributions (bootstrap_coef_distribution.npz):
           - coef_dist shape changes from (B, K, P) to (B, K-1, P) for multi-class
           - task_labels array changes from K class labels to K-1 contrast labels
      </spec>
      <dependencies>C1</dependencies>
      <risk>high - changes multi-class output structure (K classes to K-1 contrasts) for Tier 1, Tier 2, and saved distributions. Interaction coefficients in the multi-class branch (coef_array_interaction with shape (B, K, n_mod_cols, P) or similar) must also be reference-differenced along the K dimension. Selection frequency is deliberately left unchanged to avoid conflating coefficient interpretation with model sparsity. Existing multi-class test cases will need updating for the new output structure.</risk>
      <rollback>Remove _reference_difference function; remove application-site calls. Multi-class output reverts to K-class parameterization.</rollback>
    </change>

    <change id="C5" priority="P0" source_item="T1: fully standardized coefficient derivation in _compute_importance_preamble">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Modify _compute_importance_preamble to accept sd_y_divisor and derive std_coef (fully standardized) and raw_coef (Y-unit-scale) independently from the internal (X-standardized) bootstrap coefficient distribution. Thread sd_y_divisor through _compute_importance_report and run_bootstrap.</description>
      <spec>
        _compute_importance_preamble (lines 3008-3041):
        - New signature: `def _compute_importance_preamble(df_coef, all_feats, feat_std_map, config, sd_y_divisor=1.0):`
        - Replace current body (lines 3021-3041) with:

            out_dir = config['paths']['output_dir']
            alpha = 1.0 - config['stats_params']['ci_level']

            # Internal (X-standardized) statistics from bootstrap distribution
            internal_means = df_coef.mean()
            internal_ci_low = df_coef.quantile(alpha / 2)
            internal_ci_high = df_coef.quantile(1 - alpha / 2)

            # Fully standardized: internal / SD(Y or Y*)
            # sd_y_divisor is always a positive scalar at this call site
            std_means = internal_means / sd_y_divisor
            std_ci_low = internal_ci_low / sd_y_divisor
            std_ci_high = internal_ci_high / sd_y_divisor

            # Raw (Y-unit-scale): internal / SD(X) — independent derivation
            feat_std_map_aligned = feat_std_map.reindex(df_coef.columns).fillna(1.0)
            raw_means = np.divide(internal_means, feat_std_map_aligned,
                                  out=np.zeros_like(internal_means.values, dtype=float),
                                  where=feat_std_map_aligned != 0)
            raw_ci_low = np.divide(internal_ci_low, feat_std_map_aligned,
                                   out=np.zeros_like(internal_ci_low.values, dtype=float),
                                   where=feat_std_map_aligned != 0)
            raw_ci_high = np.divide(internal_ci_high, feat_std_map_aligned,
                                    out=np.zeros_like(internal_ci_high.values, dtype=float),
                                    where=feat_std_map_aligned != 0)

            # Scale-invariant quantities (unchanged by positive-constant division)
            is_sig = (std_ci_low > 0) | (std_ci_high < 0)
            pd_val = pd.concat([(df_coef > 0).mean(), (df_coef < 0).mean()], axis=1).max(axis=1)

            return dict(
                std_means=std_means, std_ci_low=std_ci_low, std_ci_high=std_ci_high,
                raw_means=raw_means, raw_ci_low=raw_ci_low, raw_ci_high=raw_ci_high,
                is_sig=is_sig, pd_val=pd_val, out_dir=out_dir, alpha=alpha
            )

        Note on np.divide: the current code uses `np.divide(std_means, feat_std_map_aligned, out=np.zeros_like(std_means), where=...)`.
        After the change, `std_means` is a pd.Series (result of division), so the out= allocation must
        use `.values` explicitly or use a float-typed array to avoid dtype issues.

        _compute_importance_report (line 3149-3165):
        - New signature: add sd_y_divisor=1.0 parameter
        - Thread to _compute_importance_preamble call (line 3155):
          `stats = _compute_importance_preamble(df_coef, all_feats, feat_std_map, config, sd_y_divisor=sd_y_divisor)`

        _report_standard (lines 3097-3146):
        - New signature: add sd_y_divisor=1.0 parameter (passed from _compute_importance_report)
        - Note: _report_standard does NOT call _compute_importance_preamble directly; it receives `stats`
          from the dispatcher. No change needed within _report_standard itself.
        - Wait, correction: _report_standard receives `stats` as a positional arg. The sd_y_divisor
          threading is handled by _compute_importance_report calling _compute_importance_preamble.
          But _report_standard's interaction branch also calls _compute_importance_preamble via the
          interaction_report_df path. Let me trace:
          - In run_bootstrap, the interaction stats are computed by a separate
            `_compute_importance_preamble(df_coef_interaction, ...)` call (line 3446).
          - That call also needs sd_y_divisor.

        run_bootstrap (lines 3285-3577):
        - New signature: add sd_y_divisor=1.0 parameter
        - Single-output branch (not is_multi):
          - Main effect _compute_importance_preamble call (via _compute_importance_report):
            thread sd_y_divisor
          - Interaction _compute_importance_preamble call (line 3446):
            `int_stats = _compute_importance_preamble(df_coef_interaction, all_feats_report, feat_std_map_report, config, sd_y_divisor=sd_y_divisor)`
          - _compute_importance_report call (line 3454):
            `_compute_importance_report(df_coef, ..., sd_y_divisor=sd_y_divisor)`

        - Multi-output branch (is_multi, line 3488+):
          - Per-task loop: use per-task divisor
            `sd_y_k = sd_y_divisor[k] if isinstance(sd_y_divisor, np.ndarray) else sd_y_divisor`
          - Interaction _compute_importance_preamble call (line 3531):
            `int_stats_k = _compute_importance_preamble(df_coef_int_k, ..., sd_y_divisor=sd_y_k)`
          - _compute_importance_report call (line 3539):
            `_compute_importance_report(df_coef_k, ..., sd_y_divisor=sd_y_k)`

        main() (line 3859):
        - `run_bootstrap(config, X_brain, Y, weights, subj_ids, X_cov, active_covs, fold_models, apriori_map, moderator, sd_y_divisor=sd_y_divisor)`
      </spec>
      <dependencies>C3</dependencies>
      <risk>medium - core reporting change affecting all output files with std_coef or raw_coef columns. Risk mitigated by: (1) is_significant, pd, p_value are scale-invariant (zero-crossing and sign-proportion tests unchanged by positive-constant division); (2) default sd_y_divisor=1.0 preserves current behavior when unspecified; (3) the derivation formula is algebraically equivalent to Long (1997)/Menard (2004) for classification and to the standard fully standardized beta for regression.</risk>
      <rollback>Revert _compute_importance_preamble to original body; remove sd_y_divisor parameter from _compute_importance_report and run_bootstrap; revert main() call.</rollback>
    </change>

    <change id="C6" priority="P0" source_item="T2: interaction CSV residualization + T1xT2 cross-cutting visualization fix">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>
        Two coupled changes in calculate_visualization_data:
        (a) T1 cross-cut: switch coefficient source from std_coef_mean * f_val_scaled to raw_coef_mean * (f_val_raw - mean) for all effect types. After T1 redefines std_coef as fully standardized (in SDs of Y), using std_coef for partial residuals would produce values in SD(Y) units rather than Y units.
        (b) T2: for interaction effect_type, extend lin_contrib to include the full conditional relationship (brain main + moderator main + interaction) instead of the interaction term alone. This matches the standard convention (Long 2019 interactions package, jtools partialize(), Aiken and West 1991, Hayes 2022).
      </description>
      <spec>
        calculate_visualization_data (lines 2841-2953):

        1. Signature change (line 2841):
           Add main_report_df=None parameter:
           `def calculate_visualization_data(config, X_full, Y, weights, subject_ids, best_model, report_df, level, X_brain_raw=None, moderator=None, effect_type=None, main_report_df=None):`

        2. T1 cross-cut: main-effect coefficient source (lines 2933-2938):
           Replace:
             f_weight = row['std_coef_mean']
             f_val_scaled = (f_val_raw - f_val_raw.mean()) / (f_val_raw.std() + 1e-10)
             if effect_type == 'interaction' and mod_coded_vals is not None:
                 lin_contrib = f_weight * f_val_scaled * mod_coded_vals
             else:
                 lin_contrib = f_weight * f_val_scaled

           With:
             f_raw_coef = row['raw_coef_mean']
             f_centered = f_val_raw - f_val_raw.mean()

             if effect_type == 'interaction' and mod_coded_vals is not None:
                 # T2: Full conditional relationship (brain main + moderator main + interaction)

                 # 1. Brain main effect from main-effect report
                 brain_raw_coef = 0.0
                 if main_report_df is not None:
                     feat_col_main = ('component_id' if 'component_id' in main_report_df.columns else
                                      ('cluster_id' if 'cluster_id' in main_report_df.columns else 'feature'))
                     match = main_report_df[main_report_df[feat_col_main] == f_name]
                     if not match.empty:
                         brain_raw_coef = match['raw_coef_mean'].values[0]
                 brain_main = brain_raw_coef * f_centered

                 # 2. Moderator main effect from model coefficients
                 # Identify moderator column(s) in X_full
                 mod_col_name = moderator['series'].name
                 mod_main = np.zeros(len(Y_arr))
                 if mod_col_name in X_full.columns:
                     mod_col_idx = list(X_full.columns).index(mod_col_name)
                     mod_main = coeffs[mod_col_idx] * X_scaled.iloc[:, mod_col_idx].values
                 else:
                     # Deviation-coded moderator: find coded column(s)
                     mod_cols_in_X = [c for c in X_full.columns
                                      if c.startswith(f'{mod_col_name}_')]
                     for mc in mod_cols_in_X:
                         mc_idx = list(X_full.columns).index(mc)
                         mod_main = mod_main + coeffs[mc_idx] * X_scaled.iloc[:, mc_idx].values

                 # 3. Interaction from interaction report
                 int_contrib = f_raw_coef * f_centered * mod_coded_vals

                 lin_contrib = brain_main + mod_main + int_contrib
             else:
                 lin_contrib = f_raw_coef * f_centered

        3. Update call sites that invoke calculate_visualization_data with effect_type='interaction':

           _report_standard (line 3141-3146):
           Add main_report_df=indiv_df:
             calculate_visualization_data(
                 config, X_full, Y, weights, subject_ids, best_model,
                 interaction_report_df, 'individual',
                 X_brain if red_method != 'none' else None,
                 moderator=moderator, effect_type='interaction',
                 main_report_df=indiv_df
             )

           _report_apriori: check if it calls calculate_visualization_data with
           effect_type='interaction'. If so, pass main_report_df=indiv_df. If not (apriori
           interaction visualization is handled differently), no change needed. Build phase
           will verify.

        Algebraic verification of the coefficient switch:
        - Before T1: std_coef = internal (X-standardized). f_val_scaled = (X - mean_X) / SD(X).
          Product: internal * (X - mean_X) / SD(X) = (internal / SD(X)) * (X - mean_X) = raw_coef * (X - mean_X).
        - After T1: raw_coef = internal / SD(X) (unchanged definition).
          Product: raw_coef * (X - mean_X) = same result.
        - Algebraic equivalence holds. The switch is necessary because std_coef is redefined
          to internal / SD(Y), which would produce incorrect Y-scale partial residuals.
      </spec>
      <dependencies>C5</dependencies>
      <risk>medium - T2 interaction change modifies the partial residual formula. The moderator main-effect extraction from the representative model (fold-0) is consistent with how linear_pred_full is computed (also from fold-0 model). The brain main-effect lookup requires main_report_df to contain the same features as the interaction report; this is guaranteed by the report construction flow (_build_individual_report_df uses the same all_feats for both). Risk: moderator column identification relies on moderator['series'].name matching the column name in X_full or a prefix pattern for coded columns. This should hold given the _code_moderator naming convention but needs verification during build.</risk>
      <rollback>Revert calculate_visualization_data body to original; remove main_report_df parameter; revert call sites.</rollback>
    </change>

    <change id="C7" priority="P1" source_item="T1: documentation caveats">
      <file path="fmri-elastic-net.py" action="modify" />
      <file path="INPUT_SPECIFICATION.md" action="modify" />
      <file path="README.md" action="modify" />
      <description>Add four documentation caveats specified in the brainstorm report for the fully standardized coefficient redefinition, and document the new reference_class parameter.</description>
      <spec>
        fmri-elastic-net.py module docstring (after line 83, before "Written by"):
        Add caveats:
        - For classification, SD(Y*) is model-dependent (Menard, 2011): it changes as
          predictors are added or removed, unlike SD(Y) for regression.
        - The SD(Y/Y*) divisor is computed once from cross-validated predictions and applied
          uniformly to all bootstrap statistics (approximation analogous to the raw_coef
          back-projection caveat above).
        - Fully standardized coefficients remain unbounded in multiple regression (Friedman
          and Wall, 2005). The [-1, 1] bound applies only to simple bivariate regression.
        - Multi-class per-contrast latent-variable standardization is a principled extension
          of the binary-case derivation (Long, 1997; Menard, 2004) but lacks direct published
          validation for multinomial softmax models.

        Update existing caveat (line 60-63):
        - The raw_coef caveat currently reads: "raw_coef_mean with reduction methods ... is approximate:
          the back-projected standardized coefficient is divided by original-feature SD..."
        - Update to reflect the new derivation: raw_coef is now derived independently from the
          internal (X-standardized) coefficient by dividing by SD(X), not from the redefined
          std_coef. The approximation caveat for reduction methods still applies (back-projected
          coefficient divided by original-feature SD is not equivalent to direct regression).

        INPUT_SPECIFICATION.md:
        - Document reference_class parameter under the data_cols section
        - Document updated std_coef semantics (fully standardized, not X-standardized)
        - Add note about multi-class output structure change (K-1 contrasts vs K classes)

        README.md:
        - Update coefficient interpretation section (if one exists) or add brief note about
          std_coef being fully standardized (SDs of Y per 1 SD of X)
        - Document reference_class parameter
      </spec>
      <dependencies>C1, C5</dependencies>
      <risk>low - text-only changes with no behavioral impact</risk>
      <rollback>Revert text changes to module docstring, INPUT_SPECIFICATION.md, README.md.</rollback>
    </change>
  </changes>

  <execution_order>C1, C2, C3, C4, C5, C6, C7</execution_order>

  <notes>
    Assumptions made without explicit approval:
    1. Tier 1 inference (report_fold_ensemble_importance.csv) column names (fold_mean_coef, ci_low_t,
       ci_high_t) remain in X-standardized (internal) space. These are diagnostic fold-level statistics,
       not the primary inferential quantities. The fully standardized transformation applies only to the
       Tier 2 importance report (report_feature_importance.csv: std_coef_mean, std_ci_low, std_ci_high)
       and the interaction importance report. Rationale: the Tier 1 t-test is a zero-crossing test on
       internal coefficients; its significance determination is invariant to the divisor.
    2. Selection frequency (report_selection_frequency.csv) remains in the K-class parameterization for
       multi-class classification, reporting per-class selection indicators. Reference-differencing is NOT
       applied to selection frequency because selection indicators reflect model sparsity (whether a
       feature's coefficient is nonzero), which is a property of the model parameterization, not the
       coefficient interpretation. Rationale: a feature selected in the softmax model for class k and not
       for class j would have a nonzero contrast coefficient regardless of reference-differencing, so the
       information content is equivalent; but the K-class format is more directly interpretable for
       understanding model sparsity.
    3. The Tier 2 bootstrap CI report (report_fold_bootstrap_ci.csv) column boot_mean_coef remains in
       internal (X-standardized) space. This is a diagnostic output from the raw bootstrap distribution,
       separate from the importance report's fully standardized std_coef.
    4. For the interaction visualization (C6/T2), the moderator main-effect contribution is extracted from
       the representative fold-0 model's fitted coefficients, consistent with how linear_pred_full is
       computed. The brain main-effect and interaction coefficients come from the bootstrap-averaged
       report. This mixes coefficient sources (fold-0 model vs bootstrap average) for the moderator
       main-effect, which is an acceptable approximation because the moderator has a protected near-zero
       penalty weight (0.001), making its coefficient highly stable across folds and bootstrap iterations.
    5. For binary classification, SD(Y*) uses all N subjects' cross-validated logits (each subject
       predicted by their held-out fold). This is the fold-wise ensemble analog of the full-data
       Var(X*beta) derivation.
  </notes>
</implement_plan>
