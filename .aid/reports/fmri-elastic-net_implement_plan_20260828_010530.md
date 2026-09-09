<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-08-28T01:05:30+00:00" />
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

    <change id="C2" priority="P0" source_item="T1: run_nested_cv expansion for per-fold data">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Expand run_nested_cv to return per-fold held-out data (indices, predictions, probabilities, true values, per-fold scores) and a fold_assignments array. This data feeds three downstream consumers: per-fold SD(Y/Y*) (C3), OOF ensemble visualization (C6), and per-fold performance metrics (C8).</description>
      <spec>
        run_nested_cv (lines 1596-1902):

        1. Initialize fold_assignments array before the fold loop (after line 1648):

            fold_assignments = np.full(len(Y), -1, dtype=int)
            fold_held_out = []

        2. Inside the fold loop, after search.fit and predictions (after line 1770):

            fold_assignments[te] = fold_idx

            # Per-fold held-out data for downstream consumers
            fold_score_k = None
            if config['analysis_type'] == 'classification':
                y_prob_fold = search.predict_proba(X_te)
                if y_prob_fold.shape[1] > 2:
                    fold_score_k = roc_auc_score(Y_te, y_prob_fold, multi_class='ovr')
                else:
                    fold_score_k = roc_auc_score(Y_te, y_prob_fold[:, 1])
            else:
                y_pred_fold = search.predict(X_te)
                fold_score_k = r2_score(
                    Y_te.values if hasattr(Y_te, 'values') else Y_te,
                    y_pred_fold,
                    multioutput='uniform_average'
                )

            fold_held_out.append({
                'fold_idx': fold_idx,
                'indices': te,
                'y_true': Y_te.values if hasattr(Y_te, 'values') else np.asarray(Y_te),
                'y_pred': y_preds[-1],
                'y_prob': y_probs[-1] if config['analysis_type'] == 'classification' else None,
                'score': fold_score_k,
            })

           NOTE: y_preds[-1] and y_probs[-1] refer to the values just appended at
           lines 1767-1770 in the current code. The per-fold score computation reuses
           the same predictions already appended to the aggregation lists; it does NOT
           call search.predict or search.predict_proba a second time.

        3. Construct cv_data dict (insert before the return, after line 1900):

            cv_data = {
                'fold_assignments': fold_assignments,
                'fold_held_out': fold_held_out,
            }
            if config['analysis_type'] == 'classification':
                cv_data['cv_probs'] = Y_prob
                cv_data['cv_true'] = Y_true
            else:
                cv_data['cv_pred'] = Y_pred
                cv_data['cv_true'] = Y_true

        4. Change return (line 1902):
           FROM: `return score, fold_models`
           TO:   `return score, fold_models, cv_data`

        5. Update call site in main() (line 3850):
           FROM: `actual, fold_models = run_nested_cv(...)`
           TO:   `actual, fold_models, cv_data = run_nested_cv(...)`

        6. Write per-fold performance metrics (insert after the aggregate
           _compute_evaluation_metrics call, ~line 3900):

            _write_fold_performance(cv_data, config, out_dir)

           New function _write_fold_performance (insert after _write_fold_diagnostics,
           ~line 1944):

            def _write_fold_performance(cv_data, config, out_dir):
                """Write per-fold performance metrics and fold-stability summary.

                Outputs
                -------
                model_performance_per_fold.csv : per-fold score (R2 or AUC)
                """
                fold_held_out = cv_data['fold_held_out']
                rows = []
                metric_name = 'R2' if config['analysis_type'] == 'regression' else 'AUC_ROC'
                for fh in fold_held_out:
                    rows.append({
                        'fold_idx': fh['fold_idx'],
                        'metric': metric_name,
                        'value': fh['score'],
                        'n_held_out': len(fh['indices']),
                    })
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(out_dir, 'model_performance_per_fold.csv'), index=False)

                scores = np.array([fh['score'] for fh in fold_held_out])
                summary = pd.DataFrame([{
                    'metric': metric_name,
                    'mean': float(scores.mean()),
                    'sd': float(scores.std(ddof=1)) if len(scores) > 1 else float('nan'),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'n_folds': len(scores),
                }])
                summary.to_csv(
                    os.path.join(out_dir, 'model_performance_fold_summary.csv'), index=False
                )
                logging.info(
                    f"Per-fold {metric_name}: mean={scores.mean():.4f}, "
                    f"SD={scores.std(ddof=1):.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]"
                )

        7. Update run_nested_cv docstring Returns section to document cv_data.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive return value; per-fold score computation reuses existing predictions (no new predict calls)</risk>
      <rollback>Revert run_nested_cv return to (score, fold_models); revert main() call site; remove _write_fold_performance.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="T1: per-fold SD(Y/Y*) divisor computation">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>New helper function to compute per-fold SD(Y/Y*) divisors. Returns a dict mapping fold_idx to the scalar (or array) divisor for that fold. SD(Y) for regression, SD(Y*) = sqrt(Var(logits) + pi^2/3) for classification. Per-fold computation prevents information leakage and maintains estimand consistency with fold-specific coefficient estimation.</description>
      <spec>
        New function _compute_sd_y_divisor (insert after _get_task_labels, ~line 177):

        def _compute_sd_y_divisor(config, Y, cv_data):
            """Compute per-fold SD(Y) or SD(Y*) divisors for fully standardized coefficients.

            Parameters
            ----------
            config : dict
                Pipeline configuration.
            Y : pd.Series or pd.DataFrame
                Outcome variable(s).
            cv_data : dict
                Cross-validated data from run_nested_cv containing 'fold_held_out'.

            Returns
            -------
            dict[int, float or ndarray]
                Maps fold_idx to the SD(Y/Y*) divisor for that fold.
                - Scalar for single-output regression and binary classification.
                - Array of length K_tasks for multi-task regression.
                - Array of length K-1 for multi-class classification (per contrast).
            """

        Implementation logic (branched by analysis type):

        1. Single-output regression:
           For each fold in cv_data['fold_held_out']:
               y_held = fold['y_true']
               sd_y_divisors[fold['fold_idx']] = float(np.std(y_held, ddof=1))

        2. Multi-task regression:
           For each fold:
               y_held = fold['y_true']  # (n_held, K_tasks)
               sd_y_divisors[fold['fold_idx']] = np.array([
                   np.std(y_held[:, k], ddof=1) for k in range(y_held.shape[1])
               ])

        3. Binary classification:
           For each fold:
               y_prob = fold['y_prob']  # (n_held, 2)
               p1 = np.clip(y_prob[:, 1], 1e-15, 1 - 1e-15)
               logits = np.log(p1 / (1 - p1))
               sd_y_divisors[fold['fold_idx']] = float(
                   np.sqrt(np.var(logits, ddof=1) + np.pi**2 / 3)
               )

        4. Multi-class classification:
           For each fold:
               y_prob = fold['y_prob']  # (n_held, K)
               class_labels = config['_runtime']['class_labels']
               ref_class = config['_runtime']['reference_class']
               ref_idx = class_labels.index(ref_class)
               sd_contrasts = []
               for k_idx in range(len(class_labels)):
                   if k_idx == ref_idx:
                       continue
                   p_k = np.clip(y_prob[:, k_idx], 1e-15, 1 - 1e-15)
                   p_ref = np.clip(y_prob[:, ref_idx], 1e-15, 1 - 1e-15)
                   logits_k = np.log(p_k / p_ref)
                   sd_contrasts.append(float(
                       np.sqrt(np.var(logits_k, ddof=1) + np.pi**2 / 3)
                   ))
               sd_y_divisors[fold['fold_idx']] = np.array(sd_contrasts)

        Guard: if any per-fold divisor is zero or near-zero (< 1e-10), log a warning
        and set to 1.0 (equivalent to reporting internal coefficients for that fold).

        Branching: use _is_multitask(config, Y) for multi-task detection;
        config['analysis_type'] for classification vs regression;
        len(config['_runtime']['class_labels']) > 2 for multi-class (only valid after C1 validation).

        main() invocation (after the C1 validation block, ~line 3852):
            sd_y_divisors = _compute_sd_y_divisor(config, Y, cv_data)
            logging.info(f"Per-fold SD(Y/Y*) divisors computed for {len(sd_y_divisors)} folds.")
      </spec>
      <dependencies>C1, C2</dependencies>
      <risk>medium - classification path uses log-odds requiring numerical guards. The 1e-15 clip provides 15 orders of magnitude of safety. Per-fold computation with small held-out sets (e.g., LOO with 1 subject per fold) produces degenerate SD(Y*) values; the near-zero guard handles this gracefully.</risk>
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
            ref_slice = np.take(coef_array, [ref_idx], axis=-2)
            non_ref_slices = np.take(coef_array, non_ref, axis=-2)
            contrasts = non_ref_slices - ref_slice
            contrast_labels = [f'{class_labels[i]}_vs_{reference_class}' for i in non_ref]
            return contrasts, contrast_labels

        Application sites:

        1. run_bootstrap (~line 3489, the is_multi branch):
           After stacking coef_array (B, K, P), detect multi-class:
           `is_multiclass = config['analysis_type'] == 'classification' and coef_array.shape[-2] > 2`
           If is_multiclass:
             class_labels = config['_runtime']['class_labels']
             ref_class = config['_runtime']['reference_class']
             coef_array, task_labels = _reference_difference(coef_array, class_labels, ref_class)
             Also reference-difference coef_array_interaction if has_moderator.
           Per-task loop then iterates K-1 contrasts. Output dirs: task_{contrast_label}/.
           Also update the save_distributions block to save contrast_labels as task_labels.

        2. run_tier1_inference (~line 2113, the is_multi branch):
           After stacking coef_array (K_folds, K_tasks, P), detect multi-class.
           If is_multiclass:
             coef_array, task_labels = _reference_difference(coef_array, class_labels, ref_class)
             Also reference-difference coef_array_interaction if has_moderator.
           Per-task loop iterates K-1 contrasts.

        3. Selection frequency (run_selection_frequency): NO CHANGE.
           Selection indicators are binary (selected/not-selected). Reference-differencing
           affects coefficient magnitude, not sparsity. Multi-class selection frequency
           continues to report per-class indicators.

        4. Saved distributions (bootstrap_coef_distribution.npz):
           coef_dist shape changes from (B, K, P) to (B, K-1, P) for multi-class.
           task_labels array changes from K class labels to K-1 contrast labels.
      </spec>
      <dependencies>C1</dependencies>
      <risk>high - changes multi-class output structure (K classes to K-1 contrasts). Interaction coefficients in the multi-class branch must also be reference-differenced. Existing multi-class test cases will need updating.</risk>
      <rollback>Remove _reference_difference function; remove application-site calls.</rollback>
    </change>

    <change id="C5" priority="P0" source_item="T1: fully standardized coefficients + both raw/std in all reports">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>
        Comprehensive coefficient reporting overhaul. All coefficient reports produce both
        raw (Y-unit-scale) and fully standardized (dimensionless) columns. Per-fold SD(Y/Y*)
        divisors are applied at the fold level before pooling in the bootstrap distribution.
        Affects: _compute_importance_preamble, _build_individual_report_df (no change needed,
        already has both column types), _write_tier1_report, _write_tier2_single,
        run_bootstrap (fold-index tracking + sd_y_divisors threading), _compute_importance_report,
        _report_standard, _report_apriori (cluster-level statistics).
      </description>
      <spec>
        --- _compute_importance_preamble (lines 3008-3041) ---

        New signature:
            def _compute_importance_preamble(df_coef_internal, df_coef_std, all_feats, feat_std_map, config):

        Replaces the single df_coef parameter with two:
        - df_coef_internal: bootstrap distribution in internal (X-standardized) space, for raw derivation
        - df_coef_std: bootstrap distribution in fully standardized space, for std derivation

        Replace body (lines 3021-3041) with:

            out_dir = config['paths']['output_dir']
            alpha = 1.0 - config['stats_params']['ci_level']

            # Fully standardized statistics (from pre-standardized distribution)
            std_means = df_coef_std.mean()
            std_ci_low = df_coef_std.quantile(alpha / 2)
            std_ci_high = df_coef_std.quantile(1 - alpha / 2)

            # Raw (Y-unit-scale) statistics: internal / SD(X), independent derivation
            feat_std_map_aligned = feat_std_map.reindex(df_coef_internal.columns).fillna(1.0)
            raw_means = np.divide(
                df_coef_internal.mean().values, feat_std_map_aligned.values,
                out=np.zeros(len(feat_std_map_aligned), dtype=float),
                where=feat_std_map_aligned.values != 0
            )
            raw_ci_low = np.divide(
                df_coef_internal.quantile(alpha / 2).values, feat_std_map_aligned.values,
                out=np.zeros(len(feat_std_map_aligned), dtype=float),
                where=feat_std_map_aligned.values != 0
            )
            raw_ci_high = np.divide(
                df_coef_internal.quantile(1 - alpha / 2).values, feat_std_map_aligned.values,
                out=np.zeros(len(feat_std_map_aligned), dtype=float),
                where=feat_std_map_aligned.values != 0
            )
            raw_means = pd.Series(raw_means, index=df_coef_internal.columns)
            raw_ci_low = pd.Series(raw_ci_low, index=df_coef_internal.columns)
            raw_ci_high = pd.Series(raw_ci_high, index=df_coef_internal.columns)

            # Scale-invariant quantities (sign proportions invariant under positive-constant division)
            is_sig = (std_ci_low > 0) | (std_ci_high < 0)
            pd_val = pd.concat([(df_coef_std > 0).mean(), (df_coef_std < 0).mean()], axis=1).max(axis=1)

            return dict(
                std_means=std_means, std_ci_low=std_ci_low, std_ci_high=std_ci_high,
                raw_means=raw_means, raw_ci_low=raw_ci_low, raw_ci_high=raw_ci_high,
                is_sig=is_sig, pd_val=pd_val, out_dir=out_dir, alpha=alpha
            )

        --- _write_tier1_report (lines 1946-2014) ---

        New signature:
            def _write_tier1_report(coef_matrix, feature_names, out_dir, ci_level,
                                    sd_y_divisors, feat_std_map, fold_model_indices,
                                    effect_type=None, contrast=None):

        New parameters:
        - sd_y_divisors: dict[int, float|ndarray] from C3
        - feat_std_map: pd.Series of SD(X) per feature (from X_brain.std(ddof=1))
        - fold_model_indices: list of fold_idx values corresponding to each row of coef_matrix

        Replace body (lines 1976-2014) with:

            KR, P = coef_matrix.shape
            feat_std_arr = feat_std_map.reindex(feature_names).fillna(1.0).values

            # Per-fold standardization: divide each fold's coefficient by that fold's SD(Y/Y*)
            sd_y_per_fold = np.array([sd_y_divisors[fi] for fi in fold_model_indices])
            if sd_y_per_fold.ndim == 1:
                coef_std = coef_matrix / sd_y_per_fold[:, np.newaxis]
            else:
                # Multi-task/multi-class: sd_y_per_fold is (K, n_tasks); handled per-task upstream
                coef_std = coef_matrix / sd_y_per_fold[:, np.newaxis]

            # Raw: divide each fold's coefficient by SD(X)
            coef_raw = coef_matrix / feat_std_arr[np.newaxis, :]

            # Fully standardized statistics
            fold_mean_std = coef_std.mean(axis=0)
            fold_sd_std = coef_std.std(axis=0, ddof=1)
            fold_cv = np.where(np.abs(fold_mean_std) > 1e-30,
                               np.abs(fold_sd_std / fold_mean_std), np.nan)

            # t-test on internal coefficients (scale-invariant: same t regardless of divisor)
            t_stat, p_val = ttest_1samp(coef_matrix, popmean=0, axis=0)
            alpha_t = 1.0 - ci_level
            dof = KR - 1
            t_crit = t_dist.ppf(1.0 - alpha_t / 2, dof)

            # CIs in both scales
            se_std = fold_sd_std / np.sqrt(KR)
            ci_low_std = fold_mean_std - t_crit * se_std
            ci_high_std = fold_mean_std + t_crit * se_std

            fold_mean_raw = coef_raw.mean(axis=0)
            fold_sd_raw = coef_raw.std(axis=0, ddof=1)
            se_raw = fold_sd_raw / np.sqrt(KR)
            ci_low_raw = fold_mean_raw - t_crit * se_raw
            ci_high_raw = fold_mean_raw + t_crit * se_raw

            is_sig = (ci_low_std > 0) | (ci_high_std < 0)
            is_sig_fdr = _bh_fdr(p_val, q=0.05)

            df_out = pd.DataFrame({
                'feature': feature_names,
                'fold_mean_raw': fold_mean_raw,
                'fold_mean_std': fold_mean_std,
                'fold_cv_coef': fold_cv,
                't_statistic': t_stat,
                'p_value_t': p_val,
                'ci_low_raw_t': ci_low_raw,
                'ci_high_raw_t': ci_high_raw,
                'ci_low_std_t': ci_low_std,
                'ci_high_std_t': ci_high_std,
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

        --- _write_tier2_single (lines 3224-3282) ---

        New signature:
            def _write_tier2_single(coef_pool_internal, coef_pool_std, feature_names,
                                    out_dir, ci_level, feat_std_map,
                                    effect_type=None, contrast=None):

        New parameters:
        - coef_pool_internal: ndarray (B, P) in internal space (for raw derivation)
        - coef_pool_std: ndarray (B, P) in fully standardized space
        - feat_std_map: pd.Series of SD(X) per feature

        Replace body with:
            alpha_b = 1.0 - ci_level
            feat_std_arr = feat_std_map.reindex(feature_names).fillna(1.0).values

            # Fully standardized
            boot_mean_std = coef_pool_std.mean(axis=0)
            boot_ci_low_std = np.percentile(coef_pool_std, 100 * alpha_b / 2, axis=0)
            boot_ci_high_std = np.percentile(coef_pool_std, 100 * (1 - alpha_b / 2), axis=0)

            # Raw (Y-unit-scale)
            coef_pool_raw = coef_pool_internal / feat_std_arr[np.newaxis, :]
            boot_mean_raw = coef_pool_raw.mean(axis=0)
            boot_ci_low_raw = np.percentile(coef_pool_raw, 100 * alpha_b / 2, axis=0)
            boot_ci_high_raw = np.percentile(coef_pool_raw, 100 * (1 - alpha_b / 2), axis=0)

            # Scale-invariant
            B = coef_pool_std.shape[0]
            pd_val = np.maximum(
                (coef_pool_std > 0).sum(axis=0) / B,
                (coef_pool_std < 0).sum(axis=0) / B
            )
            p_value = np.clip(2 * (1 - pd_val), 0.0, 1.0)
            is_sig = (boot_ci_low_std > 0) | (boot_ci_high_std < 0)
            is_sig_fdr = _bh_fdr(p_value, q=0.05)

            df_result = pd.DataFrame({
                'feature': feature_names,
                'boot_mean_raw': boot_mean_raw,
                'boot_mean_std': boot_mean_std,
                'boot_ci_low_raw': boot_ci_low_raw,
                'boot_ci_high_raw': boot_ci_high_raw,
                'boot_ci_low_std': boot_ci_low_std,
                'boot_ci_high_std': boot_ci_high_std,
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

        --- run_bootstrap (lines 3285-3577): fold-index tracking + sd_y_divisors ---

        New signature: add sd_y_divisors parameter:
            def run_bootstrap(config, X_brain, Y, weights, subject_ids, X_cov, active_covs,
                              fold_models, apriori_map=None, moderator=None, sd_y_divisors=None):

        1. After the Parallel call (line 3347), build fold-index tracking:

            # Reconstruct fold index for each result (task_list is fold-sequential)
            fold_indices_all = []
            for fold_idx_track in range(len(fold_models)):
                fold_indices_all.extend([fold_idx_track] * n_per_fold_repeat)

            # Filter valid results, preserving fold indices
            valid_pairs = [
                (r, fi) for r, fi in zip(all_res, fold_indices_all) if r is not None
            ]
            valid_res = [pair[0] for pair in valid_pairs]
            valid_fold_indices = [pair[1] for pair in valid_pairs]

           (Replaces the existing `valid_res = [r for r in all_res if r is not None]` at line 3349.)

        2. Compute per-iteration SD(Y/Y*) divisor array for standardization:

            if sd_y_divisors is not None:
                sd_y_per_iter = np.array([sd_y_divisors[fi] for fi in valid_fold_indices])
            else:
                sd_y_per_iter = np.ones(len(valid_res))

        3. Single-output branch (not is_multi, line 3408+):

           After stacking coef_matrix (B, P):
            coef_matrix_internal = coef_matrix  # rename for clarity
            if sd_y_per_iter.ndim == 1:
                coef_matrix_std = coef_matrix_internal / sd_y_per_iter[:, np.newaxis]
            else:
                coef_matrix_std = coef_matrix_internal / sd_y_per_iter[:, np.newaxis]

            df_coef_internal = pd.DataFrame(coef_matrix_internal, columns=df_coef_cols)
            df_coef_std = pd.DataFrame(coef_matrix_std, columns=df_coef_cols)

           Interaction coefficient standardization (if has_moderator):
            coef_matrix_interaction_internal = coef_matrix_interaction  # existing
            coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]
            (For n_mod_cols > 1 with shape (B, n_mod_cols, P): / sd_y_per_iter[:, np.newaxis, np.newaxis])

           Thread to _compute_importance_report:
            _compute_importance_report(
                df_coef_internal, df_coef_std, all_feats_report, feat_std_map_report, config, ...
            )

           Thread to _write_tier2_single:
            _write_tier2_single(
                coef_matrix_internal, coef_matrix_std, all_feats_report,
                config['paths']['output_dir'], config['stats_params']['ci_level'],
                feat_std_map_report, effect_type=..., contrast=...
            )

           save_distributions: save coef_matrix_internal (unchanged naming: coef_dist).

        4. Multi-output branch (is_multi, line 3488+):

           After reference-differencing (C4) produces coef_array (B, K-1, P) or (B, K_tasks, P):
            coef_array_internal = coef_array
            # sd_y_per_iter: (B,) for scalar divisor, (B, K-1) or (B, K_tasks) for array
            if sd_y_per_iter.ndim == 1:
                coef_array_std = coef_array_internal / sd_y_per_iter[:, np.newaxis, np.newaxis]
            else:
                coef_array_std = coef_array_internal / sd_y_per_iter[:, :, np.newaxis]

           Per-task loop:
            df_coef_k_internal = pd.DataFrame(coef_array_internal[:, k, :], columns=df_coef_cols)
            df_coef_k_std = pd.DataFrame(coef_array_std[:, k, :], columns=df_coef_cols)
            _compute_importance_report(
                df_coef_k_internal, df_coef_k_std, all_feats_report, feat_std_map_report, config_k, ...
            )
            _write_tier2_single(
                coef_array_internal[:, k, :], coef_array_std[:, k, :],
                all_feats_report, out_dir_k, ci_level, feat_std_map_report, ...
            )

        --- _compute_importance_report (lines 3149-3165) ---

        New signature:
            def _compute_importance_report(df_coef_internal, df_coef_std, all_feats,
                                           feat_std_map, config, active_covs, reducer_full,
                                           X_brain, X_full, Y, weights, subject_ids,
                                           oof_linear_pred, oof_mod_main,
                                           moderator=None, interaction_report_df=None):

        Key change: replaces best_model with oof_linear_pred + oof_mod_main (C6 coupling).
        Calls _compute_importance_preamble with both df_coef_internal and df_coef_std.

        --- _report_standard (lines 3099-3146) ---

        New signature:
            def _report_standard(df_coef_internal, df_coef_std, all_feats, stats, config,
                                 active_covs, reducer_full, X_brain, X_full, Y, weights,
                                 subject_ids, oof_linear_pred, oof_mod_main,
                                 moderator=None, interaction_report_df=None):

        Replaces best_model with oof_linear_pred + oof_mod_main. Threads to
        calculate_visualization_data (see C6).

        --- _report_apriori (lines 3044-3097) ---

        New signature:
            def _report_apriori(df_coef_internal, df_coef_std, all_feats, stats, config,
                                active_covs, reducer_full, X_brain, X_full, Y, weights,
                                subject_ids, oof_linear_pred, oof_mod_main,
                                moderator=None, interaction_report_df=None):

        Cluster-level statistics (lines 3069-3081) must produce both raw and std columns.

        For std: cluster_boot_std = df_coef_std[cluster_feats_in].mean(axis=1)
        (SD(Y/Y*) is feature-invariant within a fold, so averaging features then dividing
        equals dividing then averaging; the pre-standardized df_coef_std handles this.)

        For raw: cluster_boot_raw = df_coef_raw[cluster_feats_in].mean(axis=1)
        where df_coef_raw = df_coef_internal.div(feat_std_map.reindex(df_coef_internal.columns).fillna(1.0), axis=1)
        (SD(X) is per-feature, so division MUST precede cluster averaging.)

        Update cluster_rows to include both std_coef_mean + raw_coef_mean + CIs for both.
        Add raw_coef_mean, raw_ci_low, raw_ci_high columns to the cluster report DataFrame.

        Replaces best_model with oof_linear_pred + oof_mod_main in the
        calculate_visualization_data call (line 3086-3088). See C6.

        --- run_tier1_inference (lines 2017-2173) ---

        Update all _write_tier1_report call sites to pass sd_y_divisors, feat_std_map,
        and fold_model_indices. Add sd_y_divisors and feat_std_map parameters to
        run_tier1_inference signature.

        feat_std_map: `X_brain.std(ddof=1).replace(0, 1.0)` (same as brain_std in run_bootstrap)
        fold_model_indices: `[fm['fold_idx'] for fm in fold_models]`

        Single-output branch (line 2057-2062):
            _write_tier1_report(coef_matrix, original_feature_names, out_dir, ci_level,
                                sd_y_divisors, feat_std_map, fold_model_indices,
                                effect_type=..., contrast=...)

        Interaction branch: same treatment for coef_matrix_interaction.

        Multi-output branch (line 2114-2123): per-task loop, use per-task divisor:
            For each task k:
            sd_y_divisors_k = {fi: sd_y_divisors[fi][k] if isinstance(sd_y_divisors[fi], np.ndarray)
                               else sd_y_divisors[fi]
                               for fi in sd_y_divisors}
            _write_tier1_report(coef_matrix_k, ..., sd_y_divisors_k, feat_std_map, fold_model_indices, ...)

        main() call site update (line 3856):
            run_tier1_inference(config, fold_models, X_brain, Y, active_covs, moderator,
                                sd_y_divisors=sd_y_divisors)

        main() run_bootstrap call site update (line 3859):
            run_bootstrap(config, X_brain, Y, weights, subj_ids, X_cov, active_covs,
                          fold_models, apriori_map, moderator, sd_y_divisors=sd_y_divisors)
      </spec>
      <dependencies>C2, C3, C4</dependencies>
      <risk>high - core reporting change affecting all output files with coefficient columns (report_feature_importance.csv, report_fold_ensemble_importance.csv, report_fold_bootstrap_ci.csv, report_cluster_importance.csv, report_interaction_importance.csv). Risk mitigated by: (1) scale-invariant quantities (pd, p_value, is_significant, is_significant_fdr) are unaffected; (2) raw_coef derivation formula is algebraically unchanged (internal / SD(X)); (3) all changes are deterministic arithmetic (division by positive scalars); (4) default sd_y_divisors=None falls through to ones, preserving current behavior in degenerate cases.</risk>
      <rollback>Revert all signature changes and function bodies. Restore single df_coef parameter in _compute_importance_preamble. Restore single coef_pool parameter in _write_tier2_single. Remove fold-index tracking from run_bootstrap.</rollback>
    </change>

    <change id="C6" priority="P0" source_item="T2: OOF ensemble visualization + interaction residualization">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>
        Replace fold-0 representative model in visualization with OOF ensemble predictions.
        New helper _compute_oof_visualization_data computes per-subject linear predictions
        from each subject's held-out fold model. Refactor calculate_visualization_data to
        accept pre-computed OOF quantities instead of a single best_model. Implement T2
        interaction residualization: extend lin_contrib to include brain main + moderator
        main + interaction (full conditional relationship). Switch coefficient source from
        std_coef_mean to raw_coef_mean for partial residuals.
      </description>
      <spec>
        --- New function _compute_oof_visualization_data (insert before calculate_visualization_data) ---

        def _compute_oof_visualization_data(fold_models, cv_data, X_brain, X_cov, config,
                                            active_covs, moderator=None):
            """Compute per-subject OOF linear predictions for visualization.

            For each subject, uses the pipeline from the fold where that subject was
            held out: that fold's reducer, scaler, and model coefficients.

            Parameters
            ----------
            fold_models : list of dict
            cv_data : dict with 'fold_assignments'
            X_brain : pd.DataFrame, original brain features
            X_cov : pd.DataFrame, covariates
            config : dict
            active_covs : list
            moderator : dict or None

            Returns
            -------
            oof_linear_pred : ndarray, shape (N,)
                Per-subject linear prediction from held-out fold model.
            oof_mod_main : ndarray, shape (N,) or None
                Per-subject moderator main-effect contribution (None if no moderator).
            X_full_repr : pd.DataFrame
                Representative assembled feature matrix for value lookup (uses fold-0
                reducer for column structure; values are from original data).
            """

        Implementation:
            N = len(X_brain)
            fold_assignments = cv_data['fold_assignments']
            oof_linear_pred = np.zeros(N)
            oof_mod_main = np.zeros(N) if moderator is not None else None
            is_pre = config['covariate_method'] == 'pre_regress'

            for fm in fold_models:
                k = fm['fold_idx']
                held_out = np.where(fold_assignments == k)[0]
                if len(held_out) == 0:
                    continue

                pipeline_k = fm['pipeline']
                reducer_k = fm['reducer']

                X_brain_ho = X_brain.iloc[held_out].reset_index(drop=True)
                if reducer_k is not None:
                    X_brain_red_ho = reducer_k.transform(X_brain_ho)
                else:
                    X_brain_red_ho = X_brain_ho

                moderator_coded = None
                interactions = None
                if moderator is not None:
                    mod_coded, _ = _code_moderator(
                        moderator['series'], moderator['type'],
                        np.arange(len(moderator['series']))
                    )
                    moderator_coded = mod_coded.iloc[held_out].reset_index(drop=True)
                    interactions = _construct_interactions(
                        X_brain_red_ho.reset_index(drop=True), moderator_coded
                    )

                parts = []
                if not X_cov.empty and not is_pre:
                    parts.append(X_cov.iloc[held_out].reset_index(drop=True))
                if moderator_coded is not None:
                    parts.append(moderator_coded)
                parts.append(X_brain_red_ho.reset_index(drop=True))
                if interactions is not None:
                    parts.append(interactions)
                X_full_ho = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

                scaler_k = pipeline_k.named_steps['scaler']
                model_k = pipeline_k.named_steps['model']
                X_scaled_ho = scaler_k.transform(X_full_ho)
                coeffs_k = _squeeze_binary_coef(model_k.coef_)
                if coeffs_k.ndim > 1:
                    coeffs_k = coeffs_k.mean(axis=0)
                intercept_k = getattr(model_k, 'intercept_', 0.0)
                if isinstance(intercept_k, np.ndarray):
                    intercept_k = intercept_k.mean()

                oof_linear_pred[held_out] = X_scaled_ho @ coeffs_k + intercept_k

                if moderator is not None:
                    n_covs_k = fm['n_covs']
                    n_mod_cols_k = fm['n_moderator_cols']
                    mod_indices = list(range(n_covs_k, n_covs_k + n_mod_cols_k))
                    mod_contrib = sum(
                        coeffs_k[idx] * X_scaled_ho[:, idx] for idx in mod_indices
                    )
                    oof_mod_main[held_out] = mod_contrib

            # Value-lookup X_full from fold-0 (for feature name resolution only)
            X_full_repr, _ = _reconstruct_x_full(
                fold_models, X_brain, X_cov, config, active_covs, moderator
            )

            return oof_linear_pred, oof_mod_main, X_full_repr

        --- calculate_visualization_data (lines 2841-2953) ---

        New signature:
            def calculate_visualization_data(config, X_full, Y, weights, subject_ids,
                                             oof_linear_pred, report_df, level,
                                             X_brain_raw=None, moderator=None,
                                             effect_type=None, main_report_df=None,
                                             oof_mod_main=None):

        Key changes:
        - best_model replaced by oof_linear_pred (ndarray, N)
        - Added main_report_df (for brain main-effect coefficient lookup in interaction case)
        - Added oof_mod_main (for moderator main-effect contribution in interaction case)

        1. Remove lines 2885-2890 (coeffs extraction, X_scaled computation, linear_pred_full):
           Replace with:
            linear_pred_full = pd.Series(oof_linear_pred, index=X_full.index)

        2. Replace lines 2933-2938 (lin_contrib computation):

            f_raw_coef = row['raw_coef_mean']
            f_centered = f_val_raw - f_val_raw.mean()

            if effect_type == 'interaction' and mod_coded_vals is not None:
                # Full conditional relationship: brain_main + mod_main + interaction

                # Brain main effect from main-effect report
                brain_raw_coef = 0.0
                if main_report_df is not None:
                    feat_col_main = ('component_id' if 'component_id' in main_report_df.columns else
                                     ('cluster_id' if 'cluster_id' in main_report_df.columns else 'feature'))
                    match = main_report_df[main_report_df[feat_col_main] == f_name]
                    if not match.empty:
                        brain_raw_coef = match['raw_coef_mean'].values[0]
                brain_main = brain_raw_coef * f_centered

                # Moderator main effect (pre-computed OOF)
                mod_main = oof_mod_main if oof_mod_main is not None else np.zeros(len(Y_arr))

                # Interaction
                int_contrib = f_raw_coef * f_centered * mod_coded_vals

                lin_contrib = brain_main + mod_main + int_contrib
            else:
                lin_contrib = f_raw_coef * f_centered

        3. cov_score computation (line 2939) unchanged:
            cov_score = linear_pred_full - lin_contrib

        Algebraic verification:
        - Before T1: std_coef * f_scaled = (internal / 1.0) * ((X - mean) / SD_X) = (internal / SD_X) * (X - mean) = raw_coef * (X - mean)
        - After T1: raw_coef * f_centered = (internal / SD_X) * (X - mean) = same result
        - The switch is necessary because std_coef is redefined to internal / SD(Y*), making
          std_coef * f_scaled produce values in SD(Y) units rather than Y units.

        --- run_bootstrap call site (lines 3385-3389) ---

        Replace:
            X_full_repr, fm0 = _reconstruct_x_full(fold_models, X_brain, X_cov, config, active_covs, moderator=moderator)
            best_model_repr = fm0['pipeline']
            reducer_full_repr = fm0['reducer']

        With:
            oof_linear_pred, oof_mod_main, X_full_repr = _compute_oof_visualization_data(
                fold_models, cv_data, X_brain, X_cov, config, active_covs, moderator
            )
            reducer_full_repr = fold_models[0]['reducer']

        NOTE: run_bootstrap needs cv_data passed in. Add cv_data parameter:
            def run_bootstrap(config, X_brain, Y, weights, subject_ids, X_cov, active_covs,
                              fold_models, apriori_map=None, moderator=None,
                              sd_y_divisors=None, cv_data=None):

        Thread oof_linear_pred, oof_mod_main to _compute_importance_report (which threads
        to _report_standard/_report_apriori, which thread to calculate_visualization_data).

        --- _report_standard interaction call site (lines 3141-3146) ---

        Pass main_report_df and oof_mod_main:
            calculate_visualization_data(
                config, X_full, Y, weights, subject_ids,
                oof_linear_pred,
                interaction_report_df, 'individual',
                X_brain if red_method != 'none' else None,
                moderator=moderator, effect_type='interaction',
                main_report_df=indiv_df,
                oof_mod_main=oof_mod_main
            )

        --- main() call site (line 3859) ---

            run_bootstrap(config, X_brain, Y, weights, subj_ids, X_cov, active_covs,
                          fold_models, apriori_map, moderator,
                          sd_y_divisors=sd_y_divisors, cv_data=cv_data)
      </spec>
      <dependencies>C2, C5</dependencies>
      <risk>medium - OOF visualization replaces the fold-0 approximation with a statistically principled approach. Risk from: (1) fold-specific reducers producing different column layouts, handled by per-fold assembly; (2) moderator coefficient extraction relies on consistent column ordering across folds, guaranteed by the canonical column assembly order [covariates, moderator_main, brain_reduced, interactions]; (3) _reconstruct_x_full is preserved for value-lookup only (not model prediction), maintaining backward compatibility for the feature-name resolution path.</risk>
      <rollback>Revert calculate_visualization_data to accept best_model. Restore _reconstruct_x_full usage for model prediction. Remove _compute_oof_visualization_data.</rollback>
    </change>

    <change id="C7" priority="P1" source_item="T1: documentation caveats">
      <file path="fmri-elastic-net.py" action="modify" />
      <file path="INPUT_SPECIFICATION.md" action="modify" />
      <file path="README.md" action="modify" />
      <description>Add documentation caveats for the fully standardized coefficient redefinition and document the new reference_class parameter, OOF visualization, and per-fold metrics.</description>
      <spec>
        fmri-elastic-net.py module docstring (after line 83, before "Written by"):
        Add caveats:
        - For classification, SD(Y*) is model-dependent (Menard, 2011): it changes as
          predictors are added or removed, unlike SD(Y) for regression.
        - SD(Y/Y*) divisors are computed per-fold from held-out predictions and applied
          at the fold level before pooling (fold-specific standardization prevents
          information leakage from training-set predictions).
        - Fully standardized coefficients remain unbounded in multiple regression (Friedman
          and Wall, 2005). The [-1, 1] bound applies only to simple bivariate regression.
        - Multi-class per-contrast latent-variable standardization is a principled extension
          of the binary-case derivation (Long, 1997; Menard, 2004) but lacks direct published
          validation for multinomial softmax models.

        Update existing caveat (lines 60-63):
        - raw_coef and std_coef are now independently derived from the internal
          (X-standardized) coefficient: raw_coef = internal / SD(X),
          std_coef = internal / SD(Y/Y*). The approximation caveat for reduction methods
          still applies (back-projected coefficient divided by original-feature SD is not
          equivalent to a standardized beta from direct regression on original features).

        INPUT_SPECIFICATION.md:
        - Document reference_class parameter under data_cols section
        - Document updated std_coef semantics (fully standardized, not X-standardized)
        - Add note about multi-class output structure (K-1 contrasts vs K classes)
        - Document new per-fold output files

        README.md:
        - Update coefficient interpretation section with fully standardized semantics
        - Document reference_class parameter
        - Note per-fold performance metrics
      </spec>
      <dependencies>C1, C2, C5</dependencies>
      <risk>low - text-only changes with no behavioral impact</risk>
      <rollback>Revert text changes.</rollback>
    </change>

    <change id="C8" priority="P0" source_item="T1: _get_task_labels update for reference-differenced multi-class">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Update _get_task_labels to return K-1 contrast labels (instead of K class labels) when multi-class reference-differencing is active. This function is called in run_bootstrap and run_tier1_inference before the reference-differencing step, so the update must be applied AFTER _reference_difference returns the contrast_labels.</description>
      <spec>
        _get_task_labels (lines 164-176):
        - No change to the function itself. It continues to return K class labels for
          multi-class classification.
        - In run_bootstrap and run_tier1_inference, after _reference_difference returns
          contrast_labels, the task_labels variable is reassigned:
            task_labels = contrast_labels
          This is already specified in C4's application sites. C8 is a cross-reference
          to ensure _get_task_labels is NOT modified (it still returns raw class labels
          for use by _reference_difference to identify the reference index).

        ACTUAL CHANGE for C8: ensure run_tier1_inference also updates task_labels
        after reference-differencing. In the multi-output branch (line 2116):
            task_labels = _get_task_labels(Y, config)
        Then after reference-differencing (C4 application site):
            if is_multiclass:
                coef_array, task_labels = _reference_difference(coef_array, class_labels, ref_class)
        The returned contrast_labels replace task_labels. This is consistent with
        run_bootstrap's C4 application site.
      </spec>
      <dependencies>C4</dependencies>
      <risk>low - ensures consistent label handling across Tier 1 and Tier 2</risk>
      <rollback>No code change beyond C4's application sites.</rollback>
    </change>
  </changes>

  <execution_order>C1, C2, C3, C4, C8, C5, C6, C7</execution_order>

  <notes>
    All decisions in this spec were explicitly resolved during the planning discussion:
    1. Tier 1 and Tier 2 reports include both raw AND fully standardized columns. No X-standardized-only
       columns remain. Resolved: user explicitly directed "no more X-only" and "both raw and fully
       standardized."
    2. Selection frequency remains in the K-class parameterization for multi-class classification.
       Reference-differencing is not applied because selection indicators reflect model sparsity (whether
       a coefficient is nonzero), which is a property of the parameterization, not the interpretation.
       Resolved: this was assumption 2 from the original spec, which the user confirmed stands.
    3. Visualization uses OOF ensemble (each subject's held-out fold model), not fold-0 representative.
       Resolved: user explicitly directed "the plotting should come from the model fold in which the
       subject was held out" and "this SHOULD be a fully OOF ensemble model."
    4. SD(Y/Y*) divisors are per-fold (including SD(Y) for regression). Applied at the fold level in
       bootstrap before pooling, and per-fold in Tier 1 before mean/CI computation.
       Resolved: user explicitly directed "per fold SD(Y*) values to prevent information leak" and
       confirmed "if coefficients are estimated per fold, then the SD's should also be per-fold."
    5. Per-fold performance metrics are written to model_performance_per_fold.csv alongside the
       existing aggregate metrics. Resolved: user confirmed "per-fold performance metrics should
       be added as a new item."
  </notes>
</implement_plan>
