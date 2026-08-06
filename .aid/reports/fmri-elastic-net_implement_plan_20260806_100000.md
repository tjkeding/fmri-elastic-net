<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-08-06T10:00:00-04:00" />
  <input_reports>
    <report path="brainstorm_history/fmri-elastic-net_brainstorm_20260805_140000.md" mode="brainstorm" key_items="4" />
  </input_reports>

  <plan_decisions>
    <!-- The brainstorm T1 description references "fold-0 training indices from fm0,"
         but fold_models records do not store training indices. Using full-sample
         indices (np.arange(N)) is justified: _reconstruct_x_full is documented as
         a descriptive approximation (fold-0 reducer on full data), and moderator
         coding with full-sample indices is consistent with that approximation.
         For continuous moderators, the pipeline's StandardScaler re-centers anyway.
         For nominal moderators, full-sample levels match training levels for any
         reasonable K-fold split. -->
    <!-- C1 changes X_full_repr column set, which requires fixing the reporting
         path in run_bootstrap: the else branch at line 3316-3320 derives
         all_feats_report from X_full_repr.columns, but df_coef is brain-only.
         With moderator columns in X_full_repr, the column count mismatches.
         Fix: use brain-only features unconditionally when moderator is present. -->
    <!-- T3 interaction visualization for K>2 nominal (n_mod_cols > 1) is skipped
         with an INFO log. The partialling formula requires per-contrast
         interaction coefficients, but the interaction report_df uses L2 norms
         (which are non-negative and lack directionality). K=2 and continuous
         moderators cover the common use case. -->
  </plan_decisions>

  <changes>
    <change id="C1" priority="P0" source_item="T1: _reconstruct_x_full missing moderator and interaction columns">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Fix _reconstruct_x_full to produce the canonical column layout
        [covariates, moderator_main, brain_reduced, interactions] when a moderator is
        configured. Update the run_bootstrap call site to pass the moderator parameter.
        Fix the reporting path in run_bootstrap to use brain-only feature names and
        standard deviations (not X_full_repr columns) when moderator is present, since
        df_coef is brain-only after _strip_protected.</description>
      <spec>
        1. _reconstruct_x_full (line 3105):
           - Add parameter: moderator=None (dict with 'series', 'type', or None)
           - After X_brain_red is computed (line 3124-3127), add moderator block:
             ```
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
             ```
           - Replace the existing assembly block (lines 3129-3134) with
             parts-based assembly mirroring run_nested_cv (line 1706-1723):
             ```
             parts = []
             if not X_cov.empty and not is_pre:
                 parts.append(X_cov.reset_index(drop=True))
             if moderator_coded is not None:
                 parts.append(moderator_coded)
             parts.append(X_brain_red.reset_index(drop=True))
             if interactions is not None:
                 parts.append(interactions)
             X_full_repr = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
             ```
           - Update docstring: note that when moderator is provided, the output
             includes moderator and interaction columns in canonical order.

        2. run_bootstrap call site (line 3302):
           - Change to: _reconstruct_x_full(fold_models, X_brain, X_cov, config,
             active_covs, moderator=moderator)

        3. run_bootstrap reporting path (lines 3307-3326):
           - Replace the red_method=='none' branch to always use brain-only features:
             ```
             if red_method == 'none':
                 brain_std = X_brain.std(ddof=1).replace(0, 1.0)
                 feat_std_map_report = brain_std
                 all_feats_report = original_feature_names
                 df_coef_cols = original_feature_names
             else:
                 brain_std = X_brain.std(ddof=1).replace(0, 1.0)
                 feat_std_map_report = brain_std
                 all_feats_report = original_feature_names
                 df_coef_cols = original_feature_names
             ```
             This simplifies to an unconditional block (both branches identical).
             The red_method branch is removed entirely:
             ```
             brain_std = X_brain.std(ddof=1).replace(0, 1.0)
             feat_std_map_report = brain_std
             all_feats_report = original_feature_names
             df_coef_cols = original_feature_names
             ```
             Justification: df_coef is always brain-only after _strip_protected,
             regardless of reduction method. The prior branching was vestigial from
             before the fold-wise ensemble architecture stripped covariates. This
             change is correct for both moderator and non-moderator runs.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - Deterministic fix. Parts-based assembly mirrors run_nested_cv. The
        reporting path simplification removes an unnecessary branch that only
        coincidentally worked for non-moderator runs.</risk>
      <rollback>Revert _reconstruct_x_full signature and body, revert run_bootstrap
        call site and reporting path.</rollback>
    </change>

    <change id="C2" priority="P0" source_item="T4: Subsample size diagnostic underestimates model dimensionality with interactions">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Update the per-fold subsample size diagnostic in run_nested_cv to
        use P_model (total model dimensionality including moderator and interaction
        columns) instead of P_reduced (brain features only) when a moderator is
        configured.</description>
      <spec>
        1. run_nested_cv, after P_reduced computation (line 1664), before the warning
           check (line 1666), add:
           ```
           if moderator is not None:
               if moderator['type'] == 'continuous':
                   n_mod_cols_diag = 1
               else:
                   n_mod_cols_diag = len(set(moderator['series'].iloc[tr])) - 1
               P_model = P_reduced + n_mod_cols_diag + P_reduced * n_mod_cols_diag
           else:
               P_model = P_reduced
           ```
        2. Update the warning check (line 1666) to use P_model:
           ```
           if half_n < max(3 * P_model, 30):
               logging.warning(
                   f"Fold {fold_idx}: 50%% subsample size ({half_n}) is below the "
                   f"recommended minimum (max(3*P_model={3*P_model}, 30)="
                   f"{max(3*P_model, 30)}). Selection frequency and bootstrap CIs "
                   f"may be unreliable for this fold."
               )
           ```
           Note: "P_model" in the format string (replacing "P_reduced") accurately
           communicates what the threshold represents.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - Diagnostic message only, no data flow change. The warning threshold
        becomes more conservative when interactions are present, which is correct.</risk>
      <rollback>Revert the P_model computation and warning message.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="T2: predict_ensemble missing moderator and interaction columns">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Fix predict_ensemble to construct the full feature matrix including
        moderator and interaction columns when a moderator is configured. The coding
        is computed once (not per-fold) using the full new-data sample, with a
        validation check against the expected moderator column count from fold_models.
      </description>
      <spec>
        1. predict_ensemble (line 3468):
           - Add parameter: moderator=None (dict with 'series' and 'type', or None)
           - Update docstring to document the new parameter and its structure.

        2. Before the fold loop (after line 3500), add moderator coding:
           ```
           moderator_coded = None
           interactions_need = moderator is not None
           if interactions_need:
               n_mod_expected = fold_models[0]['n_moderator_cols']
               mod_coded, _ = _code_moderator(
                   moderator['series'], moderator['type'],
                   np.arange(len(moderator['series']))
               )
               moderator_coded = mod_coded.reset_index(drop=True)
               if moderator_coded.shape[1] != n_mod_expected:
                   raise ValueError(
                       f"Moderator coding produced {moderator_coded.shape[1]} columns "
                       f"but fold models expect {n_mod_expected}. Ensure the new data's "
                       f"moderator levels match the training data."
                   )
           ```

        3. Inside the fold loop (after X_br_red is computed at line 3506-3508), add
           interaction construction and update the assembly block:
           ```
           parts = []
           if (X_cov_new is not None and not X_cov_new.empty and not is_pre):
               parts.append(X_cov_new.reset_index(drop=True))
           if moderator_coded is not None:
               parts.append(moderator_coded)
           parts.append(X_br_red.reset_index(drop=True))
           if interactions_need:
               int_new = _construct_interactions(
                   X_br_red.reset_index(drop=True), moderator_coded
               )
               parts.append(int_new)
           X_new = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
           ```
           This replaces the existing if/else block at lines 3509-3515.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - Same assembly pattern as C1 and run_nested_cv. Backward compatible
        (moderator defaults to None). The validation check catches level mismatches
        early with a clear error message.</risk>
      <rollback>Revert predict_ensemble signature and body.</rollback>
    </change>

    <change id="C4" priority="P0" source_item="T3: calculate_visualization_data interaction-aware partial-dependence redesign">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Redesign calculate_visualization_data for interaction-aware
        partial-dependence with two separate output CSVs (main effect and interaction).
        Thread moderator and interaction report data through the reporting chain:
        run_bootstrap computes interaction importance stats, passes them through
        _compute_importance_report to _report_standard/_report_apriori, which write
        the interaction importance report and call calculate_visualization_data for
        interaction visualization. K>2 nominal interaction visualization is skipped.
        Multi-output Y interaction visualization is skipped (existing guard).
      </description>
      <spec>
        --- Part A: calculate_visualization_data (line 2835) ---

        1. Add parameters: moderator=None, effect_type=None
           - moderator: dict with 'series' (pd.Series), 'type' (str), or None
           - effect_type: None, 'main', or 'interaction'
           - Update docstring accordingly.

        2. After the multi-output Y early return (line 2884-2889), add K>2 guard
           for interaction visualization:
           ```
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
           ```

        3. In the per-feature loop (line 2893-2912), modify the partialling logic:
           - For effect_type is None or 'main': current partialling unchanged
             (lin_contrib = f_weight * f_val_scaled). This is correct because
             f_weight from the main-effect report_df IS the brain-only coefficient.
           - For effect_type == 'interaction':
             ```
             M_values = mod_coded.values.flatten()  # (N,) for n_mod_cols==1
             lin_contrib = f_weight * f_val_scaled * M_values
             ```
             Where f_weight = row['std_coef_mean'] from the interaction report_df,
             and M_values are the coded moderator values for each subject.

        4. Add moderator_value column when moderator is present:
           ```
           mod_raw = moderator['series'].values if moderator is not None else None
           ```
           In the per-subject inner loop, add:
           ```
           if mod_raw is not None:
               record['moderator_value'] = mod_raw[i]
           ```

        5. Change the output filename for interaction visualization:
           - For effect_type == 'interaction':
             output to report_{level}_interaction_plotting.csv
           - For effect_type is None or 'main':
             output to report_{level}_plotting.csv (unchanged)

        --- Part B: _report_standard (line 3054) ---

        1. Add parameters: moderator=None, interaction_report_df=None
        2. Pass moderator and effect_type to calculate_visualization_data for the
           main-effect call:
           ```
           calculate_visualization_data(config, X_full, Y, weights, subject_ids,
               best_model, indiv_df, 'individual', None,
               moderator=moderator,
               effect_type='main' if moderator is not None else None)
           ```
           (and similarly for the reduced-method branch with X_brain_raw)
        3. After the main-effect visualization, add interaction block:
           ```
           if interaction_report_df is not None:
               interaction_report_df.to_csv(
                   os.path.join(out_dir, 'report_interaction_importance.csv'),
                   index=False
               )
               calculate_visualization_data(
                   config, X_full, Y, weights, subject_ids, best_model,
                   interaction_report_df, 'individual',
                   X_brain if red_method != 'none' else None,
                   moderator=moderator, effect_type='interaction'
               )
           ```

        --- Part C: _report_apriori (line 3005) ---

        1. Add parameters: moderator=None, interaction_report_df=None
        2. Pass moderator and effect_type to calculate_visualization_data for the
           cluster-level main-effect call:
           ```
           calculate_visualization_data(config, X_full, Y, weights, subject_ids,
               best_model, net_rep, 'cluster', X_brain,
               moderator=moderator,
               effect_type='main' if moderator is not None else None)
           ```
        3. After the cluster-level main-effect visualization, write the interaction
           importance report (individual-level) but skip interaction visualization
           (per brainstorm scope exclusion):
           ```
           if interaction_report_df is not None:
               interaction_report_df.to_csv(
                   os.path.join(out_dir, 'report_interaction_importance.csv'),
                   index=False
               )
           ```
           No interaction visualization call for apriori.

        --- Part D: _compute_importance_report (line 3090) ---

        1. Add parameters: moderator=None, interaction_report_df=None
        2. Pass through to branch functions via keyword args:
           ```
           if red_method == 'apriori':
               _report_apriori(*branch_args, moderator=moderator,
                   interaction_report_df=interaction_report_df)
           else:
               _report_standard(*branch_args, moderator=moderator,
                   interaction_report_df=interaction_report_df)
           ```

        --- Part E: run_bootstrap interaction importance computation ---

        After the main-effect _compute_importance_report call (line 3363-3366 for
        single-output, line 3427-3430 for multi-output), compute interaction
        importance stats and thread them through:

        Single-output (has_moderator=True), insert before the existing
        _compute_importance_report call at line 3363:
        ```
        interaction_report_df = None
        if has_moderator:
            if n_mod_cols <= 1:
                df_coef_interaction = pd.DataFrame(
                    coef_matrix_interaction, columns=df_coef_cols
                )
            else:
                l2_norms = np.sqrt(np.sum(
                    coef_matrix_interaction ** 2, axis=1
                ))
                df_coef_interaction = pd.DataFrame(
                    l2_norms, columns=df_coef_cols
                )
            int_stats = _compute_importance_preamble(
                df_coef_interaction, all_feats_report,
                feat_std_map_report, config
            )
            interaction_report_df = _build_individual_report_df(
                all_feats_report, int_stats
            )
        ```
        Then update the _compute_importance_report call to pass through:
        ```
        _compute_importance_report(
            df_coef, all_feats_report, feat_std_map_report, config,
            active_covs, reducer_full_repr, X_brain, X_full_repr, Y,
            weights, subject_ids, best_model_repr,
            moderator=moderator,
            interaction_report_df=interaction_report_df
        )
        ```

        Multi-output (has_moderator=True), inside the per-task loop at line 3421:
        ```
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
        ```
        Then update the per-task _compute_importance_report call:
        ```
        _compute_importance_report(
            df_coef_k, all_feats_report, feat_std_map_report, config_k,
            active_covs, reducer_full_repr, X_brain, X_full_repr, Y_k,
            weights, subject_ids, best_model_repr,
            moderator=moderator,
            interaction_report_df=interaction_report_df_k
        )
        ```
      </spec>
      <dependencies>C1 (X_full_repr must include moderator and interaction columns
        for calculate_visualization_data's linear_pred_full computation to be
        correct with the fold-0 pipeline's scaler and coefficients)</dependencies>
      <risk>medium - Largest change; spans 5 functions across the reporting and
        visualization pipeline. The interaction partialling formula is
        straightforward (f_weight * f_scaled * M_coded), but the data flow through
        the report chain has many touch points. Per-function verification is
        critical. K>2 interaction visualization is intentionally deferred.</risk>
      <rollback>Revert all 5 modified function signatures and bodies. No new files
        created; only existing functions modified.</rollback>
    </change>
  </changes>

  <execution_order>C1, C2, C3, C4</execution_order>
  <!-- C1 first (unblocks crash, prerequisite for C4).
       C2 and C3 are independent of each other and of C4; placed before C4 for
       simplicity (single-file, no parallel dispatch needed).
       C4 last (depends on C1's X_full_repr fix). -->

  <new_output_files>
    <file>report_interaction_importance.csv - Per-feature interaction importance
      statistics (std_coef_mean, CIs, pd, significance). Written by _report_standard
      and _report_apriori when moderator is configured. For K>2 nominal, statistics
      are based on L2 norm across contrasts.</file>
    <file>report_{level}_interaction_plotting.csv - Subject-level interaction
      partial-dependence visualization data. Written by calculate_visualization_data
      when effect_type='interaction'. Columns: subject_id, outcome_raw, feature_name,
      y_axis_value, moderator_value. Produced for K=2/continuous moderators only;
      K>2 skipped with INFO log.</file>
  </new_output_files>
</implement_plan>
