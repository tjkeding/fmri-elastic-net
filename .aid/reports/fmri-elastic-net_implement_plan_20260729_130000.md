<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-07-29T13:00:00-04:00" />
  <input_reports>
    <report path="fmri-elastic-net_brainstorm_20260729_120000.md" mode="brainstorm" key_items="4 P0 implement items, 2 P1 test items" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="Brainstorm P0 items 1 (Tier 1 multi-output interaction reporting)">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Extend the multi-output (else) branch of run_tier1_inference (lines 2086-2095) to report interaction effects per task, and fix the main-effect calls to pass effect_type/contrast when a moderator is present.</description>
      <spec>
        In run_tier1_inference, replace lines 2086-2095 (the entire else branch up to the _write_fold_diagnostics call) with the following logic:

        ```
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
                    # Stack interaction coefficients across folds
                    # Per-fold shape: (T, P) for continuous/binary, (T, n_mod_cols, P) for K>2 nominal
                    coef_array_interaction = np.stack(
                        [fm['coef_original_interaction'] for fm in fold_models], axis=0
                    )
                    # coef_array_interaction shape:
                    #   continuous/binary: (K_folds, T, P)
                    #   K>2 nominal: (K_folds, T, n_mod_cols, P)

                    for k, lbl in enumerate(task_labels):
                        out_dir_k = os.path.join(out_dir, f'task_{lbl}')
                        if n_mod_cols <= 1:
                            # Continuous or binary: single contrast per task
                            coef_matrix_interaction_k = coef_array_interaction[:, k, :]  # (K_folds, P)
                            _write_tier1_report(
                                coef_matrix_interaction_k, original_feature_names, out_dir_k, ci_level,
                                effect_type='interaction', contrast='moderator',
                            )
                        else:
                            # K>2 nominal: Hotelling omnibus + per-contrast t-tests per task
                            coef_slice_k = coef_array_interaction[:, k, :, :]  # (K_folds, n_mod_cols, P)
                            K_folds_local = coef_slice_k.shape[0]
                            P_local = coef_slice_k.shape[2]

                            # Per-feature Hotelling's T-squared omnibus test
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

                            # Per-contrast univariate t-tests
                            for j in range(n_mod_cols):
                                contrast_col = coef_slice_k[:, j, :]  # (K_folds, P)
                                _write_tier1_report(
                                    contrast_col, original_feature_names, out_dir_k, ci_level,
                                    effect_type='interaction', contrast=f'contrast_{j + 1}',
                                )
        ```

        Key details:
        - The main-effect _write_tier1_report call now passes effect_type='main', contrast='main' when moderator is not None (was None/None previously).
        - The interaction block is structurally identical to the single-output interaction block (lines 2038-2085), wrapped in a per-task iteration.
        - Hotelling omnibus rows are appended to the per-task report file (mode='a', header=False), matching the single-output convention.
        - Per-contrast t-tests use _write_tier1_report which handles append internally.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - structural replication of existing single-output logic; no new statistical methods</risk>
      <rollback>Revert run_tier1_inference else branch to original lines 2086-2095</rollback>
    </change>

    <change id="C2" priority="P0" source_item="Brainstorm P0 items 2 and 3 (Tier 2 multi-output interaction reporting + distribution saving)">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Extend the multi-output (else) branch of run_bootstrap (lines 3229-3257) to report interaction Tier 2 CIs per task, fix main-effect Tier 2 calls to pass effect_type/contrast when moderator is present, and add interaction distribution saving to both single-output and multi-output paths.</description>
      <spec>
        Two sub-changes within run_bootstrap:

        **C2a: Single-output interaction distribution saving (lines 3186-3191)**

        Replace the existing save_distributions block (lines 3186-3191) with:

        ```
        if config['stats_params'].get('save_distributions', True):
            save_kwargs = dict(
                coef_dist=coef_matrix,
                feature_names=np.array(df_coef_cols)
            )
            if has_moderator:
                coef_matrix_interaction = np.stack([r[1] for r in valid_res], axis=0)
                save_kwargs['coef_dist_interaction'] = coef_matrix_interaction
                n_mod_cols = fold_models[0]['n_moderator_cols']
                if n_mod_cols > 1:
                    save_kwargs['moderator_contrasts'] = np.array([f'contrast_{j+1}' for j in range(n_mod_cols)])
            np.savez_compressed(
                os.path.join(config['paths']['output_dir'], 'bootstrap_coef_distribution.npz'),
                **save_kwargs
            )
        ```

        Note: has_moderator is computed at line 3208. The save_distributions block executes at line 3186, which is BEFORE has_moderator is computed. Therefore, move the has_moderator computation (line 3208) to BEFORE the save_distributions block, placing it right after the coef_matrix materialization (after line 3184). Then reference has_moderator in both the save block and the existing Tier 2 interaction reporting block. The has_moderator line currently reads:
        `has_moderator = moderator is not None and fold_models[0].get('coef_original_interaction') is not None`

        After moving has_moderator up and modifying save_distributions, the interaction coef_matrix_interaction is now materialized inside the save block (conditionally). The existing Tier 2 interaction reporting block (lines 3209-3228) also materializes coef_matrix_interaction (line 3210). To avoid double materialization, extract coef_matrix_interaction once: materialize it right after has_moderator (guarded by `if has_moderator`), then reference it in both the save block and the Tier 2 reporting block. The n_mod_cols variable is also used in both blocks.

        Revised single-output path structure (lines 3182 onward):

        ```
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

            _compute_importance_report(
                df_coef, all_feats_report, feat_std_map_report, config, active_covs,
                reducer_full_repr, X_brain, X_full_repr, Y, weights, subject_ids, best_model_repr
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
        ```

        **C2b: Multi-output interaction Tier 2 reporting + distribution saving (lines 3229-3257)**

        Replace the entire else branch (lines 3229-3257) with:

        ```
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
                _compute_importance_report(
                    df_coef_k, all_feats_report, feat_std_map_report, config_k, active_covs,
                    reducer_full_repr, X_brain, X_full_repr, Y_k, weights, subject_ids, best_model_repr
                )
                # Tier 2 per-task main effects
                _write_tier2_single(
                    coef_array[:, k, :],
                    all_feats_report,
                    out_dir_k,
                    config['stats_params']['ci_level'],
                    effect_type='main' if moderator is not None else None,
                    contrast='main' if moderator is not None else None,
                )

            # Per-task interaction Tier 2 CIs
            if has_moderator:
                for k, lbl in enumerate(task_labels):
                    out_dir_k = os.path.join(config['paths']['output_dir'], f'task_{lbl}')
                    if n_mod_cols <= 1:
                        # Continuous/binary: coef_array_interaction shape (B, T, P)
                        _write_tier2_single(
                            coef_array_interaction[:, k, :],
                            all_feats_report,
                            out_dir_k,
                            config['stats_params']['ci_level'],
                            effect_type='interaction', contrast='moderator',
                        )
                    else:
                        # K>2 nominal: coef_array_interaction shape (B, T, n_mod_cols, P)
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
        ```

        Key details:
        - has_moderator and coef_array_interaction materialized once before both save and reporting blocks.
        - Interaction Tier 2 reporting uses a separate per-task loop after the main-effect loop to keep the logic clean.
        - Main-effect _write_tier2_single calls now pass effect_type='main', contrast='main' when moderator is not None.
        - coef_dist_interaction saved in the .npz with shape (B, T, P) or (B, T, n_mod_cols, P).
        - moderator_contrasts metadata array included when n_mod_cols > 1.
      </spec>
      <dependencies>None (independent of C1; both modify fmri-elastic-net.py but in non-overlapping regions)</dependencies>
      <risk>low - structural replication of existing patterns; save_distributions adds keys to existing .npz without changing existing keys</risk>
      <rollback>Revert run_bootstrap if-not-is_multi and else branches to original lines 3182-3257</rollback>
    </change>

    <change id="C3" priority="P0" source_item="Brainstorm P0 item 4 (selection frequency multi-output interaction reporting)">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Extend the multi-output (else) branch of run_selection_frequency (lines 2478-2494) to report per-task interaction selection frequencies and a root-level union aggregate, and fix main-effect rows to include effect_type/contrast when moderator is present.</description>
      <spec>
        Replace lines 2478-2494 (the entire else branch) with:

        ```
        else:
            # Multi-output: stack to (n_iter, K, P), write per-task files + union aggregate
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

            # Main-effect union aggregate: feature selected in >=1 task per iteration
            union_sel = (sel_array.sum(axis=1) > 0).astype(int)
            main_union_df = pd.DataFrame({
                'feature': out_feat_names,
                'selection_probability': union_sel.mean(axis=0)
            })
            if moderator is not None:
                main_union_df['effect_type'] = 'main'
                main_union_df['contrast'] = 'main'
            main_union_df.to_csv(os.path.join(out_dir, 'report_selection_frequency.csv'), index=False)

            # Interaction selection frequency (multi-output)
            if moderator is not None:
                interaction_results = [r[1] for r in all_results if r[1] is not None]
                if interaction_results:
                    n_mod_cols_report = fold_models[0]['n_moderator_cols']

                    if n_mod_cols_report <= 1:
                        # Continuous/binary: interaction indicators shape (T, P) per iteration
                        interaction_array = np.stack(interaction_results, axis=0)  # (n_iter, T, P)
                        # Per-task interaction selection frequency
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
                        # Root-level union aggregate: feature interaction selected in >=1 task
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
                        # K>2 nominal: interaction indicators shape (T, n_mod_cols, P) per iteration
                        interaction_array = np.stack(interaction_results, axis=0)  # (n_iter, T, n_mod_cols, P)
                        # Per-task interaction selection frequency
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
                        # Root-level union aggregate: collapse across tasks AND contrasts
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
        ```

        Key details:
        - Per-task main-effect DataFrames now include effect_type='main', contrast='main' when moderator is present.
        - Root-level main-effect union aggregate also includes these columns when moderator is present.
        - Per-task interaction selection frequencies are appended to the per-task report file (mode='a', header=False).
        - Root-level interaction union aggregate appended to root report (mode='a', header=False) with contrast='union'.
        - K>2 nominal union reshapes (n_iter, T, n_mod_cols, P) to (n_iter, T*n_mod_cols, P) before summing across axis 1.
      </spec>
      <dependencies>None (non-overlapping region of fmri-elastic-net.py from C1 and C2)</dependencies>
      <risk>low - structural replication of existing patterns with per-task iteration</risk>
      <rollback>Revert run_selection_frequency else branch to original lines 2478-2494</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3 (independent; any order is valid, but sequential avoids line-number drift within the same file)</execution_order>
  <notes>
    All three changes modify non-overlapping regions of fmri-elastic-net.py:
    - C1: run_tier1_inference else branch (lines ~2086-2095)
    - C2: run_bootstrap if-not-is_multi and else branches (lines ~3182-3257)
    - C3: run_selection_frequency else branch (lines ~2478-2494)

    No changes to config_template.yaml, helper functions, or any other file. The computational backbone (_boot_task, _subsample_iter, _strip_protected, _backproject_coef_original_space) already produces correct multi-output interaction shapes and requires no modification.
  </notes>
</implement_plan>
