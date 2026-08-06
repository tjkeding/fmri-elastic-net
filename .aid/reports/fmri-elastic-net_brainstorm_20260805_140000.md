<brainstorm_report>
  <meta project="fmri-elastic-net" mode="brainstorm" timestamp="2026-08-05T14:00:00-04:00" />
  <context_files>
    <file path="fmri-elastic-net.py" relevance="Main pipeline script; all 4 findings are in this file" />
    <file path="DEBUG_EXAMPLES/main_&lt;job_id&gt;.log" relevance="Example job log showing the ValueError crash at bootstrap reporting" />
    <file path="DEBUG_EXAMPLES/main_&lt;job_id&gt;.err" relevance="Example stderr: 100% SAGA convergence warnings across all bootstrap iterations" />
    <file path="DEBUG_EXAMPLES/pipeline (1).log" relevance="Example SLURM pipeline log showing interleaved main + permutation worker output and the crash" />
  </context_files>
  <topics>
    <topic id="T1" title="_reconstruct_x_full missing moderator and interaction columns">
      <summary>
        The function _reconstruct_x_full (line 3105-3136) builds a representative full-data feature
        matrix for visualization by transforming X_brain through the fold-0 reducer and optionally
        prepending covariates. It produces [covariates, brain_reduced]. But the fold-0 pipeline was
        fit during run_nested_cv on [covariates, moderator_main, brain_reduced, interactions]
        (line 1706). When calculate_visualization_data passes the narrower DataFrame through the
        pipeline's scaler at line 2879, sklearn raises a ValueError because the interaction columns
        (e.g., IC_10_x_moderator through IC_14_x_moderator) are missing.

        This is the root cause of the reported crash. The error occurs at the bootstrap importance
        reporting stage, after all bootstrap iterations complete successfully. Every moderator run
        crashes here regardless of reduction method (none, ica, cluster_pca, apriori).

        Downstream callers affected:
        - _report_standard -> calculate_visualization_data (line 3082-3087)
        - _report_apriori -> calculate_visualization_data (line 3047-3048)
        - Multi-output path in run_bootstrap (line 3427-3429)
      </summary>
      <research>
        No external research required. Finding verified by tracing the data flow from
        _reconstruct_x_full through _compute_importance_report to calculate_visualization_data
        and comparing the column set against the fold-0 pipeline's expected input shape.
        The assembly order in run_nested_cv (line 1706-1723) is the authoritative reference
        for the expected column layout.
      </research>
      <approaches>
        <approach id="A1" label="Add moderator param to _reconstruct_x_full" feasibility="high" risk="low">
          <description>
            Add a moderator parameter to _reconstruct_x_full. When present:
            1. Code the full-sample moderator via _code_moderator(moderator['series'], moderator['type'], train_idx)
               using the fold-0 training indices from fm0.
            2. Construct interactions via _construct_interactions(X_brain_red, moderator_coded).
            3. Assemble in the canonical column order: [covariates, moderator_main, brain_reduced, interactions].
            The caller (run_bootstrap) already has the moderator dict available and passes it through.
          </description>
          <pros>Deterministic fix. Single correct approach. Mirrors the assembly logic already used in
                run_nested_cv, _run_cv_fold_loop, _boot_task, and run_selection_frequency.</pros>
          <cons>None. The only alternative (skipping visualization) was rejected in favor of the
                interaction-aware redesign (T3).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Deterministic fix; no design alternatives. The function must produce the same column layout
        that the fold-0 pipeline was fit on.
      </decision>
    </topic>

    <topic id="T2" title="predict_ensemble missing moderator and interaction columns">
      <summary>
        The predict_ensemble utility function (line 3468-3520) assembles X_new as
        [covariates, brain_reduced] without moderator or interaction columns. Each fold's pipeline
        expects the full augmented matrix [covariates, moderator_main, brain_reduced, interactions]
        when a moderator was configured during training. pipe.predict(X_new) at line 3516 will crash
        with the same feature-name mismatch ValueError.

        This function is not called from main(); it is exposed as a published API for downstream
        prediction on held-out cohorts. The bug makes it unusable for any moderator-configured run.
      </summary>
      <research>
        No external research required. Same pattern as T1, verified by reading the function's
        assembly logic (line 3501-3515) and comparing against the fold pipeline's expected input.
      </research>
      <approaches>
        <approach id="A1" label="Add moderator param to predict_ensemble" feasibility="high" risk="low">
          <description>
            Add a moderator parameter (and the new-data moderator series). When present:
            1. Code the new-data moderator via _code_moderator.
            2. Construct interactions via _construct_interactions.
            3. Assemble the full matrix before passing to each fold's pipeline.
            The function signature gains a moderator parameter with a default of None for
            backward compatibility.
          </description>
          <pros>Deterministic fix. Consistent with the T1 fix pattern.</pros>
          <cons>None.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Deterministic fix; same pattern as T1. The utility function must produce the same column
        layout that each fold's pipeline was fit on.
      </decision>
    </topic>

    <topic id="T3" title="calculate_visualization_data interaction-aware partial-dependence redesign">
      <summary>
        Once T1 is fixed (the crash), calculate_visualization_data's partial-dependence logic still
        conflates main effects and interaction effects. The function computes
        linear_pred_full = X_scaled.dot(coeffs) + intercept, where coeffs spans all columns
        (covariates, moderator_main, brain, interactions). For each significant brain feature, it
        subtracts only that feature's main-effect contribution, leaving the correlated interaction
        contribution embedded in the "covariate score." This produces a visualization that mixes
        the brain feature's main effect with its interaction effect.

        The approved redesign produces two separate outputs that each isolate one effect type:
        main-effect visualization (existing CSV, enhanced) and interaction visualization (new CSV).
      </summary>
      <research>
        No external research required. The partial-dependence decomposition is a standard
        added-variable-plot technique. The extension to interaction models follows directly from
        the linear model structure: Y = beta_0 + sum(beta_brain_j * X_j) + beta_mod * M +
        sum(beta_int_j * X_j * M) + epsilon. Each effect type can be isolated by subtracting
        only that term from the full linear prediction.
      </research>
      <approaches>
        <approach id="A1" label="Log message only" feasibility="high" risk="low">
          <description>Add an INFO log message noting the approximation. No logic changes.</description>
          <pros>Simplest.</pros>
          <cons>Produces a visualization artifact that cannot be interpreted correctly for
                interaction models. Defensibility risk for publication-bound pipeline.</cons>
        </approach>
        <approach id="A2" label="Skip visualization when moderator present" feasibility="high" risk="low">
          <description>Return early with an INFO log when moderator is not None.</description>
          <pros>Avoids producing a misleading artifact.</pros>
          <cons>Removes functionality entirely.</cons>
        </approach>
        <approach id="A3" label="Decompose main and interaction effects separately" feasibility="high" risk="low">
          <description>
            Produce two separate CSVs with effect-type-specific partialling:

            Main-effect CSV (report_{level}_plotting.csv):
            - Triggered by features significant in the main-effect report_df.
            - Partialling: lin_contrib = beta_brain_f * f_scaled. Removes only the main-effect
              contribution of the plotted feature. Interaction contribution stays in background.
              y_val isolates the main effect.

            Interaction CSV (report_{level}_interaction_plotting.csv):
            - Triggered by features significant in the interaction report_df.
            - Partialling: lin_contrib = beta_int_f * f_scaled * M_coded. Removes only the
              interaction contribution of the plotted feature. Main-effect contribution stays
              in background. y_val isolates the interaction effect.

            Output columns (both CSVs): subject_id, outcome_raw, feature_name, y_axis_value,
            moderator_value. moderator_value is the raw value (level label for nominal, numeric
            for continuous). No binning; user decides stratification downstream.

            Call sites: main-effect call stays in _report_standard / _report_apriori (existing,
            enhanced with moderator parameter). Interaction call is new: callers pass the
            interaction report_df in a second invocation.

            Scope exclusions:
            - Multi-output Y: both CSVs skip with INFO log (existing early-return preserved).
            - Apriori network-level: interaction visualization at cluster level not produced
              (interactions are constructed in reduced space, not cluster space).
          </description>
          <pros>Scientifically correct. Each visualization isolates one effect type. Downstream
                user can stratify by moderator group for interaction plots. No binning decisions
                imposed by the pipeline.</pros>
          <cons>More implementation work than A1/A2. Requires a second call to
                calculate_visualization_data from the callers.</cons>
          <statistical_considerations>
            The partial-dependence decomposition for each effect type follows directly from
            the additive structure of the linear model. For main effects, subtracting
            beta_brain_f * f_scaled from the total prediction leaves the residual attributable
            to all other terms plus noise, and y_val = Y - cov_score recovers the main-effect
            signal. For interactions, subtracting beta_int_f * f_scaled * M_coded isolates the
            moderator-dependent component. The two decompositions are orthogonal: the
            main-effect contribution of feature f is constant across moderator levels, while
            the interaction contribution varies. This justifies separate visualizations.
          </statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A3">
        User approved approach A3. Main effects should show main effects independent of
        interactions; interactions should show interactions independent of main effects.
        Moderator stratification is left as a downstream plotting decision; the pipeline outputs
        the raw moderator_value per subject without binning.
      </decision>
    </topic>

    <topic id="T4" title="Subsample size diagnostic underestimates model dimensionality with interactions">
      <summary>
        The per-fold subsample size warning in run_nested_cv (line 1662-1671) uses P_reduced
        (brain features after reduction) to compute the recommended minimum subsample size via the
        3*P rule. When a moderator is configured, the actual model dimensionality is
        P_reduced + n_moderator_cols + P_reduced * n_moderator_cols. For a representative run
        (ICA-40, K=2 nominal moderator): reported P_reduced = 40 (minimum = 120), actual
        P_model = 81 (minimum = 243). The diagnostic underreports the severity of the
        sample-size concern.

        The warning is emitted at line 1666, BEFORE interaction construction (line 1697-1704).
        The fix should pre-compute the effective P_model from the moderator configuration.
      </summary>
      <research>
        No external research required. The 3*P threshold is a standard rule-of-thumb for the
        minimum ratio of observations to parameters in regularized regression (ensuring
        subsample viability for bootstrap and selection frequency). The threshold should reflect
        the total number of parameters being estimated, including moderator and interaction terms.
      </research>
      <approaches>
        <approach id="A1" label="Pre-compute P_model including interactions" feasibility="high" risk="low">
          <description>
            After determining P_reduced and before the warning check, compute:
            P_model = P_reduced + n_moderator_cols + P_reduced * n_moderator_cols
            (when moderator is configured; P_model = P_reduced otherwise).
            Use P_model in the 3*P threshold calculation and in the warning message.
            n_moderator_cols is derivable from the moderator dict: K-1 for nominal (K = number
            of unique levels), 1 for continuous.
          </description>
          <pros>Accurate diagnostic. Does not require reordering code (no need to move the
                warning after interaction construction). Minimal change.</pros>
          <cons>None.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Deterministic fix. The diagnostic should reflect the actual model dimensionality.
      </decision>
    </topic>
  </topics>

  <audit_summary>
    <verified_correct>
      The following functions were verified as correctly handling moderator data:
      - run_nested_cv: constructs interactions within the fold CV loop (line 1703-1704)
      - _run_cv_fold_loop (permutation tests): constructs interactions within the fold CV loop (line 2217-2218)
      - _boot_task: constructs interactions within each bootstrap iteration (line 2733)
      - run_selection_frequency: constructs interactions within each subsample iteration (line 2408)
      - run_tier1_inference: operates on pre-back-projected coef_original from fold_models; never passes data through the pipeline scaler
      - _partial_ridge_refit: receives pre-assembled X_boot_transformed from _boot_task (line 2798)
      - _strip_protected: correctly handles moderator column count in the coefficient split
      - _code_moderator / _construct_interactions: standalone helpers, no integration issue
      - create_model_and_param_dist / ModeratorScaler: correctly places moderator indices in pipeline
    </verified_correct>
    <root_cause_pattern>
      The interaction modeling implementation (Sessions 14-15) correctly handled every code path
      that operates within a CV, bootstrap, or subsample loop. Two paths that operate outside
      those loops (reconstructing the feature matrix from fold-0 artifacts for descriptive and
      visualization purposes) were not updated. One diagnostic (subsample size) was not updated
      to reflect the expanded model dimensionality. The test suite tested component functions and
      inference pathways but lacked end-to-end integration tests exercising the visualization and
      reporting branches with a moderator configured. The Session 15 CR focused on
      coefficient-level correctness (sign errors, mask dimensionality, degeneracy guards) and did
      not audit the visualization data flow or diagnostic accuracy.
    </root_cause_pattern>
    <log_non_bugs>
      The following items from the example logs were verified as expected behavior, not pipeline
      defects:
      - 100% convergence warning rate (4955/4955 bootstrap iterations): SAGA solver with
        max_iter=5000 struggling with P_model=81 features and subsample size=67. Data/configuration
        characteristic, not a code defect. Pipeline correctly reports the warnings.
      - Bootstrap failure rate (45/5000 = 0.9%): within the pipeline's acceptable threshold.
      - Firth fallback rate (86/4955 = 1.7%): expected for classification with small samples.
      - N:P diagnostic (P_brain=128, N:P=1.16): intentionally computed on raw brain features as a
        pre-reduction data-quality indicator, not meant to reflect post-reduction model dimensionality.
      - Heredity diagnostic (22 features): expected informational diagnostic.
    </log_non_bugs>
  </audit_summary>

  <action_items>
    <item priority="P0" target_mode="implement" description="T1: Add moderator parameter to _reconstruct_x_full; code moderator and construct interactions when present; assemble in canonical column order [covariates, moderator_main, brain_reduced, interactions]" />
    <item priority="P0" target_mode="implement" description="T2: Add moderator parameter to predict_ensemble; code new-data moderator and construct interactions when present; assemble full feature matrix before each fold's pipeline.predict()" />
    <item priority="P0" target_mode="implement" description="T3: Redesign calculate_visualization_data for interaction-aware partial-dependence. Main-effect CSV: partial out only beta_brain_f * f_scaled. New interaction CSV (report_{level}_interaction_plotting.csv): partial out only beta_int_f * f_scaled * M_coded. Both CSVs include moderator_value column (raw, no binning). New call site in _report_standard / _report_apriori for interaction report_df. Multi-output Y and apriori cluster-level interaction visualization excluded." />
    <item priority="P0" target_mode="implement" description="T4: Update subsample size diagnostic (line 1662-1671) to use P_model = P_reduced + n_moderator_cols + P_reduced * n_moderator_cols when moderator is configured" />
    <item priority="P0" target_mode="test" description="End-to-end integration test: run full pipeline with moderator_col configured through run_bootstrap to calculate_visualization_data. Cover all 4 reduction methods (none, ica, cluster_pca, apriori) x moderator. Verify no crash and correct output CSV column sets." />
    <item priority="P0" target_mode="test" description="Integration test for predict_ensemble with moderator: verify correct feature matrix assembly and successful prediction on new data." />
    <item priority="P1" target_mode="test" description="Unit test for T3 visualization outputs: verify main-effect CSV contains only main-effect partialling, interaction CSV contains only interaction partialling, moderator_value column present in both." />
    <item priority="P1" target_mode="test" description="Unit test for T4 diagnostic: verify warning message reports P_model (not P_reduced) when moderator is configured." />
  </action_items>

  <next_steps>
    Proceed to /implement for all 4 P0 code changes (T1-T4), then /test for the 4 P0/P1 test
    items. The implementation priority order is T1 (unblocks the reported crash), T4 (simple
    diagnostic fix), T2 (utility function fix), T3 (visualization redesign, largest change).
  </next_steps>
</brainstorm_report>
