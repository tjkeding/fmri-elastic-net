<brainstorm_report>
  <meta project="fmri-elastic-net" mode="brainstorm" timestamp="2026-07-29T12:00:00-04:00" />
  <context_files>
    <file path="fmri-elastic-net.py" relevance="Main pipeline script; contains all three reporting functions with the multi-output interaction gap (run_tier1_inference, run_bootstrap, run_selection_frequency) and the computational backbone (_boot_task, _subsample_iter, _strip_protected, _backproject_coef_original_space) that already handles multi-output interaction coefficients correctly." />
    <file path="config_template.yaml" relevance="Config schema for moderator_col, moderator_type, bootstrap_ci_method fields added in Session 14.5." />
  </context_files>
  <topics>
    <topic id="T4" title="Cross-cutting design: output file structure and FDR correction scope">
      <summary>Defines the output directory convention and FDR correction scope for multi-output + moderator reports. Governs T1, T2, and T3.</summary>
      <research>No external research required. Decision follows compositional application of two existing codebase conventions.</research>
      <approaches>
        <approach id="A1" label="Per-task subdirectories with interaction rows appended (Approach A)" feasibility="high" risk="low">
          <description>Each task_{lbl}/ directory gets the full moderator treatment: main-effect rows with effect_type='main', then interaction rows appended with effect_type='interaction' and per-contrast labels. Composes the per-task subdirectory structure (from multi-output) with the effect_type/contrast column convention (from moderator). No new structural patterns introduced.</description>
          <pros>Compositional: reuses both existing conventions without modification. No new file naming conventions. Consistent column structure within each per-task report file.</pros>
          <cons>None identified.</cons>
        </approach>
        <approach id="A2" label="Separate interaction report files" feasibility="high" risk="low">
          <description>Per-task interaction coefficients in separate files (e.g., task_{lbl}/report_fold_ensemble_importance_interaction.csv), keeping main and interaction effects in distinct files.</description>
          <pros>Main and interaction reports are independently parseable without filtering on effect_type.</pros>
          <cons>Introduces a new file-naming convention not used elsewhere in the pipeline. Fragments the report structure.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Approach A (per-task subdirectories with appended interaction rows). FDR correction: per-batch (each _write_tier1_report / _write_tier2_single call applies BH-FDR to its own block independently). This is the existing behavior for single-output moderator reports and is retained without change. Per-batch vs. pooled FDR has no universal directional advantage; the relationship depends on signal composition of each batch. Root-level aggregates: interaction union aggregate for selection frequency only (contrast='union', collapsing across both tasks and contrasts). No root-level Tier 1 or Tier 2 aggregates (matching existing multi-output convention). Per-task main-effect report calls must pass effect_type='main', contrast='main' when a moderator is present (currently pass None, which would create column misalignment when interaction rows are appended).
      </decision>
    </topic>

    <topic id="T1" title="Multi-output interaction Tier 1 inference reporting">
      <summary>Extends the else (multi-output) branch of run_tier1_inference to handle interaction effects, applying the same t-test and Hotelling T-squared logic per task.</summary>
      <research>No external research required. Extension reuses existing _write_tier1_report and _hotelling_t2 functions unchanged.</research>
      <approaches>
        <approach id="A1" label="Per-task interaction Tier 1 with same inference logic" feasibility="high" risk="low">
          <description>
            For multi-output with moderator, after writing per-task main-effect Tier 1 reports:

            1. Stack coef_original_interaction across folds. Shape: (K_folds, T, P) for continuous/binary moderator, or (K_folds, T, n_mod_cols, P) for K>2 nominal.

            2. Iterate over tasks (index k):
               - Continuous/binary (n_mod_cols &lt;= 1): extract (K_folds, P) at [:, k, :]. Pass to _write_tier1_report with effect_type='interaction', contrast='moderator', writing to task_{lbl}/.
               - K>2 nominal (n_mod_cols > 1): extract (K_folds, n_mod_cols, P) at [:, k, :, :]. Per-feature Hotelling T-squared omnibus (contrast='omnibus'), then per-contrast univariate t-tests (contrast=f'contrast_{j+1}'). Both append to task_{lbl}/report_fold_ensemble_importance.csv.

            3. Fix: per-task main-effect calls pass effect_type='main', contrast='main' when moderator is present.

            Shape verification (from run_nested_cv lines 1771-1787):
            - Multi-class + continuous/binary: coef_original_interaction = (C, P) per fold
            - Multi-class + K>2 nominal: coef_original_interaction = (C, n_mod_cols, P) per fold
            - Multi-task + continuous/binary: coef_original_interaction = (T, P) per fold
            - Multi-task + K>2 nominal: coef_original_interaction = (T, n_mod_cols, P) per fold
          </description>
          <pros>Structurally identical to existing single-output interaction Tier 1 logic, executed per-task. No new statistical methods or functions.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Per-task interaction Tier 1 inference using existing t-test (continuous/binary) and Hotelling T-squared + per-contrast t-tests (K>2 nominal). Per-task main-effect calls updated to pass effect_type='main', contrast='main' when moderator is present.
      </decision>
    </topic>

    <topic id="T2" title="Multi-output interaction Tier 2 bootstrap reporting">
      <summary>Extends the else (multi-output) branch of run_bootstrap to handle interaction CIs and save interaction bootstrap distributions for both single-output and multi-output paths.</summary>
      <research>No external research required. Extension reuses existing _write_tier2_single function unchanged.</research>
      <approaches>
        <approach id="A1" label="Per-task interaction Tier 2 with distribution saving" feasibility="high" risk="low">
          <description>
            For multi-output with moderator, after writing per-task main-effect Tier 2 reports:

            1. Stack interaction coefficients from valid bootstrap results. Shape: (B, T, P) for continuous/binary, or (B, T, n_mod_cols, P) for K>2 nominal.

            2. Iterate over tasks (index k), writing to task_{lbl}/:
               - Continuous/binary: extract (B, P) at [:, k, :]. Pass to _write_tier2_single with effect_type='interaction', contrast='moderator'.
               - K>2 nominal: extract (B, n_mod_cols, P) at [:, k, :, :]. L2 norm: sqrt(sum(slice**2, axis=1)) yields (B, P), written with contrast='L2_norm'. Per-contrast: slice[:, j, :] yields (B, P), written with contrast=f'contrast_{j+1}'.

            3. Fix: per-task main-effect Tier 2 calls pass effect_type='main', contrast='main' when moderator is present.

            4. Save interaction distributions: add coef_dist_interaction key to the existing bootstrap_coef_distribution.npz file, alongside moderator_contrasts metadata array. Apply to BOTH single-output and multi-output paths for completeness (user requirement: all relevant distributions saved for plotting and post-hoc analyses).
               - Single-output + continuous/binary moderator: coef_dist_interaction shape (B, P)
               - Single-output + K>2 nominal: coef_dist_interaction shape (B, n_mod_cols, P)
               - Multi-output + continuous/binary: coef_dist_interaction shape (B, T, P)
               - Multi-output + K>2 nominal: coef_dist_interaction shape (B, T, n_mod_cols, P)
          </description>
          <pros>Reuses existing _write_tier2_single. Interaction distributions saved for downstream analysis. Both single-output and multi-output paths receive distribution saving.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Per-task interaction Tier 2 CIs using existing _write_tier2_single (L2 norm + per-contrast for K>2). Interaction bootstrap distributions saved via coef_dist_interaction key in bootstrap_coef_distribution.npz for both single-output and multi-output paths. Per-task main-effect calls updated to pass effect_type='main', contrast='main' when moderator is present.
      </decision>
    </topic>

    <topic id="T3" title="Multi-output interaction selection frequency reporting">
      <summary>Extends the else (multi-output) branch of run_selection_frequency to handle interaction selection indicators and produce a root-level union aggregate.</summary>
      <research>No external research required. Extension reuses existing aggregation and reporting patterns.</research>
      <approaches>
        <approach id="A1" label="Per-task interaction selection frequency with union aggregate" feasibility="high" risk="low">
          <description>
            For multi-output with moderator, after writing per-task main-effect selection frequencies:

            1. Collect interaction indicators from all valid results. Stack to (n_iter, T, P) for continuous/binary, or (n_iter, T, n_mod_cols, P) for K>2 nominal.

            2. Per-task reporting (iterate over k), appending to task_{lbl}/report_selection_frequency.csv:
               - Continuous/binary: extract (n_iter, P) at [:, k, :]. Mean gives selection probability. Written with effect_type='interaction', contrast='moderator'.
               - K>2 nominal: extract (n_iter, n_mod_cols, P) at [:, k, :, :]. Mean along axis 0 gives (n_mod_cols, P). Per-contrast rows written with contrast=f'contrast_{j+1}'.

            3. Fix: per-task main-effect rows pass effect_type='main', contrast='main' when moderator is present.

            4. Root-level union aggregate: collapse across both tasks and moderator contrasts per iteration, then mean across iterations. Written to root report_selection_frequency.csv with effect_type='interaction', contrast='union'. Semantics: "was this feature's interaction with the moderator selected in any task for any contrast?"
               - Continuous/binary: (n_iter, T, P) -> per-iteration union across T -> (n_iter, P) -> mean -> (P,)
               - K>2 nominal: (n_iter, T, n_mod_cols, P) -> reshape to (n_iter, T*n_mod_cols, P) -> per-iteration union across axis 1 -> (n_iter, P) -> mean -> (P,)

            Shape verification: _subsample_iter already produces correct multi-output interaction indicator shapes (lines 2384-2416): (T, P) for continuous/binary, (T, n_mod_cols, P) for K>2 nominal. _backproject_coef_original_space handles 2D input (line 1292-1314).
          </description>
          <pros>Reuses existing aggregation patterns. Root-level union aggregate parallels main-effect convention. Full per-contrast resolution available in per-task directories.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Per-task interaction selection frequencies using existing aggregation patterns. Root-level union aggregate with contrast='union' collapsing across both tasks and contrasts. Per-task main-effect rows updated to pass effect_type='main', contrast='main' when moderator is present.
      </decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Extend run_tier1_inference else branch (line 2086): add per-task interaction Tier 1 inference (t-test for continuous/binary moderator, Hotelling T-squared omnibus + per-contrast t-tests for K>2 nominal). Fix per-task main-effect calls to pass effect_type='main', contrast='main' when moderator is present." />
    <item priority="P0" target_mode="implement" description="Extend run_bootstrap else branch (line 3229): add per-task interaction Tier 2 CIs (L2 norm + per-contrast for K>2 nominal). Fix per-task main-effect Tier 2 calls to pass effect_type='main', contrast='main' when moderator is present." />
    <item priority="P0" target_mode="implement" description="Add interaction bootstrap distribution saving (coef_dist_interaction key in bootstrap_coef_distribution.npz) for BOTH single-output path (lines 3186-3191) and multi-output path (lines 3233-3239). Include moderator_contrasts metadata array." />
    <item priority="P0" target_mode="implement" description="Extend run_selection_frequency else branch (line 2478): add per-task interaction selection frequencies and root-level union aggregate with contrast='union'. Fix per-task main-effect rows to pass effect_type='main', contrast='main' when moderator is present." />
    <item priority="P1" target_mode="test" description="Test multi-output interaction reporting: multi-task regression + continuous moderator, multi-task regression + K>2 nominal moderator, multi-class classification + continuous moderator, multi-class classification + K>2 nominal moderator. Verify Tier 1 report structure (effect_type/contrast columns, Hotelling omnibus rows), Tier 2 report structure (L2 norm + per-contrast CIs), selection frequency (per-task + root union aggregate), saved distributions (coef_dist_interaction key presence and shape), column alignment between main and interaction rows within per-task reports." />
    <item priority="P1" target_mode="test" description="Regression tests: verify single-output paths (with and without moderator) produce identical output to pre-change behavior. The main-effect effect_type='main' addition when moderator is present is a column addition, not a behavioral change, but must not break existing single-output interaction reporting." />
  </action_items>
  <next_steps>
    Proceed to /implement with this report as the spec. All four changes (P0 items) touch the same three functions (run_tier1_inference, run_bootstrap, run_selection_frequency) and follow the same structural pattern: lift the existing single-output interaction logic into the per-task loop in the multi-output branch. The interaction distribution saving (P0 item 3) also touches the single-output path. After implementation, run /test (P1 items), then proceed with the originally planned /cr on the full interaction modeling implementation (C1-C10 from Session 14.5 plus these extensions).
  </next_steps>
</brainstorm_report>
