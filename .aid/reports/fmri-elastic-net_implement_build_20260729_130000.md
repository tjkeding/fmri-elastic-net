<implement_report>
  <meta project="fmri-elastic-net" mode="implement" submodule="build" timestamp="2026-07-29T14:30:00-04:00" />
  <spec_ref>fmri-elastic-net_implement_plan_20260729_130000.md</spec_ref>
  <changes_applied>
    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="50" />
      </files_modified>
      <notes>Per-task interaction Tier 1 inference in run_tier1_inference else branch. Continuous/binary moderator: per-task t-test via _write_tier1_report with effect_type='interaction', contrast='moderator'. K>2 nominal moderator: per-feature Hotelling T-squared omnibus rows (contrast='omnibus') appended via mode='a', plus per-contrast univariate t-tests. Main-effect calls now pass effect_type='main', contrast='main' when moderator is present. No deviations from spec.</notes>
    </change>
    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="80" />
      </files_modified>
      <notes>Two sub-changes in run_bootstrap. C2a (single-output branch): moved has_moderator check before save_distributions block; materialized coef_matrix_interaction and n_mod_cols once for reuse in both .npz saving and Tier 2 reporting; added coef_dist_interaction key and moderator_contrasts metadata to bootstrap_coef_distribution.npz via save_kwargs pattern. C2b (multi-output branch): added has_moderator, coef_array_interaction, n_mod_cols materialization; interaction distributions saved in .npz; per-task main-effect Tier 2 calls now pass effect_type='main', contrast='main' when moderator present; per-task interaction Tier 2 CIs added (continuous/binary: single contrast, K>2 nominal: L2 norm + per-contrast). No deviations from spec.</notes>
    </change>
    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="84" />
      </files_modified>
      <notes>Extended run_selection_frequency else branch. Per-task main-effect DataFrames include effect_type='main', contrast='main' when moderator present. Root-level main-effect union aggregate includes same columns. Continuous/binary moderator: per-task interaction selection frequencies (contrast='moderator') appended to per-task reports; root-level interaction union aggregate (contrast='union') appended to root report. K>2 nominal moderator: per-task per-contrast interaction selection frequencies (contrast=f'contrast_{j+1}') appended; root-level union aggregate collapses across tasks AND contrasts via reshape to (n_iter, T*n_mod_cols, P) before per-iteration union. No deviations from spec.</notes>
    </change>
  </changes_applied>
  <summary>
    <total_changes>3</total_changes>
    <completed>3</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <next_steps>Recommended: run /test to validate all changes. The P1 test items from the brainstorm report specify 6 test scenarios: multi-task regression + continuous moderator, multi-task regression + K>2 nominal moderator, multi-class classification + continuous moderator, multi-class classification + K>2 nominal moderator, plus regression tests for single-output paths.</next_steps>
</implement_report>
