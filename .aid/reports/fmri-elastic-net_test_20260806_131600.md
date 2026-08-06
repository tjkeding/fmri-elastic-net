<test_report>
  <meta project="fmri-elastic-net" mode="test" timestamp="2026-08-06T09:16:00-04:00" />
  <pre_design_run>
    <total>753</total>
    <passed>753</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No failing tests in the pre-design run; no dispositions required. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>28</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_moderator_integration_s17.py" test_count="28" coverage_target="Session 17 moderator-integration fixes (C1-C4 from implement build 20260806_125800): _reconstruct_x_full moderator column layout, subsample size diagnostic P_model, predict_ensemble moderator support, and calculate_visualization_data interaction-aware partial-dependence redesign." />
    </files_created>
    <design_rationale>
      The pre-design baseline was fully green (753/753), so the design phase added
      net-new coverage rather than repairing failing tests. Coverage gaps corresponded
      directly to the four fixes applied in the prior /implement build: (1)
      _reconstruct_x_full's moderator/interaction column assembly, verified both at
      the unit level (column order for continuous, nominal K=2, nominal K=3 moderators,
      with and without covariates) and via an integration test that replicates the
      reported crash scenario (the fold-0 pipeline's scaler.transform call on the
      reconstructed matrix); (2) the P_model subsample-size diagnostic, verified via
      the exact warning-message text for continuous and nominal K=3 moderators against
      a no-moderator control; (3) predict_ensemble's moderator validation (raises when
      trained-with/called-without mismatch in either direction) and successful
      prediction paths for continuous, nominal K=2, nominal K=3 moderators, plus
      new-data with an independent moderator series; (4) calculate_visualization_data's
      main-effect/interaction-effect decomposition, verified with two known-answer
      tests that manually recompute the documented partialling formula
      (f_weight * f_scaled for main, f_weight * f_scaled * M_coded for interaction)
      and assert exact numeric agreement, plus the K&gt;2 nominal skip-with-INFO-log
      path and the report-threading chain (_compute_importance_report -&gt;
      _report_standard / _report_apriori) verified via mock-based dispatch tests
      mirroring the existing test_report_dispatch.py pattern. Two end-to-end
      run_bootstrap tests close the loop by exercising the full pipeline (nested CV
      through bootstrap importance reporting) with nominal and continuous moderators,
      confirming report_interaction_importance.csv is produced without crash.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>781</total>
    <passed>781</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
    <!-- No outstanding action items. All four Session 17 fixes (C1-C4) are covered
         by passing tests with no regressions detected in the existing 753-test suite. -->
  </action_items>
</test_report>
