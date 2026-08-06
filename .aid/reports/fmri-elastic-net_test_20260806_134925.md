<test_report>
  <meta project="fmri-elastic-net" mode="test" timestamp="2026-08-06T09:49:25-04:00" />
  <pre_design_run>
    <total>781</total>
    <passed>781</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No failing tests in the pre-design run; no dispositions required. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>27</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_moderator_integration_s17.py" test_count="27 (appended to the 28 already present from the prior test pass; file total 55)" coverage_target="Full moderator-module option matrix, closing the gap identified after the prior test report (fmri-elastic-net_test_20260806_131600.md): that pass exercised the C1-C4 fixes only with feature_reduction_method='none', regression, no covariates, single-output. This pass adds the reduction_method x moderator_config cross (the original crash was specifically ICA + K=2 nominal) plus targeted secondary-axis coverage." />
    </files_created>
    <design_rationale>
      This session's user explicitly requested extensive coverage of ALL option
      combinations possible with the moderator module, no synthetic CLI/SLURM dry
      run. Rather than a full combinatorial cross of every axis against every
      other axis (4 reduction x 3 moderator-config x 2 analysis_type x 3
      covariate_method x 3+ output-dimensionality would exceed 100 cells and be
      scientifically wasteful given shared code paths), the design gives one axis
      full coverage and every other axis its single highest-risk representative
      cell, presented to and approved by the user as a 27-row matrix before any
      file was written (Pre-Write Approval Gate):

      Group A (12 tests): the full 4 (feature_reduction_method) x 3
      (moderator_config: continuous, nominal K=2, nominal K=3) cross via
      run_bootstrap end-to-end. This is the primary gap: the original reported
      crash was specifically ICA reduction + K=2 nominal, and the prior test pass
      only exercised reduction='none'.

      Group B (3 tests): analysis_type x moderator_config (classification, which
      triggers the logistic Partial Ridge + Firth-fallback code path, distinct
      from the linear Partial Ridge path). The K=3 nominal cell combines
      Hotelling T2, L2-norm pooling, and Firth fallback simultaneously, the
      highest-complexity single path in the module.

      Group C (4 tests): covariate_method (incorporate, pre_regress) x
      moderator_config, verifying _strip_protected's n_covs offset and the
      canonical column order [covariates, moderator, brain, interactions] hold
      under both covariate strategies.

      Group D (3 tests): output dimensionality (multi-task regression, multi-class
      classification) x moderator_config, exercising run_bootstrap's per-task
      loop where has_moderator and interaction_report_df_k are threaded
      separately from the single-output path. Never tested together with a
      moderator before this session. The multi-class cell exercises the
      (K_class, n_moderator_cols, P) interaction coefficient shape, the
      deepest-nested shape variant in the module's documented API.

      Group E (3 tests): predict_ensemble x reduction_method (cluster_pca,
      apriori, ica). The prior pass only exercised predict_ensemble with
      reduction='none'; each reducer type transforms new data differently.

      Group F (2 tests): interaction visualization x reduction_method
      (cluster_pca, ica). apriori is excluded per the original brainstorm's T3
      scope decision (cluster-level interaction visualization is not produced for
      apriori). This group is also the first place _reconstruct_x_full is
      exercised with a real (non-None) fold reducer, since all prior unit tests
      used dummy fold_models with reducer=None.

      One authoring defect was caught and fixed during local verification before
      dispatching the full-suite run: an extraneous assertion
      (report_interaction_importance.csv, which calculate_visualization_data does
      not write) had been introduced into the ica interaction-visualization test
      and was removed; the corrected test passes and correctly checks only
      report_individual_interaction_plotting.csv and its moderator_value column.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>808</total>
    <passed>808</passed>
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
    <!-- No outstanding action items. Combined with the prior test pass
         (fmri-elastic-net_test_20260806_131600.md, 28 tests), the moderator
         module now has 55 dedicated tests spanning: the four C1-C4 code fixes
         individually, the full reduction_method x moderator_config cross, and
         targeted coverage of analysis_type, covariate_method, output
         dimensionality, predict_ensemble, and interaction visualization against
         every reduction method except the one explicitly excluded by design
         (apriori interaction visualization). No synthetic CLI/SLURM dry run was
         performed, per explicit user instruction; config-loading and
         argument-parsing code paths remain untested by this session. -->
  </action_items>
</test_report>
