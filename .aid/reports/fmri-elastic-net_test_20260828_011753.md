<test_report>
  <meta project="fmri-elastic-net" mode="test" timestamp="2026-08-28T01:17:53Z" />

  <pre_design_run>
    <total>811</total>
    <passed>575</passed>
    <failed>236</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures_summary>
      All 236 pre-design failures trace to two intended, spec-approved API changes from the
      implement_build_20260828_010530 build (C1-C8: T1 fully standardized coefficients,
      T2 interaction residualization). No unexplained failures; no regressions unrelated to
      the build.
    </failures_summary>
    <failure_group root_cause="A" count="160" description="run_nested_cv return signature: (score, fold_models) 2-tuple -> (score, fold_models, cv_data) 3-tuple, per build change C2">
      <affected_files>
        test_moderator_integration_s17.py (41), test_reduction_integration.py (14),
        test_pipeline_integration.py (12), test_fold_ensemble_architecture.py (10),
        test_brainstorm_s12.py (10), test_brainstorm_changes.py (10),
        test_fit_full_data_classification.py (8), test_fit_full_data_model.py (7),
        test_cr10_findings.py (7), test_covariate_paths.py (7),
        test_session11_coverage.py (6), test_implement_build_20260313.py (5),
        test_coverage_gaps.py (5), test_classification_coef_shape.py (5),
        test_adversarial_inputs.py (4), test_session11_validation.py (3),
        test_loo_integration.py (3), test_correlate_mode.py (3)
      </affected_files>
      <error_signature>ValueError: too many values to unpack (expected 2); or AssertionError on explicit len(result)==2 checks (3 tests)</error_signature>
    </failure_group>
    <failure_group root_cause="B" count="76" description="Dual raw/std coefficient reporting and OOF ensemble visualization signature changes, per build changes C5/C6">
      <affected_files>
        test_session11_validation.py (19), test_session11_coverage.py (13),
        test_fold_ensemble_architecture.py (9), test_implement_build_20260313.py (8),
        test_moderator_integration_s17.py (7), test_report_dispatch.py (6),
        test_importance_preamble.py (6), test_visualization_classification.py (3),
        test_importance_report.py (3), test_visualization_data.py (2)
      </affected_files>
      <error_signature>TypeError: missing required positional argument (sd_y_divisors/feat_std_map/fold_model_indices/config/oof_linear_pred); or KeyError: 'raw_coef_mean'</error_signature>
    </failure_group>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="ALL_ROOT_CAUSE_A (160 tests, 18 files, listed above)" file="various" classification="obsolete-test">
      <intended_contract>run_nested_cv now returns (score, fold_models, cv_data) per the locked C2 tech spec; cv_data carries fold_assignments and fold_held_out, required by C3 (per-fold SD(Y/Y*) divisors) and C6 (OOF ensemble visualization).</intended_contract>
      <current_test_claim>Tests destructure or length-check the old 2-tuple return.</current_test_claim>
      <evidence>fmri-elastic-net.py:2078 (return score, fold_models, cv_data); fmri-elastic-net_implement_plan_20260828_010530.md change C2.</evidence>
      <action>Re-expressed: 157 pure-destructuring call sites updated to a 3-element unpack (mechanical, verified via scripted diff review). 3 explicit len()==2 assertions strengthened to len()==3 plus new structural assertions on cv_data (dict keys, fold_assignments length) -- postcondition strictly strengthened, not weakened.</action>
    </disposition>
    <disposition test="ALL_ROOT_CAUSE_B (76 tests, 10 files, listed above)" file="various" classification="obsolete-test">
      <intended_contract>_write_tier1_report, _write_tier2_single, _compute_importance_preamble, _compute_importance_report, and calculate_visualization_data now require sd_y_divisors/feat_std_map/fold_model_indices/config/oof_linear_pred and produce dual raw_coef_mean/std_coef_mean columns, per the locked C5/C6 tech spec.</intended_contract>
      <current_test_claim>Tests call these functions with the pre-build 4-12 argument signatures and assert on the old single-column (fold_mean_coef, boot_mean_coef, boot_ci_low/high) schema.</current_test_claim>
      <evidence>fmri-elastic-net.py:2161 (_write_tier1_report signature), 3528 (_compute_importance_report signature), 3183 (calculate_visualization_data signature); implement_plan C5/C6 change specs.</evidence>
      <action>Re-expressed: calls updated to the new signatures (feat_std_map=1.0 / sd_y_divisors=1.0 identity values for tests not specifically validating SD(Y) standardization, isolating the arithmetic contract under test from the separately-covered SD(Y) divisor logic in test_sd_y_divisor.py). Column-name assertions updated to the new dual-column schema. Two known-answer interaction-partialling tests (test_interaction_partialling_matches_manual_computation, test_main_effect_partialling_unaffected_by_moderator_presence) rewritten to the new full-conditional-relationship formula (brain_main + mod_main + int_contrib) per T2 -- the numeric contract itself changed and the tests now encode the new formula exactly, verified against the implementation's own computation.</action>
    </disposition>
    <disposition test="27 cascaded failures across 7 files (test_pipeline_integration.py, test_correlate_mode.py, test_covariate_paths.py, test_coverage_gaps.py, test_reduction_integration.py, test_classification_coef_shape.py, test_brainstorm_changes.py)" file="various" classification="obsolete-test">
      <intended_contract>run_bootstrap's OOF visualization path (C6) requires cv_data from run_nested_cv; main() always threads it through (per C5 change item 8).</intended_contract>
      <current_test_claim>Tests fixing the Root Cause A unpacking revealed a second, masked failure: they called run_bootstrap without cv_data, discarding it from run_nested_cv's return.</current_test_claim>
      <evidence>fmri-elastic-net.py:3785 (run_bootstrap calls _compute_oof_visualization_data(fold_models, cv_data, ...) unconditionally); TypeError: 'NoneType' object is not subscriptable at fmri-elastic-net.py:3122.</evidence>
      <action>Re-expressed: all 27 call sites updated to capture and thread cv_data through to run_bootstrap, matching main()'s actual usage. Separately surfaced to the user as a design-inconsistency observation (run_bootstrap's cv_data=None default implies optionality that never holds in practice) -- see action_items below.</action>
    </disposition>
    <disposition test="TestReductionMethodModeratorMatrix::test_none_nominal_k3, test_cluster_pca_nominal_k3, test_apriori_nominal_k3, test_ica_nominal_k3; TestAnalysisTypeModeratorMatrix::test_classification_nominal_k3" file="test_moderator_integration_s17.py" classification="product-bug" verdict="CONFIRMED">
      <intended_contract>run_bootstrap standardizes interaction coefficients to fully-standardized scale for any moderator configuration, including K>2 nominal moderators (n_moderator_cols > 1, producing a 3D (B, n_mod_cols, P) interaction coefficient array).</intended_contract>
      <current_test_claim>Tests assert report_feature_importance.csv and report_interaction_importance.csv are written without crash for the reduction_method x K=3-nominal-moderator matrix.</current_test_claim>
      <evidence>fmri-elastic-net.py:3826: coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis] -- assumes a 2D interaction array; for K>2 nominal moderators the array is 3D (B, n_mod_cols, P), producing ValueError: operands could not be broadcast together with shapes (150,2,15) (150,1). The multi-output branch (fmri-elastic-net.py:3947-3949) already handles this correctly with ndim-aware broadcasting, confirming this is an isolated oversight in the single-output branch, not a systemic pattern.</evidence>
      <action>NO assertion edit (tests correctly encode the intended contract). Routed to /implement as a P0 action item (see below) -- outside /test's write scope (fmri-elastic-net.py).</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>44</tests_created>
    <tests_modified>22</tests_modified>
    <files_created>
      <file path="tests/test_sd_y_divisor.py" test_count="13" coverage_target="_compute_sd_y_divisor: single-output regression SD(Y), multi-task per-task SD(Y), binary classification latent-variable SD(Y*), multi-class per-contrast SD(Y*), near-zero degeneracy guard" />
      <file path="tests/test_reference_difference.py" test_count="9" coverage_target="_reference_difference: contrast value correctness (known-answer), 3D/4D shape handling, label generation" />
      <file path="tests/test_oof_visualization.py" test_count="7" coverage_target="_compute_oof_visualization_data: per-subject held-out-fold correctness (verified against manual per-fold recomputation), moderator main-effect threading, X_full_repr consistency, dimensionality-reduction compatibility" />
      <file path="tests/test_fold_performance.py" test_count="7" coverage_target="_write_fold_performance: R2/AUC_ROC metric naming, per-fold output structure, summary statistics (known-answer mean/SD)" />
      <file path="tests/test_reference_class_validation.py" test_count="5" coverage_target="reference_class config validation gate (CLI-level, subprocess): multi-class halt/invalid-reference/valid-reference paths, binary and regression exemption" />
      <file path="tests/test_visualization_classification.py (extension)" test_count="3" coverage_target="Interaction residualization (T2): lin_contrib = brain_main + mod_main + int_contrib known-answer verification, additive inclusion of moderator main effect, main_report_df=None fallback" />
    </files_created>
    <files_modified>
      <file path="tests/test_adversarial_inputs.py" reason="Root Cause A re-expression" />
      <file path="tests/test_brainstorm_changes.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_brainstorm_s12.py" reason="Root Cause A re-expression" />
      <file path="tests/test_classification_coef_shape.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_correlate_mode.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_covariate_paths.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_coverage_gaps.py" reason="cv_data threading" />
      <file path="tests/test_cr10_findings.py" reason="Root Cause A re-expression" />
      <file path="tests/test_fit_full_data_classification.py" reason="Root Cause A re-expression (incl. known-answer tuple-length strengthening)" />
      <file path="tests/test_fit_full_data_model.py" reason="Root Cause A re-expression (incl. known-answer tuple-length strengthening, test renamed to reflect new contract)" />
      <file path="tests/test_fold_ensemble_architecture.py" reason="Root Cause A+B re-expression (incl. known-answer tuple-length strengthening)" />
      <file path="tests/test_implement_build_20260313.py" reason="Root Cause A+B re-expression" />
      <file path="tests/test_importance_preamble.py" reason="Root Cause B re-expression" />
      <file path="tests/test_importance_report.py" reason="Root Cause B re-expression" />
      <file path="tests/test_loo_integration.py" reason="Root Cause A re-expression" />
      <file path="tests/test_moderator_integration_s17.py" reason="Root Cause A+B re-expression + cv_data threading + interaction-formula known-answer rewrite" />
      <file path="tests/test_pipeline_integration.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_reduction_integration.py" reason="Root Cause A re-expression + cv_data threading" />
      <file path="tests/test_report_dispatch.py" reason="Root Cause B re-expression" />
      <file path="tests/test_session11_coverage.py" reason="Root Cause A+B re-expression + cv_data threading" />
      <file path="tests/test_session11_validation.py" reason="Root Cause A+B re-expression + cv_data threading" />
      <file path="tests/test_visualization_data.py" reason="Root Cause B re-expression" />
    </files_modified>
    <design_rationale>
      Re-expression scope was determined entirely by the pre-design failure ledger (236 failures,
      2 root causes, both traced to locked, user-approved tech spec changes). New test files target
      the build's genuinely new functionality that had zero prior coverage: _compute_sd_y_divisor,
      _reference_difference, _compute_oof_visualization_data, _write_fold_performance, the
      reference_class config gate, and the T2 interaction residualization formula. Re-expression
      surfaced two additional defects not visible in the original brainstorm/implement cycle
      (masked by the upstream Root Cause A/B failures): the run_bootstrap cv_data=None crash
      (resolved for all test call sites; production-code hardening decision already made by the
      user mid-session -- see action_items) and the K>2 nominal moderator interaction-standardization
      shape-broadcast bug (a genuine, isolated product defect, left unmodified in the 5 affected
      tests and routed to /implement).
    </design_rationale>
  </design_phase>

  <post_design_run>
    <total>855</total>
    <passed>850</passed>
    <failed>5</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestReductionMethodModeratorMatrix::test_none_nominal_k3" file="tests/test_moderator_integration_s17.py" line="3826">
        <error_type>ValueError</error_type>
        <message>operands could not be broadcast together with shapes (150,2,15) (150,1)</message>
        <traceback>coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]</traceback>
        <likely_cause>run_bootstrap's single-output branch does not handle the 3D interaction coefficient array shape produced by K>2 nominal moderators (n_moderator_cols > 1). Fix: mirror the ndim-aware broadcasting already used in the multi-output branch (fmri-elastic-net.py:3947-3949).</likely_cause>
      </failure>
      <failure test="TestReductionMethodModeratorMatrix::test_cluster_pca_nominal_k3" file="tests/test_moderator_integration_s17.py" line="3826">
        <error_type>ValueError</error_type>
        <message>operands could not be broadcast together with shapes (150,2,15) (150,1)</message>
        <traceback>coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]</traceback>
        <likely_cause>Same root cause as test_none_nominal_k3; reduction_method does not affect the defect.</likely_cause>
      </failure>
      <failure test="TestReductionMethodModeratorMatrix::test_apriori_nominal_k3" file="tests/test_moderator_integration_s17.py" line="3826">
        <error_type>ValueError</error_type>
        <message>operands could not be broadcast together with shapes (150,2,15) (150,1)</message>
        <traceback>coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]</traceback>
        <likely_cause>Same root cause as test_none_nominal_k3; reduction_method does not affect the defect.</likely_cause>
      </failure>
      <failure test="TestReductionMethodModeratorMatrix::test_ica_nominal_k3" file="tests/test_moderator_integration_s17.py" line="3826">
        <error_type>ValueError</error_type>
        <message>operands could not be broadcast together with shapes (150,2,15) (150,1)</message>
        <traceback>coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]</traceback>
        <likely_cause>Same root cause as test_none_nominal_k3; reduction_method does not affect the defect.</likely_cause>
      </failure>
      <failure test="TestAnalysisTypeModeratorMatrix::test_classification_nominal_k3" file="tests/test_moderator_integration_s17.py" line="3826">
        <error_type>ValueError</error_type>
        <message>operands could not be broadcast together with shapes (150,2,15) (150,1)</message>
        <traceback>coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis]</traceback>
        <likely_cause>Same root cause as test_none_nominal_k3; analysis_type (binary classification here) does not affect the defect -- driven entirely by n_moderator_cols > 1 (K=3 nominal).</likely_cause>
      </failure>
    </failures>
  </post_design_run>

  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>2</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>

  <action_items>
    <item priority="P0" target_mode="implement" description="run_bootstrap: interaction coefficient standardization crashes for K&gt;2 nominal moderators. At fmri-elastic-net.py:3826, coef_matrix_interaction_std = coef_matrix_interaction_internal / sd_y_per_iter[:, np.newaxis] assumes a 2D (B,P) interaction array; for K&gt;2 nominal moderators (n_moderator_cols &gt; 1) the array is 3D (B, n_mod_cols, P), raising ValueError on broadcast. Fix: branch on coef_matrix_interaction_internal.ndim, using sd_y_per_iter[:, np.newaxis, np.newaxis] for the 3D case, mirroring the existing correct logic in the multi-output branch at fmri-elastic-net.py:3947-3949. Confirmed via 5 reproducing tests (all reduction methods x K=3 nominal moderator, both regression and binary classification) in tests/test_moderator_integration_s17.py; do not modify these tests, they correctly encode the intended contract." />
    <item priority="P0" target_mode="implement" description="run_bootstrap: cv_data parameter should be made required (default removed), not left as cv_data=None. Decided by the user during this /test session: the None default falsely implies optionality -- main() always constructs and passes cv_data (real usage never omits it), and run_bootstrap's body unconditionally calls _compute_oof_visualization_data(fold_models, cv_data, ...) which crashes uninformatively (TypeError: 'NoneType' object is not subscriptable, 3 frames deep) if cv_data is omitted. Remove the default from run_bootstrap's signature (fmri-elastic-net.py, def run_bootstrap(..., apriori_map=None, moderator=None, sd_y_divisors=None, cv_data=None): -> make cv_data a required parameter, e.g. by moving it before the optional args or keeping it keyword-only without a default). All test call sites already thread cv_data through correctly in this build; no test changes needed for this item." />
  </action_items>
</test_report>
