<test_report>
  <meta project="fmri-elastic-net" mode="test" timestamp="2026-07-29T22:30:00-04:00" />
  <pre_design_run>
    <total>742</total>
    <passed>715</passed>
    <failed>27</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestLoadBasic::test_loads_correct_dimensions" file="tests/test_load_and_prep.py" line="87">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>load_and_prep_data now returns 8 values (added moderator); test unpacks 7.</traceback>
      </failure>
      <failure test="TestLoadBasic::test_covariate_none_returns_empty_cov" file="tests/test_load_and_prep.py" line="121">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestLoadBasic::test_covariate_incorporate_returns_covariates" file="tests/test_load_and_prep.py" line="138">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestMissingData::test_listwise_deletion_removes_nan_rows" file="tests/test_load_and_prep.py" line="161">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestAprioriMapLoading::test_apriori_map_loaded" file="tests/test_load_and_prep.py" line="245">
        <error_type>AssertionError</error_type>
        <message>assert None is not None</message>
        <traceback>*_ unpacking captured the new trailing moderator field instead of apriori_map (now second-to-last, not last).</traceback>
      </failure>
      <failure test="TestWeightNormalization::test_weights_normalized_to_sum_n" file="tests/test_brainstorm_s12.py" line="107">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same load_and_prep_data 8-value return.</traceback>
      </failure>
      <failure test="TestWeightNormalization::test_relative_weights_preserved" file="tests/test_brainstorm_s12.py" line="123">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestWeightNormalization::test_uniform_weights_unchanged_ratio" file="tests/test_brainstorm_s12.py" line="138">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestWeightNormalization::test_ess_computation_correctness" file="tests/test_brainstorm_s12.py" line="230">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestWeightNormalization::test_no_weights_returns_none" file="tests/test_brainstorm_s12.py" line="248">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 7)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestR1ScaffoldingRemoval::test_fold_models_required_keys_present" file="tests/test_brainstorm_s12.py" line="427">
        <error_type>AssertionError</error_type>
        <message>Unexpected keys in fold_models: coef_original_interaction, n_reduced, n_moderator_cols</message>
        <traceback>fold_models dict gained 3 keys from the interaction architecture; expected_keys set not updated.</traceback>
      </failure>
      <failure test="TestBootTaskBasic::test_returns_coef_and_converged" file="tests/test_boot_task_unit.py" line="86">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>_boot_task now returns 4-tuple (c_main, c_interaction, converged, firth_used); test unpacks 2.</traceback>
      </failure>
      <failure test="TestBootTaskBasic::test_coef_length_matches_original_features" file="tests/test_boot_task_unit.py" line="122">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootTaskPreRegress::test_pre_regress_uses_covariates" file="tests/test_boot_task_unit.py" line="206">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootTaskClassification::test_classification_boot_task" file="tests/test_boot_task_unit.py" line="245">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBinaryClassificationCoefSqueeze::test_boot_task_binary_squeeze" file="tests/test_brainstorm_changes.py" line="558">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootTaskMultiOutput::test_boot_task_multitask_returns_2d" file="tests/test_brainstorm_changes.py" line="727">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootTaskClassificationShape::test_boot_task_binary_returns_1d" file="tests/test_classification_coef_shape.py" line="156">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootTaskClassificationShape::test_boot_task_binary_coef_length" file="tests/test_classification_coef_shape.py" line="175">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestBootstrapConvergenceWarning::test_convergence_warnings_logged" file="tests/test_coverage_gaps.py" line="467">
        <error_type>IndexError</error_type>
        <message>tuple index out of range</message>
        <traceback>Mock side_effect returned a 2-tuple; run_bootstrap indexes r[2] for convergence in the new 4-tuple format.</traceback>
      </failure>
      <failure test="TestBootTaskConvergenceTracking::test_boot_task_returns_convergence_flag" file="tests/test_coverage_gaps.py" line="574">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause as other _boot_task unpacking failures.</traceback>
      </failure>
      <failure test="TestMultiClassCovariateReductionBootstrap::test_multiclass_incorporate_boot_task" file="tests/test_coverage_gaps.py" line="631">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestCovariateStripping::test_boot_task_returns_brain_only_incorporate" file="tests/test_fold_ensemble_architecture.py" line="415">
        <error_type>ValueError</error_type>
        <message>too many values to unpack (expected 2)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestPipelineStructure::test_pipeline_has_four_steps" file="tests/test_create_model_validation.py" line="205">
        <error_type>AssertionError</error_type>
        <message>4-item list != 5-item list (extra: mod_scaler)</message>
        <traceback>Pipeline gained a ModeratorScaler step; test asserts the pre-interaction 4-step structure.</traceback>
      </failure>
      <failure test="TestCreateModelAndParamDist::test_pipeline_has_four_steps" file="tests/test_model.py" line="116">
        <error_type>AssertionError</error_type>
        <message>4-item list != 5-item list (extra: mod_scaler)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestCreateModelWeightTransformer::test_pipeline_step_order" file="tests/test_session11_validation.py" line="918">
        <error_type>AssertionError</error_type>
        <message>4-item list != 5-item list (extra: mod_scaler)</message>
        <traceback>Same root cause.</traceback>
      </failure>
      <failure test="TestFoldWiseEnsembleE2E::test_run_nested_cv_returns_fold_models" file="tests/test_session11_validation.py" line="956">
        <error_type>AssertionError</error_type>
        <message>Extra items in fold_models keys: coef_original_interaction, n_reduced, n_moderator_cols</message>
        <traceback>Same root cause as the test_brainstorm_s12.py fold_models keys failure.</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestLoadBasic::test_loads_correct_dimensions (+4 sibling tests in test_load_and_prep.py, +5 in test_brainstorm_s12.py)" file="tests/test_load_and_prep.py, tests/test_brainstorm_s12.py" classification="obsolete-test">
      <intended_contract>load_and_prep_data's return-tuple contract changed in Session 14.5 to append a moderator specification as an 8th value (docstring and line 1186 confirm: X_brain, X_cov, Y, weights, subj_ids, active_covs, apriori_map, moderator).</intended_contract>
      <current_test_claim>Tests unpack exactly 7 values, written before the interaction modeling feature existed.</current_test_claim>
      <evidence>fmri-elastic-net.py:1186 return statement; git history shows load_and_prep_data predates Session 13-14 interaction modeling work.</evidence>
      <action>re-express: extend each unpacking statement to 8 values, adding a moderator placeholder. Postcondition strengthened (assertions unchanged, only the unpacking arity corrected to match the real contract).</action>
    </disposition>
    <disposition test="TestAprioriMapLoading::test_apriori_map_loaded" file="tests/test_load_and_prep.py" classification="obsolete-test">
      <intended_contract>apriori_map is the 7th of 8 return values (moderator now trails it).</intended_contract>
      <current_test_claim>*_, apriori_map = load_and_prep_data(...) assumed apriori_map was the last value.</current_test_claim>
      <evidence>fmri-elastic-net.py:1186.</evidence>
      <action>re-express: *_, apriori_map, _ = load_and_prep_data(...) to correctly bind apriori_map ahead of the trailing moderator value.</action>
    </disposition>
    <disposition test="TestR1ScaffoldingRemoval::test_fold_models_required_keys_present, TestFoldWiseEnsembleE2E::test_run_nested_cv_returns_fold_models" file="tests/test_brainstorm_s12.py, tests/test_session11_validation.py" classification="obsolete-test">
      <intended_contract>fold_models entries carry coef_original_interaction, n_moderator_cols, and n_reduced alongside the pre-existing keys, per the Session 14.5 interaction architecture (fmri-elastic-net.py:1814-1826).</intended_contract>
      <current_test_claim>expected_keys set omits the three interaction-era keys, reflecting the pre-Session-14.5 fold_models schema.</current_test_claim>
      <evidence>fmri-elastic-net.py:1814-1826 fold_models.append(...) construction.</evidence>
      <action>re-express: add the three keys to each expected_keys set. Postcondition strengthened (asserts the complete current schema rather than a stale subset).</action>
    </disposition>
    <disposition test="TestBootTaskBasic::test_returns_coef_and_converged (+10 sibling tests across 5 files)" file="tests/test_boot_task_unit.py, tests/test_brainstorm_changes.py, tests/test_classification_coef_shape.py, tests/test_coverage_gaps.py, tests/test_fold_ensemble_architecture.py" classification="obsolete-test">
      <intended_contract>_boot_task returns a 4-tuple (c_original_main, c_original_interaction, converged, firth_used) per fmri-elastic-net.py:2829, reflecting the Session 14.5 interaction architecture's need to report interaction coefficients and Firth-fallback status alongside the main-effect coefficients.</intended_contract>
      <current_test_claim>Tests unpack a 2-tuple (coefs, converged), the pre-interaction contract.</current_test_claim>
      <evidence>fmri-elastic-net.py:2829 return c_original_main, c_original_interaction, converged, firth_used.</evidence>
      <action>re-express: extend each unpacking statement to 4 values. Assertions against coefs/converged are unchanged; only the arity is corrected.</action>
    </disposition>
    <disposition test="TestBootstrapConvergenceWarning::test_convergence_warnings_logged" file="tests/test_coverage_gaps.py" classification="obsolete-test">
      <intended_contract>run_bootstrap indexes r[2] as the convergence flag within the 4-tuple _boot_task contract (fmri-elastic-net.py:3275).</intended_contract>
      <current_test_claim>The test's mock side_effect fabricated a 2-tuple (result[0], False) to simulate a non-converged iteration, which is inconsistent with the 4-tuple contract the mock is standing in for.</current_test_claim>
      <evidence>fmri-elastic-net.py:3275 n_conv_warn = sum(1 for r in valid_res if not r[2]); fmri-elastic-net.py:2829 four-tuple return.</evidence>
      <action>re-express: mock now returns (result[0], result[1], False, result[3]), preserving c_main and firth_used from the real call while forcing converged=False at index 2, matching the real return shape the mock is intercepting.</action>
    </disposition>
    <disposition test="TestPipelineStructure::test_pipeline_has_four_steps (+2 sibling tests)" file="tests/test_create_model_validation.py, tests/test_model.py, tests/test_session11_validation.py" classification="obsolete-test">
      <intended_contract>create_model_and_param_dist's pipeline has 5 steps: scaler, weight_transformer, cov_scaler, mod_scaler, model (fmri-elastic-net.py:1428-1434), the mod_scaler step (ModeratorScaler) having been added in Session 14.5 to protect the moderator main effect from covariate-strength regularization.</intended_contract>
      <current_test_claim>Tests assert the pre-interaction 4-step pipeline.</current_test_claim>
      <evidence>fmri-elastic-net.py:1428-1434 Pipeline([...]) construction, 5 named steps.</evidence>
      <action>re-express: update expected step lists to include mod_scaler at its documented position (after cov_scaler, before model); rename test methods from four_steps to five_steps to keep the name accurate. Postcondition strengthened (asserts the complete current 5-step contract).</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>11</tests_created>
    <tests_modified>26</tests_modified>
    <files_created>
      <file path="tests/test_interaction_modeling.py" test_count="11" coverage_target="Session-14.5/15 fixes: logistic Partial Ridge back-transformation direction (known-answer test against independent sklearn fit), regression Partial Ridge OLS-consistency regression guard, Hotelling T-squared df2&lt;=0 and rank-deficient-covariance NaN guards plus a non-degenerate-case regression check, and _code_moderator levels_override consistency across resamples with divergent observed levels." />
    </files_created>
    <design_rationale>
      All 27 pre-design failures were classified obsolete-test: every failure traces to a return-contract or pipeline-structure change introduced by the Session 14.5 interaction modeling implementation (load_and_prep_data's 8th return value, _boot_task's 4-tuple return, the 5-step pipeline, and the expanded fold_models schema), none of which had corresponding test updates at the time of that implementation. No product-bug or ambiguous dispositions were found; all re-expressions extend unpacking arity or expected-key/expected-step sets to match the documented current contract without weakening any assertion. Zero tests were created for C1-C6 (this session's CR-driven fixes) prior to this design pass: _partial_ridge_refit, _hotelling_t2, and _code_moderator's levels_override parameter had no prior coverage at all, so test_interaction_modeling.py closes that gap with known-answer and regression-guard tests rather than mere smoke tests.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>753</total>
    <passed>753</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
  </action_items>
</test_report>
