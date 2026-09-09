<test_report>
  <meta project="fmri-elastic-net" mode="test" timestamp="2026-09-09T13:24:04Z" />
  <note>This report closes the test cycle 2 validation of the C1/C2 P0 fixes (K&gt;2 nominal moderator broadcast crash; run_bootstrap cv_data required-parameter change), whose pre-design run and design phase were completed in the prior session (2026-08-28/29) but whose post-design run and report were left incomplete when that session ended. The pre-design run and design-phase sections below are reconstructed from that session's memory record for continuity; the post-design run in this section is freshly executed and independently receipt-verified this session.</note>
  <pre_design_run source="prior session, 2026-08-28">
    <total>855</total>
    <passed>854</passed>
    <failed>1</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures>
      <failure test="TestBootstrapFailureThresholds::test_majority_failures_raises_runtime_error" file="tests/test_coverage_gaps.py" line="379">
        <error_type>TypeError</error_type>
        <message>run_bootstrap() missing 1 required keyword-only argument: 'cv_data'</message>
        <traceback>Call site omitted the cv_data argument entirely following the C2 change (run_bootstrap's cv_data parameter converted from an optional None-default to a required keyword-only parameter). The C2 verification pass had confirmed all call sites in fmri-elastic-net.py itself passed cv_data by keyword, but missed this one test call site.</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestBootstrapFailureThresholds::test_majority_failures_raises_runtime_error" file="tests/test_coverage_gaps.py" classification="obsolete-test">
      <intended_contract>run_bootstrap should raise RuntimeError when a majority of bootstrap iterations fail (via the patched _boot_task returning None), independent of cv_data content, because the RuntimeError is raised before the OOF-visualization step that consumes cv_data.</intended_contract>
      <current_test_claim>The test asserted pytest.raises(RuntimeError, match="Majority of bootstrap") but the call omitted cv_data, which is no longer optional as of C2 (run_bootstrap cv_data required-parameter change).</current_test_claim>
      <evidence>fmri-elastic-net.py run_bootstrap signature (cv_data is keyword-only, no default, post-C2); the RuntimeError raise site precedes any read of cv_data, confirmed by tracing the majority-failure branch ahead of the OOF-visualization call.</evidence>
      <action>re-express: added explicit cv_data=None keyword argument at tests/test_coverage_gaps.py:384-386, with an inline comment recording why None is safe on this code path (the majority-failure RuntimeError fires before cv_data is ever read). No assertion content changed; the pytest.raises postcondition is identical to its prior form.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase source="prior session, 2026-08-29">
    <tests_created>0</tests_created>
    <tests_modified>1</tests_modified>
    <files_created>
      <file path="tests/test_coverage_gaps.py" test_count="1" coverage_target="Call-site syntax fix only: restores compatibility with the C2 required-keyword cv_data signature for TestBootstrapFailureThresholds::test_majority_failures_raises_runtime_error. No new coverage added." />
    </files_created>
    <design_rationale>Single obsolete-test disposition; the only failure traced to a call site the C2 implementation's own verification pass had not enumerated. No broader design action was warranted since this was the sole failure in the pre-design run and the K&gt;2 nominal moderator fix (C1) was already confirmed passing across all 5 previously-failing tests in that same pre-design run.</design_rationale>
  </design_phase>
  <post_design_run source="this session, 2026-09-09, receipt-verified">
    <total>855</total>
    <passed>855</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures />
  </post_design_run>
  <verification_notes>Dispatched via execution-agent-sonnet-medium (nonce [receipt-nonce], T0=[timestamp], T1=[timestamp], wall time ~370s). Receipt verification (verify_receipts.py, dispatch-site test:run_suite) returned ok=true across all 8 checks: nonce match, clock window match, duration match, summary match, collect_total match (independent --collect-only oracle: 855), receipt-file mtime match, receipt-file nonce match, receipt-file summary match. All 856 emitted warnings are pre-existing and non-blocking: RuntimeWarning (precision loss / divide-by-zero in known-answer edge-case tests exercising zero-mean or constant-coefficient conditions by design), ConvergenceWarning (lbfgs iteration limit on a nominal K=3 classification fixture), UndefinedMetricWarning (R^2/ROC AUC undefined on single-sample or single-class LOO folds by construction). None indicate new defects.</verification_notes>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
    <item priority="P2" target_mode="document" description="Update README.md, INPUT_SPECIFICATION.md, and AID_LOG.md for: (1) the std_coef redefinition to fully standardized coefficients (beta: SDs of Y per 1 SD of X) across all analysis types, replacing the prior X-standardized-only b_x definition; (2) the interaction-visualization residualization change (full conditional relationship: brain main + moderator main + interaction, via OOF ensemble predictions); (3) the K&gt;2 nominal moderator broadcast fix; (4) the run_bootstrap cv_data required-parameter change." />
    <item priority="P2" target_mode="publish" description="After document, publish fmri-elastic-net.py, README.md, INPUT_SPECIFICATION.md, and config_template.yaml (all currently showing as modified in the working tree) along with this session's and the prior session's commits." />
  </action_items>
</test_report>
