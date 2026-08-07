<brainstorm_report>
  <meta project="fmri-elastic-net" mode="brainstorm" timestamp="2026-08-07T09:45:00-04:00" />
  <context_files>
    <file path="fmri-elastic-net.py" relevance="Contains both code sites requiring forward-compatibility fixes" />
    <file path="DEBUG_EXAMPLES/main_&lt;jobid&gt;.err" relevance="Production SLURM error log that surfaced the two FutureWarning categories" />
  </context_files>
  <topics>
    <topic id="T1" title="Pipeline slicing FutureWarning (line 2802)">
      <summary>The expression pipeline_boot[:-1].transform(X_boot) at line 2802 in _boot_task uses sklearn Pipeline slicing to extract the four preprocessing steps (StandardScaler, WeightTransformer, CovariateScaler, ModeratorScaler) and transform the bootstrap sample through them before Partial Ridge refit. In sklearn 1.5+, the sliced sub-pipeline wrapper does not inherit the fitted state of its parent Pipeline, triggering a FutureWarning on every .transform() call. This will become an error in sklearn 1.8. The individual transformers within the sub-pipeline ARE fitted; the warning pertains only to the wrapper object. The warning fires once per bootstrap iteration (n_fold_bootstraps x n_folds = 5,000 occurrences in the production log).</summary>
      <research>No external research required. The deprecation message and sklearn changelog fully specify the behavior change: Pipeline.__getitem__ slicing produces a sub-Pipeline whose fitness tracking is independent of its parent. The sub-Pipeline's .transform() will check check_is_fitted(self) on the wrapper, which fails because the wrapper was never directly .fit()'d.</research>
      <approaches>
        <approach id="A1" label="Manual step iteration" feasibility="high" risk="low">
          <description>Replace pipeline_boot[:-1].transform(X_boot) with an explicit loop over pipeline_boot.steps[:-1], calling step.transform() sequentially on each fitted transformer. This bypasses the Pipeline wrapper entirely and uses only the public step.transform() API.</description>
          <pros>Functionally identical to Pipeline.transform() internals. Uses only public API. Version-independent (works on all sklearn versions). No suppression or private-API dependency.</pros>
          <cons>Three lines instead of one.</cons>
        </approach>
        <approach id="A2" label="Warning suppression" feasibility="high" risk="high">
          <description>Wrap the call in warnings.catch_warnings() to filter the specific FutureWarning.</description>
          <pros>Minimal code change (adds a context manager around the existing line).</pros>
          <cons>Band-aid that will break in sklearn 1.8 when the warning becomes an error. Suppresses a signal that sklearn intends to enforce.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Manual step iteration is the only forward-compatible approach. It produces byte-identical output, uses only public API, and carries zero risk of breakage on any sklearn version.</decision>
    </topic>
    <topic id="T2" title="multi_class parameter deprecation (lines 514-515, 524-525)">
      <summary>Both LogisticRegression instantiations in _refit_binary (a closure within _partial_ridge_refit) pass multi_class='auto'. The multi_class parameter on LogisticRegression was deprecated in sklearn 1.5 and will be removed in 1.8; post-removal, LogisticRegression will always use 'multinomial'. The deprecation fires once per bootstrap iteration (5,000 occurrences). The multi_class parameter on roc_auc_score (lines 1569, 1880, 2268) is a distinct, non-deprecated parameter and requires no change.</summary>
      <research>No external research required. The deprecation message specifies: "From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning." With solver='lbfgs', multi_class='auto' already resolves to 'multinomial' internally, so removing the parameter produces identical behavior. Additionally, _refit_binary is always called with binary (2-class) targets: both the direct binary path (line 541) and the multi-class OvR loop (lines 546-548, which binarizes labels before calling _refit_binary) pass 2-class vectors. For 2-class problems, the multinomial softmax and OvR sigmoid are mathematically equivalent, making the multi_class parameter behaviorally irrelevant regardless of its value.</research>
      <approaches>
        <approach id="A1" label="Remove the parameter" feasibility="high" risk="low">
          <description>Delete multi_class='auto' from both LogisticRegression calls at lines 514-515 and 524-525. No replacement parameter needed.</description>
          <pros>Eliminates the deprecation warning. Semantically neutral (identical behavior pre- and post-change). Forward-compatible with sklearn 1.8. Minimal diff.</pros>
          <cons>None.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Remove multi_class='auto' from both sites. The parameter is behaviorally irrelevant for the binary targets that _refit_binary always receives, and its removal produces identical runtime behavior while eliminating 5,000 deprecation warnings per run.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="T1: Replace pipeline_boot[:-1].transform(X_boot) at line 2802 with manual step iteration through pipeline_boot.steps[:-1]" />
    <item priority="P0" target_mode="implement" description="T2: Remove multi_class='auto' from LogisticRegression calls at lines 514-515 and 524-525 in _partial_ridge_refit" />
    <item priority="P0" target_mode="test" description="Run full test suite (808 tests) to verify no regressions from T1 and T2 changes" />
  </action_items>
  <next_steps>Proceed to /implement for the two P0 code changes, then /test to verify 808/808 passing.</next_steps>
</brainstorm_report>
