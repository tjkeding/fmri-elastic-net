<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-08-07T10:00:00-04:00" />
  <input_reports>
    <report path="fmri-elastic-net_brainstorm_20260807_094500.md" mode="brainstorm" key_items="2" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="T1">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Replace pipeline slicing with manual step iteration to eliminate sklearn FutureWarning at line 2802 in _boot_task. The expression pipeline_boot[:-1].transform(X_boot) is replaced with an explicit loop over pipeline_boot.steps[:-1], calling step.transform() on each fitted transformer sequentially.</description>
      <spec>
Replace the single line:
    X_boot_transformed = pipeline_boot[:-1].transform(X_boot)

With:
    X_boot_transformed = X_boot
    for _, step in pipeline_boot.steps[:-1]:
        X_boot_transformed = step.transform(X_boot_transformed)

No other lines in the surrounding block change. The variable X_boot_transformed retains its name and is consumed identically on line 2803.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - functionally identical to Pipeline.transform() internals; same fitted transformers called in the same order</risk>
      <rollback>Revert the 3-line loop back to the single pipeline slicing expression</rollback>
    </change>
    <change id="C2" priority="P0" source_item="T2">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Remove the deprecated multi_class='auto' parameter from both LogisticRegression instantiations in _refit_binary (closure within _partial_ridge_refit). Two sites: the all-unselected Ridge-only path (lines 514-515) and the split-estimator selected+unselected path (lines 524-525).</description>
      <spec>
Site 1 (lines 514-515): Change from:
    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                            max_iter=5000, multi_class='auto')
To:
    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                            max_iter=5000)

Site 2 (lines 524-525): Same change, same two-line pattern. Remove ", multi_class='auto'" from the second line, leaving max_iter=5000 as the final keyword argument followed by the closing parenthesis.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - parameter removal is semantically neutral; multi_class='auto' with solver='lbfgs' already resolves to 'multinomial' internally, which is the post-removal default</risk>
      <rollback>Re-add multi_class='auto' to both LogisticRegression calls</rollback>
    </change>
  </changes>
  <execution_order>C1, C2 (independent; no ordering dependency)</execution_order>
</implement_plan>
