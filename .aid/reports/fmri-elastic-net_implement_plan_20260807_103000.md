<implement_plan>
  <meta project="fmri-elastic-net" mode="implement" submodule="plan" timestamp="2026-08-07T10:30:00-04:00" />
  <input_reports>
    <report path="(conversation)" mode="discussion" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="warning-filter-discussion">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Add a centralized warning filter block in main() to suppress known-harmless warnings from stderr. Three filter rules: (1) ConvergenceWarning globally (covers RandomizedSearchCV hyperparameter search and Partial Ridge refit LogisticRegression); (2) RuntimeWarning matching 'overflow encountered in exp' (from _firth_logistic sigmoid computation, numerically correct); (3) any warning matching 'Ill-conditioned matrix' (from _firth_logistic Fisher matrix solve, expected for near-separation bootstrap resamples). The existing catch_warnings(record=True) block in _boot_task (lines 2783-2791) is unaffected because its inner simplefilter("always", ConvergenceWarning) replaces the filter chain within the context manager, preserving the programmatic convergence-state capture and the downstream logging summary in run_bootstrap (lines 3366-3370).</description>
      <spec>
Insert the following block in main() immediately after the setup_logging() call (after current line 3772), before the config validation block (before current line 3774):

    # Suppress known-harmless warnings that clutter stderr during normal operation.
    # The _boot_task catch_warnings(record=True) block still captures convergence
    # state: its inner simplefilter("always") overrides this filter within that scope.
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', message='overflow encountered in exp',
                            category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='Ill-conditioned matrix')

The block is 5 lines (1 comment line + 4 code lines). No existing lines are modified or moved.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - the catch_warnings(record=True) + simplefilter("always") pattern in _boot_task overrides outer filters within its scope, so the programmatic convergence capture is preserved; the existing per-task suppression in _run_perm_task and _run_block_perm_task becomes redundant but harmless</risk>
      <rollback>Remove the 5-line block</rollback>
    </change>
    <change id="C2" priority="P1" source_item="warning-filter-discussion">
      <file path="fmri-elastic-net.py" action="modify" />
      <description>Update the module-level comment at lines 128-130 to reflect the new centralized suppression approach. The existing comment says ConvergenceWarning is suppressed "only for perm workers," which is no longer accurate after C1 adds a global filter in main().</description>
      <spec>
Replace lines 128-130:
    # Suppress ConvergenceWarning globally only for perm workers where it is expected;
    # elsewhere we count and log occurrences (see run_bootstrap).
    # UserWarning suppression removed — address specific warnings as they arise.

With:
    # Known-harmless warnings (ConvergenceWarning, Firth overflow/ill-conditioning)
    # are suppressed in main(). The _boot_task catch_warnings block still captures
    # convergence state for the run_bootstrap logging summary.
      </spec>
      <dependencies>C1</dependencies>
      <risk>low - comment-only change</risk>
      <rollback>Restore original comment text</rollback>
    </change>
  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
