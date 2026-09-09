<implement_report>
  <meta project="fmri-elastic-net" mode="implement" submodule="build" timestamp="2026-08-27T23:54:48Z" />
  <spec_ref>fmri-elastic-net_implement_plan_20260828_010530.md</spec_ref>
  <changes_applied>
    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="config_template.yaml" lines_changed="8" />
        <file path="fmri-elastic-net.py" lines_changed="16" />
      </files_modified>
      <notes>Added reference_class parameter to config_template.yaml. Added multi-class validation block in main() that detects multi-class classification, validates reference_class presence, and stores class_labels and reference_class in config['_runtime'].</notes>
    </change>
    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="95" />
      </files_modified>
      <notes>Extended run_nested_cv to track fold_assignments and fold_held_out arrays in cv_data dict. Added per-fold score computation reusing existing predictions (no duplicate predict calls). New _write_fold_performance helper writes model_performance_per_fold.csv and model_performance_fold_summary.csv. main() unpacks cv_data and calls _write_fold_performance.</notes>
    </change>
    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="58" />
      </files_modified>
      <notes>New _compute_sd_y_divisor function with four branches: single regression (per-fold SD(Y)), multi-task regression (per-task SD(Y) vector), binary classification (SD(Y*) via logit variance + pi^2/3), multi-class (per-contrast SD(Y*) after reference-differencing). Near-zero guard at 1e-10. Called in main() after fold model extraction.</notes>
    </change>
    <change id="C4" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="42" />
      </files_modified>
      <notes>New _reference_difference function computes beta_k - beta_ref via np.take on axis=-2. Returns (contrasts with shape (..., K-1, P), contrast_labels list). Applied in run_tier1_inference and run_bootstrap multi-output branches. Handles both 3D and 4D (interaction) coefficient shapes. task_labels reassigned to contrast_labels after call.</notes>
    </change>
    <change id="C8" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="0" />
      </files_modified>
      <notes>Verification-only change. Confirmed C4 already correctly reassigns task_labels to contrast_labels after _reference_difference in both run_tier1_inference and run_bootstrap. No code changes needed.</notes>
    </change>
    <change id="C5" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="298" />
      </files_modified>
      <notes>Dual reporting overhaul across 8 functions. _write_tier1_report: new sd_y_divisors/feat_std_map/fold_model_indices params; produces fold_mean_raw, fold_mean_std, dual CI columns. run_tier1_inference: computes feat_std_map internally, threads sd_y_divisors. _compute_importance_preamble: dual df_coef inputs, returns dict with std_means, raw_means. _report_apriori: dual df_coef inputs, cluster raw+std stats. _report_standard: dual df_coef inputs. _write_tier2_single: dual coef_pool inputs, produces dual raw/std output. run_bootstrap: sd_y_divisors+cv_data params, fold-index tracking via task_list reconstruction, sd_y_per_iter standardization. main() call sites updated.</notes>
    </change>
    <change id="C6" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="112" />
      </files_modified>
      <notes>New _compute_oof_visualization_data computes per-subject linear predictions from held-out fold models. calculate_visualization_data refactored: oof_linear_pred replaces best_model; interaction lin_contrib = brain_main + mod_main + int_contrib (full conditional relationship per Aiken and West 1991). _compute_importance_report: best_model replaced with oof_linear_pred/oof_mod_main. No best_model references remain in the file.</notes>
    </change>
    <change id="C7" status="done" user_decision="n/a">
      <files_modified>
        <file path="fmri-elastic-net.py" lines_changed="18" />
        <file path="INPUT_SPECIFICATION.md" lines_changed="22" />
        <file path="README.md" lines_changed="18" />
      </files_modified>
      <notes>Module docstring: 4 new caveats (classification SD(Y*) model-dependence, per-fold standardization, unboundedness in multiple regression, multi-class extension). Updated raw_coef caveat to independent-derivation framing. INPUT_SPECIFICATION.md: reference_class parameter, updated coefficient semantics (fully standardized with latent-variable formula), multi-class output structure, per-fold output file schemas. README.md: fully standardized coefficient description, reference_class documentation, per-fold output files.</notes>
    </change>
  </changes_applied>
  <summary>
    <total_changes>8</total_changes>
    <completed>8</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <file_summary>
    <file path="fmri-elastic-net.py" total_lines="4363" lines_changed="~639" note="Up from ~3860 lines. New functions: _compute_sd_y_divisor, _reference_difference, _compute_oof_visualization_data, _write_fold_performance. Refactored: _write_tier1_report, _write_tier2_single, _compute_importance_preamble, _report_standard, _report_apriori, _compute_importance_report, calculate_visualization_data, run_tier1_inference, run_bootstrap, run_nested_cv, main." />
    <file path="config_template.yaml" lines_changed="8" note="Added reference_class parameter." />
    <file path="INPUT_SPECIFICATION.md" lines_changed="22" note="Documented reference_class, updated coefficient semantics, added per-fold output files." />
    <file path="README.md" lines_changed="18" note="Updated coefficient interpretation, documented reference_class and per-fold metrics." />
  </file_summary>
  <next_steps>Recommended: run /test to validate all changes. The build introduced significant refactoring across coefficient reporting, visualization, and inference functions. Key areas to test: (1) dual raw/std coefficient output in Tier 1 and Tier 2 reports, (2) per-fold SD(Y/Y*) computation for regression and classification, (3) multi-class reference-differencing, (4) OOF ensemble visualization, (5) interaction residualization (brain main + mod main + interaction), (6) per-fold performance metrics output, (7) regression of existing tests (811 tests).</next_steps>
</implement_report>
