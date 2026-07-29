<document_report>
  <meta project="fmri-elastic-net" mode="document" timestamp="2026-07-29T23:00:00-04:00" />
  <files_updated>
    <file path="README.md" changes="Added a new 'Interaction Modeling (Brain x Moderator)' section covering moderator types/coding, post-reduction interaction construction, moderator main-effect protection (ModeratorScaler), inference integration across Tier 1/Tier 2/selection frequency, and the multi-output interaction reporting gap as a known limitation.">
      <type>readme</type>
    </file>
    <file path="INPUT_SPECIFICATION.md" changes="Added moderator_col/moderator_type to the data_cols parameter table; added bootstrap_ci_method to the stats_params table; updated load_and_prep_data's documented return tuple from 7 to 8 values (moderator); documented the 5-step pipeline (added weight_transformer and mod_scaler steps) in Step 3; documented moderator handling in Step 4 (Nested CV), Step 5 (Hotelling T-squared for K>2 nominal Tier 1), Step 7 (Selection Frequency levels_override), and Step 8 (Bootstrap Partial Ridge refit, Firth fallback, 4-tuple _boot_task return); added a new Step 2b subsection documenting the six interaction-modeling helper functions (_code_moderator, _construct_interactions, _strip_protected, _partial_ridge_refit, _hotelling_t2, _firth_logistic); added 9 new rows to the Edge Cases and Known Limitations table covering moderator/Hotelling/Partial-Ridge/Firth edge cases.">
      <type>input_spec</type>
    </file>
    <file path="fmri-elastic-net.py" changes="Added a moderator parameter note to run_bootstrap's docstring for consistency with its three sibling stage functions (run_nested_cv, run_tier1_inference, run_selection_frequency), which already documented moderator handling. No functional code changed.">
      <type>docstring</type>
    </file>
  </files_updated>
  <aid_log>
    <status>created</status>
    <sections_modified>all (new file: Purpose, Scope, Tools Used, Development Workflow, Human Oversight, Audit Trail, References)</sections_modified>
  </aid_log>
  <coverage>
    <public_functions_documented>all interaction-modeling helpers already had complete docstrings prior to this pass (_strip_protected, _code_moderator, _construct_interactions, _firth_logistic, _partial_ridge_refit, _hotelling_t2, ModeratorScaler); run_bootstrap's docstring extended for consistency</public_functions_documented>
    <classes_documented>7/7 (CovariateScaler, ModeratorScaler, WeightTransformer, _ClusterPCABase, ClusterPCATransformer, AprioriTransformer, ICATransformer — all pre-existing, verified current)</classes_documented>
    <modules_with_docstrings>1/1</modules_with_docstrings>
  </coverage>
  <summary>
    User-facing documentation (README.md, INPUT_SPECIFICATION.md) previously had zero coverage of the interaction modeling feature (moderator_col/moderator_type, brain x moderator interactions, Partial Ridge logistic adaptation, Hotelling T-squared, Firth fallback) despite the feature being fully implemented and tested (753/753 passing) across Sessions 13-15. This pass closes that gap: README gained a dedicated Interaction Modeling section; INPUT_SPECIFICATION gained parameter table entries, updated function contracts (8-value load_and_prep_data return, 5-step pipeline, 4-tuple _boot_task return), a new helper-function reference subsection, and 9 new edge-case rows. The .aid/ AI-disclosure infrastructure (AID_LOG.md, project_claude.md, reports/) was created for the first time this session. A mandatory 5-agent security gate scan found no LLM-attribution violations and one corroborated PII finding (absolute local filesystem paths in a copied brainstorm report's context_files section, flagged by 4/5 agents), which was remediated to repo-relative paths and verified clean on re-scan.
  </summary>
</document_report>
