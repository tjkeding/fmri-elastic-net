<document_report>
  <meta project="fmri-elastic-net" mode="document" timestamp="2026-09-09T13:37:00Z" />
  <note>This pass documented the remaining gaps from the Session 19 implementation (std_coef redefinition, interaction-visualization residualization, K&gt;2 nominal moderator broadcast fix, run_bootstrap cv_data required-parameter change) validated by the 2026-09-09 test cycle (855/855 passing). Pre-existing uncommitted documentation for std_coef, reference_class, and model_performance_per_fold/fold_summary was found already present in README.md and INPUT_SPECIFICATION.md and verified accurate against the code; it was left unchanged rather than reworked.</note>
  <files_updated>
    <file path="README.md" changes="Added one sentence to the '### Interaction visualization' subsection documenting that interaction-effect partial associations now reflect the full conditional relationship (brain main + moderator main + interaction), computed via out-of-fold (OOF) ensemble linear predictions.">
      <type>readme</type>
    </file>
    <file path="INPUT_SPECIFICATION.md" changes="Corrected a stale formula description in '### report_{cluster|individual}_interaction_plotting.csv': replaced the old interaction-only partialling formula (f_weight * f_scaled * M_coded) with the current lin_contrib = brain_main + mod_main + int_contrib decomposition and its OOF-ensemble derivation.">
      <type>input_spec</type>
    </file>
    <file path="fmri-elastic-net.py" changes="Added an inline comment explaining the K&gt;2 nominal-moderator ndim==3 broadcast branch (extra K-1 contrast axis requiring an additional newaxis for the SD(Y) divisor). Added a cv_data parameter entry to the run_bootstrap docstring explaining its role (fold_assignments/fold_held_out consumption by _compute_oof_visualization_data and _compute_sd_y_divisor) and why it has no default. No functional code changed (verified via git diff and AST parse).">
      <type>docstring</type>
    </file>
    <file path="AID_LOG.md" changes="Section 2 (Scope): updated the test-count reference from 811+ to 855+. Section 5 (Human Oversight) bullet (a): appended the std_coef fully-standardized-coefficient definition and the interaction-visualization residualization scheme to the list of researcher-directed methodology decisions. Section 7 (Version History) was not touched, per the AID_LOG template doctrine (that section is authored exclusively by /publish).">
      <type>aid_log</type>
    </file>
  </files_updated>
  <aid_log>
    <status>updated</status>
    <sections_modified>Section 2 (Scope), Section 5 (Human Oversight)</sections_modified>
  </aid_log>
  <coverage>
    <public_functions_documented>72/81</public_functions_documented>
    <classes_documented>7/7</classes_documented>
    <modules_with_docstrings>1/1</modules_with_docstrings>
  </coverage>
  <security_gate>
    <dispatched>5</dispatched>
    <violations_found>0</violations_found>
    <files_scanned>README.md, INPUT_SPECIFICATION.md, fmri-elastic-net.py, AID_LOG.md</files_scanned>
    <result>Clean. All 5 independent agents (SG-1 through SG-5) reported zero PII/PHI and zero LLM-attribution (Tier 1 or Tier 2) violations across all four modified files. AID_LOG.md's standalone model-name references (Claude Opus 4, Claude Sonnet 4, Claude Code) fall under the closed exemption list and were correctly not flagged.</result>
  </security_gate>
  <summary>Documentation now fully reflects the Session 19 implementation and its 2026-09-09 test validation: the std_coef fully-standardized-coefficient redefinition, the interaction-visualization OOF-ensemble residualization, the K&gt;2 nominal moderator broadcast fix, and the run_bootstrap cv_data required-parameter change are all documented at the appropriate level (user-facing docs for behavior changes; code comments/docstrings for internal API/correctness fixes). The 9 undocumented functions in fmri-elastic-net.py are private helpers consistent with the project's "do not document trivial internal functions" convention. The project is ready for /publish.</summary>
</document_report>
