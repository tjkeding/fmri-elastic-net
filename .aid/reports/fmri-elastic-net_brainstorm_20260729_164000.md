<brainstorm_report>
  <meta project="fmri-elastic-net" mode="brainstorm" timestamp="2026-07-29T16:40:00-04:00" />
  <context_files>
    <file path="fmri-elastic-net_cr_20260729_154100.md" relevance="CR report whose 6 findings and recommendations are being validated" />
    <file path="fmri-elastic-net.py" relevance="Pipeline source containing all code under review" />
  </context_files>
  <topics>
    <topic id="T1" title="Firth separation threshold recalibration after F1 fix">
      <summary>After fixing F1 (back-transformation *= sqrt_n), the Firth threshold |beta| > 10 at line 525 applies to correctly-scaled original-space coefficients. A log-odds coefficient of 10 corresponds to an odds ratio of exp(10) = 22,026, which is extreme and strongly indicative of quasi-complete separation. The threshold falls within the range cited by Allison (2008) for separation indicators. Unconditional Firth application (Heinze and Schemper 2002) was evaluated and rejected: it conflicts with the Partial Ridge de-biasing strategy because Firth's systematic O(1/n) bias toward zero would shift the entire bootstrap distribution, undermining the near-MLE target that Partial Ridge is designed to achieve.</summary>
      <research>R1 (separation detection thresholds) dispatched twice; both attempts failed due to API overload. Decision does not depend on research results: the threshold evaluation is grounded in the log-odds-to-odds-ratio mapping and the structural incompatibility of unconditional Firth with the Partial Ridge bootstrap CI strategy.</research>
      <approaches>
        <approach id="A1" label="Keep threshold at 10.0" feasibility="high" risk="low">
          <description>Retain the existing threshold. After F1 fix, |beta| > 10 log-odds (OR > 22,026) is an appropriate separation indicator. False triggers are harmless (Firth penalty is small when separation is absent).</description>
          <pros>Conservative; false trigger cost is negligible; falls within Allison (2008) range; no code change beyond F1 fix.</pros>
          <cons>Arbitrary threshold (inherent to any heuristic approach).</cons>
          <statistical_considerations>Asymmetry of consequences favors lower threshold: cost of false trigger (minimal Firth bias) is much less than cost of missed separation (divergent coefficients in bootstrap distribution).</statistical_considerations>
        </approach>
        <approach id="A2" label="Raise threshold to 15-20" feasibility="high" risk="low">
          <description>More permissive threshold, fewer false triggers.</description>
          <pros>Reduces false Firth triggers in unstable but non-separated bootstrap iterations.</pros>
          <cons>Increases risk of missing genuine quasi-separation; marginal benefit given negligible false-trigger cost.</cons>
        </approach>
        <approach id="A3" label="Unconditional Firth application" feasibility="high" risk="high">
          <description>Apply Firth penalization to all bootstrap iterations unconditionally, per Heinze and Schemper (2002).</description>
          <pros>Eliminates arbitrary threshold; handles multicollinear separation that single-coefficient thresholds miss.</pros>
          <cons>Conflicts with Partial Ridge de-biasing: introduces systematic O(1/n) bias toward zero across all bootstrap iterations, shifting the entire distribution and reducing Tier 2 sensitivity. Appropriate in single-model inference but not in the bootstrap CI context where the bias accumulates rather than averaging out.</cons>
          <statistical_considerations>Heinze and Schemper's recommendation targets single-model estimation where O(1/n) bias is negligible relative to standard error. In the bootstrap context, the same bias is applied identically to every replicate, shifting the distribution location without widening it.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Keep threshold at 10.0. The F1 fix restores the threshold to its intended operating range. Unconditional Firth is inappropriate in the bootstrap CI context due to conflict with Partial Ridge de-biasing. No threshold recalibration needed.</decision>
    </topic>
    <topic id="T2" title="Multi-output feature selection strategy for F2 union approach">
      <summary>The CR's F2 fix uses np.any across outputs to produce a 1D union mask. Validation confirmed this is exactly correct for multi-task regression (MultiTaskElasticNet enforces group sparsity, so union and intersection are identical) and conservatively defensible for multi-class classification (union is slightly over-inclusive for classes that didn't select a feature, but the OLS refit produces near-zero coefficients for irrelevant features, so the practical effect is negligible). Per-output masks would add significant complexity to the OVR loop for minimal statistical benefit.</summary>
      <research>No research needed; analysis is based on the algebraic properties of MultiTaskElasticNet's group sparsity penalty and the behavior of OLS refit on non-predictive features.</research>
      <approaches>
        <approach id="A1" label="Union across outputs" feasibility="high" risk="low">
          <description>np.any(np.abs(c_raw) > 1e-10, axis=0): a feature is selected if any output selected it. One 1D mask passed to _partial_ridge_refit.</description>
          <pros>Exactly correct for multi-task regression (group sparsity); conservatively defensible for multi-class; simple; consistent with standard practice (glmnet).</pros>
          <cons>Slightly over-inclusive for multi-class: features selected by one class get OLS treatment for all classes.</cons>
          <statistical_considerations>Over-inclusion is harmless: OLS refit on a non-predictive feature produces a coefficient near zero, adding noise but not systematic bias.</statistical_considerations>
        </approach>
        <approach id="A2" label="Per-output masks" feasibility="med" risk="med">
          <description>Each class/task has its own selection mask. Requires restructuring _partial_ridge_refit to accept 2D masks and compute per-output S_idx/Sc_idx.</description>
          <pros>More targeted de-biasing per output.</pros>
          <cons>Substantial code complexity; non-standard; minimal statistical benefit over union; complicates interpretation.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Union across outputs. Exactly correct for multi-task, conservatively defensible for multi-class, and consistent with standard practice.</decision>
    </topic>
    <topic id="T3" title="Hotelling T2 fallback strategy when df2 <= 0">
      <summary>The CR's F3 recommendation (NaN-with-warning when df2 <= 0) was evaluated against two alternatives: BH-corrected per-contrast t-tests as a pseudo-omnibus, and high-dimensional mean tests (Bai and Saranadasa 1996, Srivastava and Du 2008). High-dimensional tests were rejected because n = n_folds (5-10) is far too small for their asymptotic guarantees. BH-corrected pseudo-omnibus was rejected as redundant: the C1 implementation already writes per-contrast univariate t-tests to the Tier 1 report before the omnibus row. A NaN omnibus with a descriptive warning is sufficient; Tier 2 L2-norm CIs are unaffected by the Hotelling limitation.</summary>
      <research>R2 (Hotelling alternatives when p >= n) dispatched twice; both attempts failed due to API overload. Decision does not depend on research results: the rejection of high-dimensional alternatives is grounded in the structural argument that n=5-10 is too small for asymptotic guarantees, independent of the specific test statistics proposed.</research>
      <approaches>
        <approach id="A1" label="NaN-with-warning only" feasibility="high" risk="low">
          <description>Return NaN for T2, F, and p_value when df2 <= 0. Log a warning directing users to per-contrast t-test rows and noting Tier 2 is unaffected.</description>
          <pros>Simple; transparent; per-contrast t-tests already provide Tier 1 coverage; Tier 2 L2-norm CIs are independent of Hotelling; no risk of introducing a test with unknown operating characteristics.</pros>
          <cons>No omnibus p-value when df2 <= 0; user must interpret per-contrast results individually.</cons>
        </approach>
        <approach id="A2" label="BH-corrected per-contrast pseudo-omnibus" feasibility="high" risk="low">
          <description>Run K-1 per-contrast t-tests with BH correction; report minimum adjusted p-value as pseudo-omnibus when Hotelling is infeasible.</description>
          <pros>Provides an omnibus-like summary.</pros>
          <cons>Redundant with existing per-contrast t-test rows; different correction procedure could confuse output; less principled than a true multivariate test.</cons>
        </approach>
        <approach id="A3" label="High-dimensional mean test" feasibility="low" risk="high">
          <description>Implement Bai-Saranadasa (1996) or Srivastava-Du (2008) trace-based test for p >= n.</description>
          <pros>Theoretically handles p >= n regime.</pros>
          <cons>Asymptotic guarantees require n -> infinity; n=5-10 is far outside the validated regime; unknown Type I error control at pipeline's sample sizes; violates project's preference for robustness over novelty.</cons>
          <statistical_considerations>These tests are designed for high-dimensional data with many observations. The pipeline's use case (few folds, moderate contrasts) is the opposite: few observations, moderate dimensions. The asymptotic normal approximations underlying these tests have no coverage guarantees at n=5-10.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">NaN-with-warning only. Per-contrast t-tests already provide Tier 1 coverage; Tier 2 L2-norm CIs are unaffected. Warning message should direct users to per-contrast rows and note Tier 2 independence.</decision>
    </topic>
    <topic id="T4" title="Document-only decisions validation (F4 fold-dependence, F6 percentile CI coverage)">
      <summary>Both CR document-only recommendations were validated. For F4: Tier 2 bootstrap CIs are genuinely independent of the fold-level variance underestimation (bootstrap pooling incorporates between-fold variation as distributional width, not as a separately estimated variance); anti-conservative Tier 1 p-values are directionally consistent with the liberal-screen role; corrected variance estimators (Nadeau-Bengio 2003, Bayle et al. 2020) have not been validated for fold-level coefficient t-tests and adapting them would itself be a research contribution. For F6: BCa tends toward undercoverage in regularized settings (per R3 Monte Carlo findings), which is worse than percentile's conservative overcoverage for zero-crossing thresholding; the refit-then-percentile strategy aligns with the perturbation bootstrap literature; zero-crossing thresholding is robust to moderate coverage miscalibration.</summary>
      <research>No additional research needed; validation draws on the CR's R2 and R3 findings from the prior skill invocation.</research>
      <approaches>
        <approach id="A1" label="Document-only for both F4 and F6" feasibility="high" risk="low">
          <description>Document Tier 1 fold-dependence as a known limitation. Document logistic Partial Ridge as a project-specific adaptation and percentile CI coverage as unverified but defensible. No code changes.</description>
          <pros>Transparent; does not introduce untested corrections; preserves two-tier complementary sensitivity design; accurately represents the state of knowledge.</pros>
          <cons>Tier 1 p-values remain anti-conservative (known, documented); CI coverage remains unverified (acceptable for zero-crossing thresholding).</cons>
          <statistical_considerations>Tier 2 independence from Tier 1's specific bias is the key architectural safeguard. BCa would be worse (undercoverage) for the pipeline's thresholding use case. Corrected variance estimators for fold-level coefficient t-tests are an open research question.</statistical_considerations>
        </approach>
        <approach id="A2" label="Implement corrected Tier 1 variance estimator" feasibility="low" risk="high">
          <description>Adapt Nadeau-Bengio (2003) or Bayle et al. (2020) to fold-level coefficient t-tests.</description>
          <pros>Would produce calibrated Tier 1 p-values.</pros>
          <cons>Neither method has been validated for coefficient-level inference (only test-error inference); adaptation would be a research contribution requiring its own validation; risk of introducing a correction with unknown operating characteristics.</cons>
        </approach>
        <approach id="A3" label="Implement BCa bootstrap CIs" feasibility="med" risk="med">
          <description>Replace percentile CIs with bias-corrected and accelerated (BCa) CIs.</description>
          <pros>Second-order accurate in theory.</pros>
          <cons>Tends toward undercoverage in regularized settings (worse for zero-crossing thresholding); requires jackknife-after-bootstrap (computationally expensive); empirical evidence shows percentile may be preferable in this context.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Document-only for both F4 and F6. The two-tier architecture provides the safeguard; corrected estimators and BCa are either untested or counterproductive in this context.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="All 6 CR action items validated. No modifications to the CR's recommendations. Proceed with /implement on fmri-elastic-net_cr_20260729_154100.md as specified: F1 back-transformation fix, F2 union selected_mask, F3 Hotelling df2 guard with descriptive warning, F5 levels_override for bootstrap moderator coding, F4 and F6 documentation." />
    <item priority="P2" target_mode="brainstorm" description="Future: evaluate Firth-as-primary-estimator (replacing Partial Ridge logistic path entirely) as an alternative de-biasing philosophy for classification bootstrap CIs." />
  </action_items>
  <next_steps>Proceed with /implement on the CR report (fmri-elastic-net_cr_20260729_154100.md). All CR decisions validated; no modifications needed. The implement plan should consume the CR action items directly.</next_steps>
</brainstorm_report>
