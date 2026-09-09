<brainstorm_report>
  <meta project="fmri-elastic-net" mode="brainstorm" timestamp="2026-08-27T14:37:54-04:00" />
  <context_files>
    <file path="fmri-elastic-net.py" relevance="Primary pipeline script; _compute_importance_preamble (lines 3008-3041), calculate_visualization_data (lines 2841-2953), _partial_ridge_refit (lines 423-527), _boot_task (lines 2645-2838), run_nested_cv (lines 1596-1902)" />
    <file path="config_template.yaml" relevance="Config schema; new reference_class parameter for multi-class classification" />
    <file path="INPUT_SPECIFICATION.md" relevance="Documentation for new parameter and updated coefficient semantics" />
    <file path="README.md" relevance="User-facing documentation for coefficient interpretation" />
  </context_files>
  <topics>
    <topic id="T1" title="Fully standardized coefficients (std_coef redefinition)">
      <summary>The pipeline's std_coef column currently reports X-standardized coefficients (change in raw Y per 1 SD of X). Y is never standardized, so these values are unbounded (not constrained to [-1, 1]). Redefine std_coef as the fully standardized coefficient (dimensionless: SDs of Y per 1 SD of X), following established conventions for both regression and classification.</summary>
      <research>
        <finding src="Bring, 1994, The American Statistician 48:209-213">Three canonical coefficient types: raw (b), X-standardized (b_x), and fully standardized (beta). The standard reporting pair is raw + fully standardized (SPSS default).</finding>
        <finding src="Long, 1997; Mplus StdYX (UCLA OARC documentation)">For logistic regression, the latent variable interpretation gives SD(Y*) = sqrt(Var(X*beta) + pi^2/3), where pi^2/3 is the variance of the standard logistic error distribution. The fully standardized coefficient is b / SD(Y*). Implemented as Mplus StdYX.</finding>
        <finding src="Menard, 2004, The American Statistician 58(3); Menard, 2011, Social Forces 89(4):1409-1428">Evaluates six approaches to standardized logistic coefficients. Recommends the latent variable approach (SD of predicted logit) as the most defensible analog to linear regression beta.</finding>
        <finding src="Carroll et al., 2009; Bunea et al., 2011, NeuroImage">Neuroimaging multivariate/predictive models typically report X-standardized coefficients. No established convention for fully standardized beta in penalized regression.</finding>
        <finding src="Friedman &amp; Wall, 2005, The American Statistician">Standardized coefficients in multiple regression can exceed [-1, 1] due to suppressor variables and multicollinearity. The [-1, 1] bound applies only to the Pearson r in simple bivariate regression.</finding>
        <finding src="PV verification (3/3 concerns)">Three independent verification agents confirmed the binary classification approach (Long/Menard) but raised: (1) the fold-wise ensemble has no full-data model, so Var(X*beta) must be computed from cross-validated predictions; (2) sklearn multi-class uses softmax parameterization with translation indeterminacy, requiring reference-differencing (beta_k - beta_ref) before applying the pi^2/3 divisor; (3) the fixed divisor across bootstrap iterations is an approximation analogous to the existing raw_coef back-projection caveat.</finding>
      </research>
      <approaches>
        <approach id="A1" label="Fully standardized (decided)" feasibility="high" risk="low">
          <description>
            Redefine std_coef as the fully standardized coefficient. Compute from the model's internal (X-standardized) coefficient by dividing by the analysis-type-appropriate divisor:
            - Regression (single-output): SD(Y)
            - Multi-task regression: SD(Y_k) per task
            - Binary classification: SD(Y*) = sqrt(Var(CV logits) + pi^2/3), per Long (1997)/Menard (2004, 2011)
            - Multi-class classification: reference-difference (beta_k - beta_ref) per contrast, then SD(Y*_k) = sqrt(Var(X(beta_k - beta_ref)) + pi^2/3) per contrast

            The divisor is computed once from the fold-wise ensemble's cross-validated predictions (each subject predicted by their held-out fold). For classification, the cross-validated linear predictor (logit) variance is used.

            New config parameter: reference_class (mandatory for multi-class classification; pipeline halts if absent). Multi-class output changes from K rows to K-1 reference-coded contrasts.

            raw_coef derivation updated: both std_coef and raw_coef derived independently from the internal (X-standardized) coefficient:
            - std_coef = internal / SD(Y/Y*)
            - raw_coef = internal / SD(X)

            Column names unchanged: std_coef_mean, std_ci_low, std_ci_high, raw_coef_mean, raw_ci_low, raw_ci_high.

            Scale-invariant quantities unaffected: pd, p_value, is_significant, is_significant_fdr (zero-crossing and sign proportions are invariant under positive-constant division).
          </description>
          <pros>Dimensionless, scale-invariant coefficients; standard reporting convention (SPSS, Mplus); enables cross-study comparison; resolves the reported concern about values exceeding [-1, 1] (values will be smaller in magnitude, though still theoretically unbounded in multiple regression)</pros>
          <cons>Classification SD(Y*) is model-dependent (unlike regression SD(Y)); multi-class extension (per-contrast latent variable) is principled but unpublished; fixed divisor across bootstrap iterations is an approximation</cons>
          <statistical_considerations>
            Documentation caveats required:
            1. For classification, SD(Y*) is model-dependent (Menard, 2011): it changes as predictors are added or removed, unlike SD(Y) for regression.
            2. The divisor is computed once from cross-validated predictions and applied uniformly to all bootstrap statistics (approximation analogous to existing raw_coef back-projection caveat, line 60-63).
            3. Fully standardized coefficients remain unbounded in multiple regression (Friedman and Wall, 2005). The [-1, 1] bound applies only to simple bivariate regression.
            4. Multi-class per-contrast latent-variable standardization is a principled extension of the binary-case derivation but lacks direct published validation for multinomial softmax models.
          </statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">User confirmed: redefine std_coef as fully standardized for all analysis types. Column names stay as std_coef_*. Multi-class uses mandatory reference_class config parameter with hard halt if absent. Reference-difference applied before standardization.</decision>
    </topic>
    <topic id="T2" title="Interaction plotting residualization (full conditional relationship)">
      <summary>The interaction plotting CSV (report_{level}_interaction_plotting.csv) currently isolates the pure interaction term only, removing constituent main effects. The standard convention for residualized interaction plots (Long, 2019, interactions R package; jtools partialize()) retains the full conditional relationship: predictor main effect + moderator main effect + interaction. The reported expectation aligns with this convention.</summary>
      <research>
        <finding src="Long, 2019, interactions R package (CRAN); jtools partialize() function">The standard partial-residual interaction plot removes covariates and other non-involved predictors but retains all three moderation components (X main, M main, X*M interaction) plus the model residual. The partialize() function computes: partial_residual_Y = Y - sum(b_covariate_j * covariate_j).</finding>
        <finding src="Aiken &amp; West, 1991; Hayes, 2022, Introduction to Mediation, Moderation, and Conditional Process Analysis">The standard simple-slopes/conditional-effects plot shows predicted Y at specific moderator levels with covariates at means. Main effects are inherently included because the prediction uses the full moderation equation.</finding>
        <finding src="Wikipedia: Partial residual plot">Partial residuals retain the effect of the variable(s) of interest while removing the effects of other variables.</finding>
      </research>
      <approaches>
        <approach id="A1" label="Full conditional relationship (decided)" feasibility="high" risk="low">
          <description>
            Modify calculate_visualization_data (lines 2935-2938) so that when effect_type == 'interaction', lin_contrib includes all three moderation components:
            - Brain feature main effect: beta_brain_f * f_scaled
            - Moderator main effect: beta_mod * M_coded
            - Interaction: beta_int_f * f_scaled * M_coded

            The partial residual becomes:
            y_val = Y - (linear_pred_full - lin_contrib)
                  = model_residual + brain_main + moderator_main + interaction

            This produces the standard simple-slopes visualization: different regression lines at different moderator levels, where slope at M_k = beta_brain + beta_int * M_k_coded and intercept shift = beta_mod * M_k_coded.

            The main-effect CSV remains unchanged (isolates brain feature main effect only).

            Cross-cutting with T1: After T1 redefines std_coef, the visualization function must use raw_coef_mean * (f_val_raw - mean) instead of std_coef_mean * f_val_scaled for partial residual computation. Both are algebraically equivalent in Y units:
            raw_coef * (X - mean_X) = (beta_internal / SD_X) * (X - mean_X) = beta_internal * X_scaled
            This approach is insensitive to the T1 redefinition because raw_coef remains "change in Y per unit of X."
          </description>
          <pros>Matches field standard (Long 2019, Aiken and West 1991, Hayes 2022); produces interpretable simple-slopes plots; retains observed data points for visual assessment of fit</pros>
          <cons>Requires looking up brain main-effect and moderator coefficients in addition to the interaction coefficient (minor implementation complexity)</cons>
          <statistical_considerations>The partial residual is an approximation when the visualization model (single fold's best_model) differs from the bootstrap-averaged coefficients. This is an existing approximation in the visualization function, not introduced by this change.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">User confirmed: change interaction CSV to retain full conditional relationship (brain main + moderator main + interaction). Switch coefficient source to raw_coef_mean for partial residual computation.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="T1: Redefine std_coef as fully standardized coefficient across all analysis types. Compute SD(Y) for regression, SD(Y*) for classification from fold-wise cross-validated predictions. Add reference_class config parameter (mandatory for multi-class, hard halt if absent). Reference-difference multi-class coefficients before standardization. Update raw_coef derivation to derive from internal coefficient independently. Update _compute_importance_preamble, run_nested_cv (return cross-validated logits), run_bootstrap (pass SD(Y/Y*)), config validation in main()." />
    <item priority="P0" target_mode="implement" description="T2: Change interaction CSV residualization to retain full conditional relationship (brain main + moderator main + interaction). Switch visualization coefficient source from std_coef_mean * f_val_scaled to raw_coef_mean * (f_val_raw - mean). Requires looking up brain main-effect coefficient and moderator coefficient(s) from the fitted model. Modify calculate_visualization_data lines 2925-2940." />
    <item priority="P0" target_mode="implement" description="T1 x T2 cross-cutting: Ensure visualization partial residual computation uses raw_coef (Y-unit scale), not std_coef (now fully standardized). Both T1 and T2 changes must be implemented together to maintain consistency." />
    <item priority="P1" target_mode="implement" description="T1: Add documentation caveats to module docstring, INPUT_SPECIFICATION.md, and README.md: (1) classification SD(Y*) model-dependence (Menard 2011), (2) fixed divisor approximation, (3) coefficients remain unbounded in multiple regression, (4) multi-class per-contrast standardization is principled but unpublished." />
    <item priority="P1" target_mode="test" description="Test fully standardized coefficients for all analysis types: regression, multi-task regression, binary classification, multi-class classification. Verify raw_coef and std_coef are independently derived and internally consistent. Verify visualization partial residuals use Y-unit-scale coefficients." />
    <item priority="P1" target_mode="test" description="Test multi-class reference_class config validation: pipeline halts when reference_class is absent for multi-class; accepts valid class labels; rejects invalid labels." />
    <item priority="P1" target_mode="test" description="Test interaction CSV residualization: verify output contains full conditional relationship (not just interaction term). Verify main-effect CSV is unchanged." />
  </action_items>
  <next_steps>Proceed to /implement to generate the tech spec and build all changes. T1 and T2 should be implemented together due to the cross-cutting partial-residual concern. Then /test to verify all analysis types, /document to update documentation with caveats, /publish.</next_steps>
</brainstorm_report>
