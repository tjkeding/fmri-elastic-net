# AI Development Log

This document discloses the use of AI-assisted development tools in the creation of the **fmri-elastic-net** analysis pipeline, in accordance with emerging best practices for transparency in scientific software development.

---

## 1. Purpose

This document provides a structured disclosure of AI tool usage during the development of the fmri-elastic-net pipeline. The disclosure follows the AI Disclosure (AID) Framework (Weaver, 2025) and adheres to recommendations for responsible AI use in scientific computing (Bridgeford et al., 2025; Nussberger et al., 2024; Jamieson et al., 2024). The intent is to ensure that reviewers, collaborators, and end users can assess the nature and extent of AI involvement in the development process.

## 2. Scope

AI assistance was utilized for **analysis pipeline development**, encompassing:

- Code architecture and design decisions for an elastic net predictive modeling pipeline supporting regression (single-task and multi-task) and classification (binary and multi-class)
- Statistical methodology review and validation, including nested cross-validation, bootstrap confidence interval construction, Partial Ridge refitting, Firth-penalized logistic regression, and interaction modeling
- Implementation of pipeline modules (dimensionality reduction, covariate handling, sample weighting, moderator interaction modeling, two-tier inference)
- Test suite development and validation (742+ unit and integration tests)
- Documentation authoring and refinement (README, INPUT_SPECIFICATION, code docstrings)

AI was **not** used for:

- Running analyses on real data
- Interpreting scientific results from pipeline outputs
- Making domain-specific methodological decisions (e.g., selection of covariates, outcome definitions, analysis mode selection, moderator variable choice, or study-specific analytical choices)

The fmri-elastic-net pipeline is a general-purpose tool for activation and connectome predictive modeling. AI-assisted development covered the pipeline's implementation, testing, and documentation. All statistical and methodological decisions were made by the researcher; AI tools were used to implement, test, and critically review those decisions against the statistical literature.

## 3. Tools Used

Development utilized **Claude Code** (Anthropic), employing two model tiers:

| Model | Use Case | Tasks |
|-------|----------|-------|
| Claude Opus 4 | Analytical and review | Critical review of statistical methods, brainstorming sessions, code quality audits, risk assessment, and architectural decisions |
| Claude Sonnet 4 | Implementation | Code generation, test implementation, documentation drafting, and file management |

This dual-model approach ensured that analytical depth (Opus) was applied to decisions with statistical or methodological consequences, while implementation efficiency (Sonnet) was used for well-specified coding tasks under explicit human direction.

## 4. Development Workflow

The pipeline was developed through an iterative, mode-based workflow with the following stages:

1. **Brainstorm** -- Structured discussion of design decisions, trade-offs, and alternative approaches. Every brainstorm session produced a report with explicit decision records (accepted, rejected, deferred). Topics included: fold-wise ensemble architecture vs. full-data refit, two-tier inference design, Partial Ridge CI debiasing, interaction modeling architecture, and Firth logistic separation handling.

2. **Critical Review (CR)** -- Formal review of the codebase for statistical correctness, robustness, reproducibility, and defensive coding practices. Review dimensions included: assumptions, validity (mathematical correctness, bias/confounding), robustness, generalizability, reproducibility, inferential calibration, and scientific rigor. Each finding was classified by severity (P0/P1/P2) and required explicit human triage (accept, reject, or modify).

3. **Implement (Plan + Build)** -- Implementation proceeded in two sub-phases: (a) a technical specification mapping each approved change to specific code modifications with risk assessment and rollback strategy, and (b) execution of the specification. All plans required human approval before code generation began. No changes were self-scoped by the AI tool.

4. **Test** -- Comprehensive test suite development covering unit, integration, edge-case, and statistical invariant tests (known-answer tests, distribution checks, back-transformation verification). Tests were designed independently of the implementation to detect regressions and validate statistical properties. Every failing test required a formal disposition classification (aligned, obsolete-test, product-bug, or ambiguous) before any test modification.

5. **Clean** -- Code quality review for consistency, style, and maintainability.

6. **Document** -- Authoring and updating of user-facing documentation (README, INPUT_SPECIFICATION) and machine-readable technical specifications. Documentation claims were verified against actual code behavior.

Key properties of this workflow:

- All decisions required **explicit human approval** before implementation.
- The pipeline was developed with a **test-first** approach.
- Every statistical and algorithmic choice was subjected to **formal critical review**, with findings documented and triaged individually.

## 5. Human Oversight

The researcher maintained full oversight and decision authority throughout the development process:

- **(a)** Defined all statistical methodology and analytical approach, including: the two-tier inference design (liberal Tier 1 screen, conservative Tier 2 confirmation), the fold-wise ensemble architecture, the Partial Ridge CI debiasing strategy, the interaction modeling architecture (post-reduction construction, heredity-preserving main effect protection), and the Firth logistic fallback threshold.

- **(b)** Triaged every critical review finding with explicit accept/reject/modify decisions, documented in brainstorm reports with rationale for each determination. Multiple CR rounds were conducted per implementation phase.

- **(c)** Approved all implementation plans (technical specifications) before any code generation was executed. Each plan itemized specific code changes, dependencies, risk levels, and rollback strategies.

- **(d)** Validated all test results and ensured test coverage aligned with the statistical guarantees required by the pipeline. Test failures were investigated collaboratively, with the researcher making final determinations on whether failures indicated implementation bugs or test specification drift.

- **(e)** Made all domain-specific decisions regarding pipeline architecture, algorithmic choices, and analytical strategy, including: whether to support outer-fold repeats (removed in favor of the fold-wise ensemble architecture), weight normalization convention (sum-to-N, Hajek estimator), moderator coding scheme (deviation coding for nominal, mean-centering for continuous), and Partial Ridge logistic extension parameters (C=1, sqrt(n) scaling).

## 6. Audit Trail

A complete record of the structured development process is available in the `.aid/reports/` directory within this repository. The audit trail includes:

- **Brainstorm reports** -- Records of design discussions, decision rationale, and trade-off analyses.
- **Critical review reports** -- Formal findings with severity classifications and human triage decisions.
- **Implementation plans** -- Technical specifications mapping approved changes to code modifications.
- **Implementation build reports** -- Records of executed changes with deviation notes.
- **Test reports** -- Test suite results and coverage summaries.
- **Code quality reviews** -- Clean-pass reports on style and consistency.
- **Documentation reports** -- Records of documentation updates and revisions.

The project-level configuration file used to guide AI interactions is preserved as `.aid/project_claude.md`.

Raw session transcripts are excluded for privacy reasons. The structured reports above capture all substantive technical decisions, rationale, and implementation details.

## 7. Version History

- **2026-07-29**: Interaction modeling (brain x moderator) implementation, CR-driven fixes (logistic Partial Ridge back-transformation, multi-output selected_mask, Hotelling degeneracy guards, levels_override, documentation updates), comprehensive test suite expansion (742 to 753 tests), and AID infrastructure creation. Reports synced: 2 brainstorm, 1 CR, 2 implement (plan + build) for multi-output interaction reporting, 1 implement (plan + build) for CR-driven fixes, 1 test, 1 document.

## 8. References

- Bridgeford, E. W., et al. (2025). Ten simple rules for AI-assisted coding in science. *arXiv preprint*, arXiv:2510.22254.

- Jamieson, A. J., et al. (2024). Protecting scientific integrity in an age of generative AI. *Proceedings of the National Academy of Sciences*, 121(41), e2407886121.

- Nussberger, A.-M., et al. (2024). Ten simple rules for using large language models in science. *PLOS Computational Biology*, 20(7), e1012291.

- Weaver, J. B. (2025). The AI Disclosure (AID) Framework. *arXiv preprint*, arXiv:2408.01904v2.
