# Final Figure Specification

Status: frozen placeholder specification.

This document fixes the five figures for the current IEEE BigData paper draft.
Actual figure assets are intentionally deferred. Until assets are created, the
LaTeX draft should keep placeholders.

Claim-safety rule: no figure may imply production readiness, universal data
quality detection, legal/license certification, automatic safety clearance, or
Utility-driven Stage-B selection.

## Figure 1: Curation-Stage Pipeline

LaTeX label: `fig:pipeline`

Final caption:

> Curation-stage view of language-model training data management. Stage B is
> optional and budget-driven; Utility is observed only in Stage C and cannot
> flow back into the Stage-B selector objective.

Composition:

- Full-width horizontal pipeline with six boxes:
  - Raw candidate corpus
  - Stage 0: risk quarantine
  - Stage A: chunk-level hard gates
  - Full curated pool
  - Stage B: optional budgeted selection
  - Stage C: subset-level validation
- Add a dashed boundary around Stage C labeled `validation only`.
- Add a blocked backward arrow from Stage C to Stage B labeled
  `no Utility feedback`.
- Add a small note under the full curated pool:
  `retain-all is valid when no budget binds`.

Numbers:

- No numerical result required.
- Optional note: Stage B selected `1,424` of `1,913` Stage-A-pass code-domain
  records only when a budget was binding.

Tool prompt:

> Create a clean academic vector pipeline figure for an IEEE conference paper.
> Show six left-to-right boxes: Raw candidate corpus, Stage 0 risk quarantine,
> Stage A chunk-level hard gates, Full curated pool, Stage B optional budgeted
> selection, and Stage C subset-level validation. Use restrained colors, white
> background, dark text, and simple arrows. Stage C should be visually marked
> as validation only. Draw a blocked arrow from Stage C back to Stage B labeled
> "no Utility feedback". Add a small note under the curated pool: "retain-all
> is valid when no budget binds". Do not include production-ready, quality
> detector, legal clearance, or guaranteed improvement language.

## Figure 2: Core-Metric-Policy Map

LaTeX label: `fig:corepolicy`

Final caption:

> The framework treats metrics as policy-bound evidence, not as free-standing
> quality claims.

Composition:

- Compact matrix with five rows:
  - Validity
  - Selection Value Evidence
  - Redundancy
  - Coverage
  - Utility
- Four columns:
  - Core responsibility
  - Observable metric surface
  - Policy role
  - Claim boundary
- Mark Utility row as `Stage C only`.
- Mark Selection Value Evidence as `not intrinsic quality`.

Numbers:

- No numerical result required.

Tool prompt:

> Create a compact table-like vector matrix for an academic paper. Rows are
> Validity, Selection Value Evidence, Redundancy, Coverage, and Utility.
> Columns are Core responsibility, Observable metric surface, Policy role, and
> Claim boundary. Include these key boundaries: Validity is a Stage-A hard gate
> and not semantic quality; Selection Value Evidence is Stage-B ranking evidence
> and not intrinsic quality; Redundancy controls duplicate and saturation risk;
> Coverage tracks observable source/style/path/content/cluster retention;
> Utility is Stage-C validation only and never a selector objective. Use neutral
> colors and strong row separation.

## Figure 3: Utility Leakage Boundary

LaTeX label: `fig:utilityboundary`

Final caption:

> Utility is downstream validation evidence and is forbidden as a Stage-B
> selector objective.

Composition:

- Left panel: Stage-B selector inputs.
- Right panel: Stage-C validation outputs.
- Center: thick vertical boundary labeled `frozen policy boundary`.
- Allowed forward flow:
  `Stage-B selector -> selected subset -> Stage-C validation`.
- Blocked backward flow:
  `Stage-C outcomes -> Stage-B selector`, marked as forbidden.

Numbers:

- No numerical result required.

Tool prompt:

> Create a high-contrast academic flow diagram explaining Utility leakage
> control. On the left, show allowed Stage-B inputs: Stage-A pass chunks,
> selection value evidence, redundancy risk, coverage support metadata, and
> token budget. On the right, show Stage-C validation outputs: held-out NLL,
> guardrails, claim decision, pass/fail/abstain. Place a thick vertical
> boundary between them labeled "frozen policy boundary". Draw an allowed
> forward arrow from Stage B to Stage C through the selected subset. Draw a
> blocked backward arrow from Stage C to Stage B labeled "forbidden: outcome
> tuning". Make it readable in black and white.

## Figure 4: Natural-Budget Result Summary

LaTeX label: `fig:results`

Final caption:

> Historical Code Stage-C evidence is positive but requires a current-framework
> rerun; Math remains an abstain case.

Composition:

- Two-panel result figure.
- Panel A: Code natural-budget evidence.
  - Bar chart of mean NLL: base, raw, curated.
  - Small annotation for EvalPlus raw and curated.
  - Annotate token reduction.
- Panel B: Math natural-budget evidence.
  - Bar chart of mean NLL: base, raw, curated v2, curated v3.
  - Mark v2 as `fail`.
  - Mark v3 as `repair-only abstain`.
  - Annotate that lower NLL is better.

Numbers:

- Code:
  - Base NLL: `1.23427`
  - Raw NLL: `1.21000`
  - Curated NLL: `1.20104`
  - Raw packed training tokens: `980,992`
  - Curated packed training tokens: `385,024`
  - Token reduction: `60.8%`
  - Raw EvalPlus macro pass rate: `51.06%`
  - Curated EvalPlus macro pass rate: `57.87%`
- Math:
  - Base NLL: `1.55945`
  - Raw NLL: `1.49565`
  - Curated v2 NLL: `1.52706`
  - Curated v3 NLL: `1.49899`
  - Raw packed training tokens: `1,120,256`
  - Curated v2 packed training tokens: `626,688`
  - Curated v3 packed training tokens: `1,026,048`
  - v2 token reduction: `44.1%`
  - v3 token reduction: `8.4%`

Tool prompt:

> Create a two-panel scientific result figure on a white background. Panel A
> is Code: bar chart of mean NLL for base 1.23427, raw 1.21000, and curated
> 1.20104, with lower-is-better annotation. Add a small note: EvalPlus raw
> 51.06%, curated 57.87%; curated uses 60.8% fewer packed training tokens than
> raw. Panel B is Math: bar chart of mean NLL for base 1.55945, raw 1.49565,
> curated v2 1.52706, and curated v3 1.49899. Mark v2 as fail and v3 as
> repair-only abstain. Add a note that v3 uses 8.4% fewer packed training
> tokens than raw but still does not beat raw. Use subdued colors and avoid a
> celebratory style.

## Figure 5: Evidence and Claim Boundary

LaTeX label: `fig:evidence`

Final caption:

> The paper claim is supported by frozen artifacts, while production and
> universal data-quality claims remain outside scope.

Composition:

- Layered evidence stack:
  - Commands, configs, and source scripts
  - Scoring artifacts and selector outputs
  - Audits and guardrails
  - Stage-C results
  - Bounded paper claim
- Side-by-side claim box:
  - Supported:
    - curation-stage framework
    - historical code-domain positive evidence; current rerun required
    - math-domain abstain is explicitly reported
    - Utility is Stage C only
    - canonical evidence rebuild path exists
  - Not supported:
    - production-ready deployment
    - universal data-quality detector
    - all-domain improvement guarantee
    - legal/license certification

Numbers:

- Canonical rebuild scripts: `7`
- Support reports: `6`
- Historical/experimental numbered scripts outside canonical path: `215`

Tool prompt:

> Create a layered evidence-stack diagram for an academic paper. Bottom layer:
> commands, configs, and source scripts. Next layer: scoring artifacts and
> selector outputs. Next layer: audits and guardrails. Next layer: Stage-C
> results. Top layer: bounded paper claim. On the side, include two columns:
> Supported and Not supported. Supported items: curation-stage framework,
> historical code-domain positive evidence requiring rerun, math-domain abstain explicitly reported, Utility is
> Stage C only, canonical evidence rebuild path exists. Not supported items:
> production-ready deployment, universal data-quality detector, all-domain
> improvement guarantee, legal/license certification. Include small numerical
> callouts: 7 canonical rebuild scripts, 6 support reports, 215 historical or
> experimental numbered scripts outside the canonical path. Use check marks
> only for supported claims and warning markers for unsupported claims.

## Deferred Figures

These are not part of the current five-figure paper package unless the paper
expands:

- Dataset curation funnel.
- Multi-domain benchmark roadmap matrix.
- Stage-B mechanism diagnostic.
- Equal-token benchmark result summary.
