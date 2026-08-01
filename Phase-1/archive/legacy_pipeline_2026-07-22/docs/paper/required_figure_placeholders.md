# Required Figure Placeholders

This file maps the current LaTeX placeholders in
`paper/curation_stage_framework_ieee.tex` to the figures that should be created
before submission. Keep the figures claim-safe: do not imply production
readiness, universal quality detection, legal clearance, or Utility-driven
selection.

The source of truth for final captions, composition, numerical callouts, and
tool prompts is `paper/final_figure_spec.md`. Until actual figure assets are
created, keep the LaTeX draft on placeholders.

## Figure 1: Curation-Stage Pipeline

LaTeX label: `fig:pipeline`

Purpose: show the paper's main idea in one glance.

Required content:

- Raw candidate corpus
- Stage 0: risk quarantine
- Stage A: chunk-level hard gates
- Full curated pool
- Stage B: optional budgeted selection
- Stage C: subset-level validation
- A blocked feedback arrow from Stage C to Stage B labeled "no Utility feedback"

Design guidance:

- Full-width figure.
- Use a clean horizontal pipeline.
- Make Stage C visually distinct as validation-only.
- Add a small note that retain-all is allowed when no budget binds.

## Figure 2: Core-Metric-Policy Map

LaTeX label: `fig:corepolicy`

Purpose: make the framework look like a research method, not a log.

Required content:

- Rows: Validity, Selection Value Evidence, Redundancy, Coverage, Utility
- Columns: Core responsibility, observable metric surface, policy role, claim boundary
- Utility row must be marked "Stage C only"
- Selection Value Evidence must say "not intrinsic quality"

Design guidance:

- Use a compact matrix.
- Avoid dense prose.
- Prefer neutral colors and strong row separation.

## Figure 3: Utility Leakage Boundary

LaTeX label: `fig:utilityboundary`

Purpose: explain the most important methodological constraint.

Required content:

- Left: allowed Stage-B inputs
- Right: Stage-C outputs
- Middle: frozen policy boundary
- Forward arrow: selected subset enters Stage C
- Blocked backward arrow: Stage-C outcomes cannot tune Stage B

Design guidance:

- Should work in black and white.
- Use a red X or blocked symbol only on the forbidden feedback path.

## Figure 4: Natural-Budget Result Summary

LaTeX label: `fig:results`

Purpose: distinguish historical Code positive evidence requiring rerun from Math abstain.

Required content:

- Panel A: Code base/raw/curated NLL, plus EvalPlus raw/curated.
- Panel B: Math base/raw/curated-v2/curated-v3 NLL.
- Annotate token reduction:
  - Code curated uses 60.8% fewer packed training tokens than raw.
  - Math v2 uses 44.1% fewer packed training tokens than raw but fails.
  - Math v3 uses 8.4% fewer packed training tokens than raw and is
    repair-only abstain.

Design guidance:

- Scientific style, not celebratory.
- Explicitly mark lower NLL as better.
- Use "pass" for Code, "fail" for Math v2, and "abstain" for Math v3.

## Figure 5: Evidence and Claim Boundary

LaTeX label: `fig:evidence`

Purpose: replace internal gate/log language with an external evidence story.

Required content:

- Layer 1: commands, configs, source scripts
- Layer 2: scoring artifacts and selector outputs
- Layer 3: audits and guardrails
- Layer 4: Stage-C results
- Top: bounded research claim
- Side callout: unsupported claims
  - production-ready deployment
  - universal data-quality detector
  - legal/license certification

Design guidance:

- Stacked evidence diagram.
- Use check marks only for supported research claims.
- Use warning markers for unsupported claims.

## Optional Later Figures

These are useful if the paper grows beyond the current placeholder set:

- Stage-B mechanism diagnostic: selected vs budget-not-selected features.
- Dataset curation funnel: raw records to curated/token payload.
- Multi-domain roadmap matrix: Code, Math, General text/instruction.
