# Figure Generation Briefs

Use these briefs to create clean vector-style figures for the IEEE BigData
paper draft. The figures must be claim-safe: do not imply production readiness,
universal quality detection, or Utility-driven selection.

`paper/final_figure_spec.md` is the frozen source of truth for the current
five-figure package. This file is a tool-facing brief. Figures 6 through 9 are
deferred and should not be added to the current draft unless the paper expands.

## Figure 1: Curation-Stage Pipeline

Create a horizontal pipeline diagram for a conference paper.

Visual structure:
- Six boxes connected left to right with arrows.
- Box 1: "Raw candidate corpus"
- Box 2: "Stage 0: risk quarantine"
- Box 3: "Stage A: chunk-level hard gates"
- Box 4: "Full curated pool"
- Box 5: "Stage B: optional budgeted selection"
- Box 6: "Stage C: subset-level validation"

Small subtitles under each box:
- Raw candidate corpus: "collected data"
- Stage 0: "PII, secrets, rights, contamination, poisoning"
- Stage A: "structural validity + hard dedup"
- Full curated pool: "retain-all is allowed"
- Stage B: "pre-outcome ranking under token budget"
- Stage C: "held-out training evidence + guardrails"

Design requirements:
- Use a restrained academic color palette: dark gray text, light blue Stage A/B/C boxes, light red quarantine box, light green curated pool.
- Make Stage C visually separated with a dashed boundary labeled "validation only".
- Add a small blocked arrow from Stage C back to Stage B with the label "no Utility feedback into selector".
- Output should be vector-like, clean, flat, and readable at IEEE two-column width.

Do not include:
- "quality detector"
- "production ready"
- "automatic legal clearance"
- "guaranteed improvement"

## Figure 2: Core-Metric-Policy Mapping

Create a matrix diagram that shows how each Core maps to metrics and policy
roles.

Rows:
- Validity
- Selection Value Evidence
- Redundancy
- Coverage
- Utility

Columns:
- Core responsibility
- Observable metric surface
- Policy role
- Claim boundary

Key content:
- Validity -> structural usability -> Stage-A hard gate -> not semantic quality.
- Selection Value Evidence -> observable pre-outcome evidence -> Stage-B ranking -> not intrinsic quality.
- Redundancy -> canonical-exact duplicate and fuzzy near-duplicate risk -> exact-only Stage-A gate and reversible Stage-B penalty -> not recall-complete duplicate proof.
- Coverage -> source/style/path/content/cluster retention -> Stage-C diagnostic/validator -> not true domain coverage without metadata.
- Utility -> held-out training effect -> Stage-C validation only -> never selector objective.

Design requirements:
- Use a readable table-like grid, not a dense infographic.
- Highlight Utility row in a distinct neutral color and mark it "Stage C only".
- Keep all wording short enough for a two-column or full-width figure.

## Figure 3: Utility Leakage Boundary

Create a flow diagram explaining why Utility cannot enter Stage B.

Left side:
- "Stage-B selector inputs"
- Include allowed inputs: "Stage-A pass chunks", "selection value evidence", "redundancy risk", "coverage support metadata", "token budget".

Right side:
- "Stage-C validation outputs"
- Include outputs: "held-out NLL", "guardrails", "claim decision", "abstain/reject/pass".

Between them:
- A thick vertical boundary labeled "frozen policy boundary".
- Allowed forward arrow: Stage B -> selected subset -> Stage C.
- Blocked backward arrow: Stage C -> Stage B, marked with a red X and label "forbidden: outcome tuning".

Caption intent:
"Utility is validation evidence, not a selector objective."

Design requirements:
- Very simple, high contrast, no decorative graphics.
- Must be readable in black and white.

## Figure 4: Natural-Budget Result Summary

Create a compact two-panel results figure for the current natural-budget
paper evidence.

Panel A:
- Bar chart of Code mean NLL for base 1.23427, raw 1.21000, and curated
  1.20104.
- Add a small EvalPlus annotation: raw 51.06%, curated 57.87%.
- Annotate: "curated uses 60.8% fewer packed training tokens than raw."

Panel B:
- Bar chart of Math mean NLL for base 1.55945, raw 1.49565, curated v2
  1.52706, and curated v3 1.49899.
- Annotate: "v2 fails; v3 repairs v2 but still does not beat raw."
- Annotate: "v3 uses 8.4% fewer packed training tokens than raw."

Design requirements:
- Lower NLL is better.
- Use subdued scientific colors.
- Do not make the result look universally successful; the point is pass plus
  abstain under one validation boundary.

## Figure 5: Evidence Package and Claim Boundary

Create a layered evidence-stack diagram.

Bottom layer:
- "Commands + configs + source scripts"

Next layer:
- "Scoring manifests"
- "Core audits"
- "Stage-C validation"
- "Comparison tables"

Next layer:
- "Paper reproducibility manifest"

Top layer:
- "Bounded paper claim"

Next to the top layer, show two columns:
- Supported:
  - "curation-stage framework"
  - "code-domain equal-token NLL improvement"
  - "Utility is Stage C only"
- Not supported:
  - "production-ready deployment"
  - "universal quality detector"
  - "legal/license certification"

Design requirements:
- Use a stacked architecture diagram.
- Use check marks only for supported claims and warning icons for unsupported claims.
- Avoid celebratory or marketing style; this is an academic evidence-boundary figure.

## Deferred Figure 6: Dataset Curation Funnel

Create a funnel or stepped bar chart showing how a raw corpus becomes a
training payload.

Stages:
- Raw collected records
- Stage 0 quarantined / allowed
- Stage A pass / fail
- Full curated pool
- Stage B selected subset
- Equal-token training payload

Required labels:
- Record count
- Chunk count
- Token proxy
- Retained fraction

Design requirements:
- Use neutral gray for total input, muted red for quarantined/fail counts, and blue/green for retained or selected counts.
- Make clear that Stage B is optional and only applies when the training budget binds.
- Include a small note: "Retain-all is valid when budget allows."

## Deferred Figure 7: Multi-Domain Benchmark Matrix

Create a matrix showing the planned validation expansion.

Rows:
- Code
- Math
- General text / instruction

Columns:
- Raw mixed pool
- Known high-quality reference pool
- Equal-token training arms
- Primary benchmark
- Guardrails

Content:
- Code primary benchmark: held-out code NLL, SWE-bench Lite/Verified when compute allows.
- Math primary benchmark: held-out math NLL, GSM8K, MATH.
- General text / instruction primary benchmark: held-out general-text NLL and instruction-following evaluation.
- Guardrails: EvalPlus, general-text retention, task retention, PII/risk quarantine.

Design requirements:
- This is a roadmap figure, not a completed-results figure.
- Use "Completed" only for current code-domain NLL evidence.
- Use "Frozen next-tier protocol" for planned but incomplete benchmark rows.

## Deferred Figure 8: Stage-B Mechanism Diagnostic

Create a two-panel figure explaining why the selected subset differs from the
budget-not-selected subset.

Panel A:
- Bar chart of selected share versus budget-not-selected share for concise useful candidate, concise example support, template or boilerplate risk, and bugfix or regression test signal.

Panel B:
- Bar chart of selected mean versus budget-not-selected mean for structural richness, lexical or identifier diversity, code quality proxy, and soft redundancy risk.

Design requirements:
- Make lower redundancy risk visually favorable.
- Caption intent: "Stage B balances usefulness, redundancy, and coverage support."
- Include the pool sizes: 1,913 Stage-A-pass records, 1,424 selected, 489 budget-not-selected.

## Deferred Figure 9: Equal-Token Benchmark Result Summary

Create a compact result figure for the current code-domain experiment.

Panel A:
- Bar chart of mean held-out code NLL for Raw random 1.20424, Stage-A random 1.20877, Curated 1.20165, and Known high-quality reference 1.20510.
- Lower is better.

Panel B:
- Bar chart of Stage-B ablation redundancy risk for Full selector 0.19243, Quality only 0.22203, Redundancy only 0.04087, and No coverage support 0.18621.

Design requirements:
- Use clear y-axis labels and avoid overstating the small NLL differences.
- Add a visual annotation: "Curated beats Stage-A random by 0.00712 mean NLL."
- Do not use celebratory colors; make this look like a scientific result summary.

## Preferred Export

- SVG or PDF vector preferred.
- Also export PNG at 300 DPI for preview.
- Use white background.
- Use fonts similar to Times/Helvetica.
- Keep all labels editable if the tool supports it.
