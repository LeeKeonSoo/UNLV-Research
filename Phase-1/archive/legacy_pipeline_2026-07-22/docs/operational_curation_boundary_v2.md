# Operational Curation Boundary (v2)

## Purpose

The framework is an auditable language-model training-data curation layer. It
does not claim to calculate intrinsic data quality or to guarantee that every
incoming corpus should be made smaller.

```text
candidate corpus
  -> Stage A safety, provenance, and quarantine
  -> Stage B structural hard gate
  -> Stage C optional budget allocation
  -> curated output

frozen curated output
  -> External Evaluation Protocol
```

## What Counts as Curation

The curation engine is **Stage A, Stage B, and Stage C**. Its output is either:

- a full curated pool when no declared training budget binds;
- a budgeted training subset when a declared token or compute budget binds; or
- `abstain` when the available pre-outcome evidence cannot support allocation.

Stage B removes only content covered by explicit structural hard rules. Stage C
does not decide whether data is intrinsically good or bad. It allocates a
declared budget over Stage-A survivors using frozen pre-outcome evidence. A
`budget_not_selected` record remains in the full curated pool and is neither
rejected nor labeled low quality.

## What External Evaluation Is

The External Evaluation Protocol is an **offline validation and governance protocol**, not a
curation stage. It trains and evaluates a frozen output to establish a scoped
research or release claim. It may not re-rank, mutate, or reselect that output,
and Utility, NLL, and benchmark outcomes may never feed back into Stage C for
the same frozen cycle.

The primary operational comparison is natural-budget `Raw-safe` versus
`Curated`: each arm trains on the amount of data that its own pipeline produces.
A same-token random control is optional research evidence only when claiming
that a budget selector outperforms random selection; it is not a runtime
framework requirement.

## Current 5M Code Artifact

The 5M collection target produced a 7.03M-token legacy-Stage-0 `Raw-safe` corpus and
has completed legacy Stage 0-A-B materialization. Under the current nomenclature,
that is Stage A-B-C. External validation is now running:
three natural-budget QLoRA seeds per arm on the frozen Raw-safe and Curated
artifacts. Stage C still cannot mutate or reselect either artifact.

| Artifact | Result |
| --- | ---: |
| Mixed input | 5,319 records; 7,873,924 tokens |
| Stage-A `Raw-safe` release | 4,902 records; 7,034,169 tokens |
| Stage-B pass | 20,879 chunks |
| Historical Stage-C selected subset | 7,639 chunks; 2,563,348 tokens |
| Raw-safe to selected reduction | 63.56% |

The current Stage-B run used the frozen historical `0.4` fraction in
`configs/temporal_code_curation_protocol_v1.json`. It demonstrates one
budgeted-subset instantiation only. The reduction does not mean that the
remaining 63.56% was useless, low quality, or rejected. Future operational
runs must declare a real deployment budget or emit `retain_all`/`abstain`; they
must not silently inherit a fixed percentage.

## Execution Status

The frozen execution contract is
`configs/code_5m_natural_budget_execution_qwen3_4b_v1.json`. It runs
`Raw-safe` (7,028,736 effective tokens; 429 steps per seed) against `Curated`
(2,555,904 effective tokens; 156 steps per seed). EvalPlus is development-only
comparative evidence because the raw code corpus does not yet have a task-hash
benchmark-contamination audit.

The machine-readable source of this boundary is
`configs/operational_curation_contract_v2.json`, verified by
`validation/test_operational_curation_contract_v2.py`.
