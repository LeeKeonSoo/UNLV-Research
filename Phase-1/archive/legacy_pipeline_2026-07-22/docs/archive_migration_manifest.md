# Archive Migration Manifest

## Purpose

This manifest classifies historical numbered scripts before any physical move.
A script is not archive-safe merely because its filename has no textual
references. Its input and output artifacts must also be unused by active
validation, tests, configurations, and retained compatibility paths.

## Status Values

| Status | Meaning |
| --- | --- |
| `retain_active` | Required by an active operator path or canonical framework validation. |
| `compatibility_required` | Historical execution path is not active, but its artifact lineage is still consumed. |
| `move_safe` | Script and its artifact lineage are unused outside the archive set. |

## Audited Forward-Development Lineage

| Scripts | Status | Artifact lineage reason |
| --- | --- | --- |
| `100`, `101`, `102`, `103`, `104`, `105`, `106`, `107`, `108` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve fresh-development, executable-harness, SWE-bench, EvalPlus, and retention artifact commands. |
| `109`, `110`, `111`, `112`, `113` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve forward-E2 acquisition and productivity evidence operations. |
| `114`, `115`, `116`, `117`, `118`, `119`, `120` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve forward snapshot, candidate, accumulation, and capacity artifact lineage. |

## Audited Temporal-Code Lineage

| Scripts | Status | Artifact lineage reason |
| --- | --- | --- |
| `76`, `77` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root compatibility wrappers preserve historical automation and direct imports. |
| `78`, `79`, `80`, `81`, `82`, `83`, `84` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root compatibility wrappers preserve the documented blind-review commands and historical validation artifacts. |
| `85`, `86`, `87`, `88`, `89` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve broad-tranche manifest, readiness, and ablation commands. |
| `90`, `91`, `92`, `93`, `94`, `95` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve path-stratified, confirmatory, and Stage-C smoke artifact lineage. |
| `96`, `97`, `98`, `99` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve development-expansion/native-execution commands and validator inputs. |

## Audited Domain-Experiment Lineage

| Scripts | Status | Artifact lineage reason |
| --- | --- | --- |
| `136`, `137`, `138`, `139`, `140` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve initial Code-domain reference-pool, equal-token, QLoRA-smoke, and development-plan commands. |
| `141`, `142`, `143`, `144`, `145`, `146`, `147`, `148`, `149`, `150`, `151`, `152` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve Code-domain development/confirmatory training, EvalPlus, and general-task guardrail commands and tested helper contracts. |
| `153`, `154`, `155`, `156`, `157`, `158`, `159`, `160` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve Code-domain v2 candidate-pool, Stage-A/Stage-B, development, and confirmatory protocol commands. |
| `200` | `compatibility_required` | Builds the Code Block-3 EvalPlus report from the confirmatory guardrail artifact; it preserves a reproducible historical benchmark interpretation. |
| `201`, `202`, `203` | `compatibility_required` | Acquire, materialize, and freeze the Math pool and Stage-C protocol. Their JSONL arms, heldout data, and reports are the source lineage for Math evidence. |
| `204`, `205` | `compatibility_required` | Materialize and summarize Code/Math natural-budget arms; these reports establish the natural-budget comparison record. |
| `206`, `208` | `compatibility_required` | Materialize the Math selector-v2 output and freeze its natural-budget protocol; they are required to reproduce the unresolved Math branch. |
| `207`, `209` | `compatibility_required` | Summarize Code natural-budget Stage-C evidence and cross-domain Block-1--3 evidence; their claims must remain traceable while Code is historical-positive/rerun-required. |
| `210_build_math_failure_fixture_contract.py` | `compatibility_required` | Defines the Math failure fixture contract and writes its validation report. |
| `210_build_production_readiness_gate_report.py` | `retain_active` | Is invoked directly by retained validation tests and produces the release/readiness gate. |

## Audited Collection-Operations Lineage

| Scripts | Status | Artifact lineage reason |
| --- | --- | --- |
| `121`, `122`, `123`, `124`, `125`, `126` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve forward snapshot collection, candidate ledger, and recipe-batch operations. |
| `127`, `128`, `129`, `130`, `131`, `132`, `133`, `134`, `135` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve retrospective collection, operations, and capacity-audit artifact lineage. |

## Active Core And Historical Redundancy

| Scripts | Status | Artifact lineage reason |
| --- | --- | --- |
| `161`--`174` | `retain_active` | Current Core, Stage-0, Coverage, selection-boundary, and redundancy construct-evidence audits remain part of the active framework claim surface. |
| `175`--`189` | `compatibility_required` | Implementations moved to `archive/temporal_code`; root wrappers preserve historical redundancy holdout, ablation, QLoRA, and guardrail experiment reproduction. |

## First Move Rule

The first physical archive batch may contain only scripts marked `move_safe`
after both filename-reference and artifact-lineage checks. Moving a
`compatibility_required` script requires a root compatibility wrapper or an
updated consumer path in the same batch.
