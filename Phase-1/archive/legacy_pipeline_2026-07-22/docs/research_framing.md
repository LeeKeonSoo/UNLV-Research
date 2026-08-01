# Research Framing: LM-Training Data Curation

The canonical implementation requirements, responsibility boundaries, test
matrix, and research milestones are defined in
`docs/framework_requirements_and_test_matrix.md`.

## Purpose

This project is a training-data curation framework for language-model training.

The framework assumes that data collection happens upstream. Given an
arbitrary candidate corpus and a declared Deployment Contract, the goal is to
produce a supported language-model training release by filtering unusable
chunks, preserving the full curated pool, optionally selecting a non-Utility
evidence-supported training subset under a binding budget, validating candidate release arms, and
abstaining when no release satisfies the contract.

The core claim is not limited to a specific source, dataset, or time window:

```text
candidate corpus + Deployment Contract
-> full curated pool
-> optional budgeted training subset
-> supported training release or abstention
```

The same framework should apply to:

- pretraining from scratch, where the candidate corpus is the initial raw training pool
- continued pretraining, where the candidate corpus is data collected after a previous training run
- domain adaptation, where the candidate corpus is a new domain-specific pool
- periodic dataset refresh, where new crawls or new sources are curated before training

The time-window setting is an application scenario, not the main claim. The main claim is general-purpose curation of candidate data into data suitable for LM training.

## Non-Goal

This project is not trying to optimize a selector directly against Utility.

Utility is not a Stage-B selector objective. It is a Stage-C outcome validator. A Utility failure can indicate selector weakness, but it can also indicate probe instability, baseline confounding, token-exposure artifacts, insufficient training budget, or mismatch between the probe model and the target training scenario.

Human or LLM judgments are optional diagnostics, not canonical selector
ground truth. The primary validation path avoids requiring subjective review:
freeze the Stage-B policy before target-model outcomes, verify automated
properties and ablations, and test equal-budget downstream benefit on
development and untouched confirmatory evidence.

The framework should therefore avoid this interpretation:

```text
Utility failed -> tune selector to maximize Utility
```

The intended interpretation is:

```text
Utility failed -> diagnose whether the selected subset, the baseline, or the Utility protocol is responsible
```

## Stage Structure

The framework keeps the Core-Metric-Policy design and the three-stage execution model.

### Stage A: Chunk-Level Hard Gate

Stage A answers:

```text
Can this chunk be used at all?
```

It removes chunks that should not enter the candidate training pool:

- structurally invalid text
- extraction artifacts
- severe symbol or markup noise
- exact duplicates
- canonical exact duplicates; fuzzy near-duplicates remain Stage-B evidence
- broken or pathological repetition

Stage A should not judge downstream usefulness or training Utility.

### Stage B: Chunk-Level Selection

Stage B answers:

```text
If the declared budget is binding, which usable chunks should receive it?
```

Every Stage-A-surviving chunk remains in the full curated pool. Stage B is
competitive only under a binding token or compute budget. It then allocates
that budget using Core evidence such as:

- selection value evidence (`Quality` is a legacy alias)
- redundancy risk
- useful recurrence
- length support
- lexical and source diversity
- coverage support

Stage B may use proxies that are plausibly related to learnability, but it must not use Stage-C Utility outcomes as a selector objective.

When the budget can hold the full curated pool, Stage B emits `retain_all`.
When it cannot, records outside the training subset are
`budget_not_selected`; they are not rejected or labeled low quality.

### Stage C: Subset-Level Validation

Stage C answers:

```text
Is the selected subset acceptable as a training dataset?
```

It validates the selected subset against baseline subsets from the same candidate corpus. Stage C includes:

- coverage retention
- selected versus Stage-A random curation benefit
- selected versus matched Stage-A baseline strict counterfactual evidence
- token-inventory and token-exposure stress diagnostics
- anti-memorization or easy-NLL controls where needed
- probe stability or replicated Utility evidence before certification claims

Stage C is where Utility belongs.

## What "LM-Training-Ready" Means

A full curated pool or budgeted subset is LM-training-ready only if the
evidence supports more than surface selection-value proxies.

The dataset should be:

- structurally usable
- sufficiently supported by pre-outcome selection-value proxies
- not dominated by duplicates or harmful repetition
- representative enough for the intended training scope
- not merely easy for a small probe to memorize
- useful under a defensible training-Utility validation protocol
- accompanied by caveats when the evidence is incomplete

High scores under pre-outcome Selection Value Evidence are not enough. A corpus
can be readable and coherent while still being too narrow, repetitive,
templated, easy, hard, or unstable under training validation.

## Utility Interpretation

Utility should be interpreted as evidence about training usefulness, not as a standalone truth signal.

The current Utility protocol separates several claims:

| Evidence | Meaning |
| --- | --- |
| selected > Stage-A random | curation benefit over feasible random usable data |
| selected > multi-matched Stage-A baseline | strict counterfactual benefit |
| selected > anti-memorization matched baseline | evidence against easy-NLL/repetition confounding |
| clean token-shuffle stress | less token-exposure confounding |
| replicated valid probe family | more stable Stage-C protocol evidence |

These should not be collapsed into one naive pass/fail label. Different combinations imply different curation decisions.

## Curation Decision Layer

The final output should be a curation decision, not merely a Utility score.

Recommended decision categories:

| Decision | Meaning |
| --- | --- |
| `accepted_for_training` | Stage A/B/C evidence supports using the selected subset for training |
| `accepted_for_training_with_caveat` | usable for development or targeted training, but caveats must be reported |
| `needs_certification_utility` | Stage A/B evidence is good, but Stage-C Utility is not certification-grade yet |
| `utility_probe_unstable` | Utility probe controls are not stable enough to support a training claim |
| `strict_baseline_confounded` | canonical strict baseline appears biased by length, repetition, or easy-NLL signal |
| `token_exposure_caveat` | selected subset may be helped or hurt by token inventory/exposure artifacts |
| `rejected_for_training` | evidence indicates the selected subset is not suitable for the intended training use |

This decision layer keeps the project aligned with the practical goal:

```text
Given a candidate corpus, decide what should actually be trained on.
```

The curation decision is followed by a Deployment-Contract-conditioned release
decision. The same Stage-C evidence may support different releases for
different predeclared objectives without changing Stage B:

```text
broad refresh -> selected_only, stageA_broad, or reject
targeted update -> selected_only, coverage_backfilled, or reject
capability-preserving update -> release only when target and retention
guardrails both pass
```

Read `docs/deployment_contract_and_release_policy.md` before changing release
actions or interpreting target-SLM outcomes.

## Relationship to Current Literature

The detailed evidence mapping and revised execution order are maintained in
`docs/literature_grounded_curation_direction.md`.

This framing is aligned with current LM data work:

- DataComp-LM fixes candidate data, model/training recipes, and evaluations,
  then tests filtering, deduplication, and mixing by downstream model
  performance. This is the closest public precedent for Stage C.
- FineWeb and FineWeb-Edu show that transparent filtering and curation choices
  should be evaluated through controlled model-training ablations.
- Google deduplication work and BigCode's The Stack provide strong evidence for
  exact and near deduplication, but not for this project's particular SimHash,
  Jaccard, containment, or AST thresholds.
- DoReMi and DSIR show that mixture design and target alignment are distinct
  from preserving the raw candidate distribution.
- A Pretrainer's Guide shows that filtering has trade-offs and that no single
  quality filter is optimal across objectives.
- SmolLM shows that small language models can be competitive when trained on carefully curated high-quality corpora.
- Time-continual pretraining work such as TiC-LM shows that updating models with new data is a real use case, but it is only one application of the broader curation problem.

The project should therefore be presented as a general LM-training-data
curation framework, with continual or periodic updates as an important but
optional setting. Public literature supports the procedure and major method
families; it does not validate project-specific weights, thresholds, or proxy
formulas.

## Current Research Implication

The current Utility debugging results should not be interpreted as evidence that the research goal is invalid.

Instead, they show that Stage-C Utility must be treated as a careful validation protocol:

- `tiny_textbooks` is development-ready but has a token-exposure caveat.
- `wikitext103_subset` suggests canonical strict-baseline confounding and needs reported control revision.
- `openwebtext2_subset` has a dataset-level replicated Utility protocol candidate, but no global default Utility family.

This means the next research step is not to force Utility to pass. The next step is to formalize the curation decision layer and decide which evidence is required for each training-use claim.

## Next Improvement Direction

Future code and report changes should move toward:

1. Adding an explicit curation decision report.
2. Separating development readiness from certification readiness.
3. Reporting Utility caveats as first-class decision evidence.
4. Keeping dataset-specific Stage-C diagnostics from becoming dataset-specific selector objectives.
5. Defining what evidence is sufficient to release or use a curated training subset.
