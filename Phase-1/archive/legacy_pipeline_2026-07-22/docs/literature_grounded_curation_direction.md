# Literature-Grounded Curation Direction

## Decision

The project remains a language-model training-data curation framework. The
scientifically defensible target is not a universal intrinsic `Quality` score.
It is an auditable procedure that:

```text
candidate corpus + Deployment Contract
-> quarantine and hard usability gates
-> full curated pool
-> optional budgeted subset under frozen pre-outcome rules
-> fixed-recipe downstream validation
-> scoped release or abstention
```

The closest public precedent is DataComp-LM. It fixes candidate data, model
scale, training recipe, and evaluations, then compares filtering,
deduplication, and mixture policies by training models. This project extends
that experimental logic with explicit Stage-0 hazards, Stage A/B/C ownership,
full-pool retention, deployment-conditioned decisions, and abstention.

## What Public Evidence Supports

| Method or principle | Public evidence | Framework implication |
| --- | --- | --- |
| Exact and near deduplication | Google deduplication experiments and BigCode's The Stack report lower memorization and improve or preserve model performance | Keep Redundancy, but calibrate each implementation and threshold |
| Rule-based structural filtering | C4, Gopher/MassiveText, RefinedWeb, FineWeb, Dolma, and open code corpora use explicit usability and artifact filters | Keep Stage A narrow, auditable, and high-precision |
| Model-based selection evidence | GPT-3 Common Crawl filtering, DCLM, FineWeb-Edu, Llama 3, Qwen, and MiniCPM use learned or reference-based filters | Keep Selection Value Evidence as a policy input, not ground-truth quality |
| Controlled training validation | DCLM and FineWeb ablations train fixed models under comparable recipes | Stage C must be the scientific validator |
| Data mixture optimization | DoReMi and DSIR show that source/domain mixture and target alignment affect downstream performance | Split Coverage into collapse prevention and objective-conditioned alignment |
| Heterogeneous-source retention | A Pretrainer's Guide finds broad source mixtures useful and no universal filter optimum | Avoid one global quality threshold or forced reduction ratio |
| Repository-level code construction | The Stack, DeepSeek-Coder, Code Llama, and Qwen2.5-Coder use repository, license, language, code/documentation, and mixture structure | Code curation must preserve repository provenance and content type |
| Retention guardrails | Qwen2.5-Coder and continued-pretraining work report code gains together with general or math retention | Target gains alone are insufficient for release |
| Temporal evaluation | A Pretrainer's Guide and contamination work show data age and train-test overlap matter | Preserve time metadata and use post-cutoff, repository-disjoint evaluation |
| Hazard and provenance controls | The Stack emphasizes licensing and governance; contamination and poisoning work shows small unsafe slices can invalidate evaluation or training | Stage 0 is a required boundary, not an optional preprocessing script |

## What Public Evidence Does Not Validate

The following current project choices are hypotheses until independently
calibrated:

- SimHash Hamming thresholds `3` and `10`
- Jaccard threshold `0.75`
- containment threshold `0.88`
- structural duplicate risk `0.85`
- Stage-B weights `0.8 / 0.2`
- marker-based useful-recurrence relief
- style-specific redundancy discounts and caps
- any fixed selection ratio
- raw-distribution retention as a sufficient definition of Coverage
- small-probe NLL as a universal Utility instrument

These values may remain in frozen development arms, but a citation cannot turn
them into validated measurements. Each must be either calibrated, ablated,
replaced by a reproduced method, or labeled diagnostic-only.

## Revised Core Interpretation

### Validity

Validity is a high-precision structural usability construct. It covers parsing,
encoding, extraction residue, minimum usable units, and pathological
repetition. It must not estimate semantic worth.

### Selection Value Evidence

Selection Value Evidence is a vector of observable pre-outcome features rather
than a universal scalar truth:

- information and token density
- structural substance
- boilerplate and generated-content risk
- source and content-type metadata
- target-distribution relevance when a Deployment Contract supplies a target

An aggregate score is a policy hypothesis. The component evidence must remain
available for audit and ablation.

### Redundancy

Redundancy has two evidence tiers:

1. Stage A removes byte-identical and canonical exact duplicates.
2. Stage B treats fuzzy near-duplicate and recurrence signals as soft evidence
   until an independent holdout validates irreversible use, and controls
   saturation without declaring related
   examples intrinsically bad.

Useful recurrence, common APIs, test templates, and repeated pedagogical
structure must not be removed solely because they share structure. Duplicate
representatives must be selected deterministically from eligible records, not
by incidental input order.

### Coverage

Coverage must be separated into:

- `retention_support`: prevents collapse of source, language, content type,
  repository, path, and validated semantic clusters;
- `target_alignment`: measures compatibility with the predeclared Deployment
  Contract or target sample.

Preserving the raw distribution is not automatically desirable because the raw
distribution may be dominated by spam, templates, or collection bias.

### Utility

Utility is the downstream effect of a complete candidate release under a
specified target model, token budget, compute budget, and evaluation
distribution. It remains Stage C only and cannot be consumed by Stage B.

## Required Validation Pattern

The primary scientific comparison is:

```text
same candidate pool
+ same base checkpoint
+ equal training tokens and compute
+ frozen training recipe
+ repository/time-disjoint evaluation

Framework-selected vs Stage-A random
```

Supporting arms:

- base checkpoint with no update
- raw-random equal-budget
- Stage-A-all and raw-all when compute permits
- Selection-Value-only
- Redundancy-only or Redundancy-light
- no-Coverage-support
- target-alignment ablation
- known-high-quality reference when available

Stage-A random estimates the total contribution of Stage B under feasible
usable data. Feature-matched baselines diagnose mechanisms but must not replace
the primary total-effect baseline.

## Execution Order

Current implementation checkpoint:

- Stage-A duplicate representatives are now selected only from local-gate-pass
  chunks using a deterministic `chunk_uid` order.
- Permuting fixture input order produces identical decisions.
- Existing broad and path-stratified frozen corpora have zero decision changes.
- Code-domain v2 keeps the same `3313` Stage-A-pass count and swaps only one
  exact-duplicate representative within a three-member group.
- The first 10-pair Redundancy benchmark reports current threshold precision
  `1.0`, recall `0.5`, and F1 `0.666667`.
- The fixture-optimal combination is SimHash `18`, Jaccard `0.50`,
  containment `0.95`, but it is not eligible for promotion from this bounded
  development fixture.
- Stage-B structural match count increases from `1` to `4` between duplicate
  group sizes `2` and `5`, while mean soft risk remains fixed at `0.85`.
- A 25-repository, 111-pair development silver benchmark reports current
  precision `1.0`, recall `0.626667`, and near-only recall `0.44`.
- No relaxed threshold candidate passed the independent 13-repository holdout.
  The conservative Hamming `5` challenger had precision `0.964286`, near-only
  recall `0.538462`, and useful-data dropout `0.058824`.
- Cluster audit found that challenger would additionally remove 13 current
  records, including four Stage-B-selected records and 1,241 selected token
  proxies. The canonical Stage-A threshold therefore remains unchanged.
- Count-sensitive Stage-B saturation arms were run outcome-free. `log_count`
  preserves 47 repositories, all 610 selected tests, and selection Jaccard
  `0.986072` versus the binary current policy. It is frozen as the sole
  proxy-training candidate, not promoted as canonical.

### Phase 1: Freeze Claims and Evidence Classes

1. Adopt this document as the research direction.
2. Classify every metric and parameter as:
   `reproduced_method`, `paper_aligned_principle`,
   `project_hypothesis_frozen`, or `engineering_diagnostic`.
3. Remove claims that Core behavior tests prove construct validity.
4. Keep `Quality` as a legacy alias only.

Exit criterion: every binding decision can be traced to its evidence class and
claim boundary.

### Phase 2: Repair Irreversible Boundaries

1. Complete Stage-0 labeled validation for license, secrets, PII, benchmark
   contamination, generated content, and poisoning indicators.
2. Fix Stage-A duplicate representative selection so failed or quarantined
   records cannot suppress usable representatives.
3. Build labeled and adversarial Validity and hard-Redundancy fixtures.
4. Report false-positive and false-negative slices by content type.

Exit criterion: Stage 0 and Stage A have measured operational error rates and
high-confidence irreversible actions.

### Phase 3: Calibrate Redundancy

1. Build exact, near-duplicate, related-but-useful, templated, and independent
   code/text pair benchmarks.
2. Sweep SimHash, Jaccard, containment, shingle, and AST thresholds.
3. Replace max-pair-only soft risk with explicit cluster frequency and
   saturation evidence.
4. Keep useful recurrence as a separate reported signal until validated.
5. Align generic-text and temporal-code semantics or document domain-specific
   implementations explicitly.

Exit criterion: thresholds have precision/recall and dropout evidence, and
soft Redundancy responds to saturation magnitude.

### Phase 4: Rebuild Stage B as Auditable Policy Arms

1. Preserve Selection Value Evidence as components, not only one scalar.
2. Separate Coverage retention from target alignment.
3. Define a small preregistered policy family instead of tuning one formula:
   full selector, Selection-Value-only, Redundancy-only, no-Coverage, and
   target-aligned.
4. Emit `retain_all` whenever no binding budget exists.
5. Freeze policy arms before target-model outcomes.

Exit criterion: each Stage-B contribution can be isolated by an outcome-free
ablation.

### Phase 5: Run Cheap Proxy-Scale Screening

1. Use identical 0.4B-1.5B proxy models and fixed recipes to compare
   `binary_current`, the frozen `log_count` saturation candidate, and
   Stage-A-random.
2. Evaluate heldout NLL, target tasks, retention, contamination, and seed
   variance.
3. Do not promote an arm from one dataset or one probe setting.

Exit criterion: a development policy is selected by replicated, multi-slice
evidence without touching confirmatory data.

### Phase 6: Run the Decisive 4B Continued-Pretraining Experiment

1. Freeze a pre-update checkpoint and genuinely later raw-like corpus.
2. Train Framework-selected and common disjoint Stage-A-random arms under
   equal tokens and compute, with at least three seeds.
3. Include raw-random and ablations as supporting arms.
4. Evaluate temporal target tasks, EvalPlus or equivalent code tasks, general
   retention, forgetting, contamination, and training efficiency.
5. Use the same Stage-A baseline pool for all sensitivity arms and keep it
   disjoint from their union.

Exit criterion: either a scoped benefit passes the preregistered margin and
guardrails, or the framework issues a defensible abstention/negative result.

### Phase 7: Untouched Confirmation and Paper Freeze

1. Run the frozen policy on untouched repositories, time windows, and
   confirmatory tasks.
2. Rebuild all decision and guardrail reports without changing thresholds.
3. Freeze the paper claim at the strongest supported level.

Exit criterion: reproducible artifacts support the exact claim, including
negative and abstention outcomes.

## Paper Position

The strongest intended contribution is:

> An auditable, abstention-capable, objective-conditioned framework for
> constructing LM training releases from candidate corpora, with irreversible
> usability and hazard gates, optional budget allocation from frozen
> pre-outcome evidence, and controlled downstream validation.

It should not be positioned as a solved universal measurement of data quality.

## Primary References

- DataComp-LM: https://arxiv.org/abs/2406.11794
- FineWeb and FineWeb-Edu: https://arxiv.org/abs/2406.17557
- Deduplicating Training Data Makes Language Models Better:
  https://arxiv.org/abs/2107.06499
- A Pretrainer's Guide to Training Data:
  https://arxiv.org/abs/2305.13169
- DoReMi: https://arxiv.org/abs/2305.10429
- DSIR: https://arxiv.org/abs/2302.03169
- The Stack: https://arxiv.org/abs/2211.15533
- GPT-3 data construction: https://arxiv.org/abs/2005.14165
- Llama 3: https://arxiv.org/abs/2407.21783
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- Dolma: https://arxiv.org/abs/2402.00159
- RefinedWeb: https://arxiv.org/abs/2306.01116
- Data Selection via Importance Resampling:
  https://arxiv.org/abs/2302.03169
- Scaling Data-Constrained Language Models:
  https://arxiv.org/abs/2305.16264
