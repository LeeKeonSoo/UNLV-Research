# Literature-Grounded Core-Policy Redesign

## Decision

The framework must not claim to measure one intrinsic property called
"Quality." Published LM pipelines combine different operations: provenance and
risk controls, deduplication, document filtering, model-based selection, and
source-aware mixing. Those operations answer different questions and have
different failure modes.

The framework should be domain-general at the **contract and policy interface**
level, not by promising that one threshold creates the best dataset for every
domain. A run accepts declared input metadata, executes a versioned policy
profile, emits a reason-coded pool, and reports composition changes. A policy
claiming a downstream gain must also be calibrated and externally validated for
its stated scope.

The current v3 implementation remains `safe_structural_v3`: a high-precision
baseline. Its approximately 2.1% removal on the current Code corpus means that
the corpus has little evidence for v3's narrow rules. It does not mean all
remaining content is necessary or high-value.

## What The Literature Supports

| Observation | Evidence | Design consequence |
| --- | --- | --- |
| Exact and near duplication increase memorization, privacy exposure, and wasted training. | [Lee et al.](https://arxiv.org/abs/2107.06499), [Kandpal et al.](https://arxiv.org/abs/2202.06539) | Redundancy is a first-class Core with record, span, and cross-source scopes. |
| Successful web corpora use multiple filters and ablate their choices. | [FineWeb](https://arxiv.org/abs/2406.17557), [Dolma](https://arxiv.org/abs/2402.00159), [RefinedWeb](https://arxiv.org/abs/2306.01116) | Policies must be individually versioned and false-positive audited. |
| Model-based filtering can be strong, but it is a learned proxy, not intrinsic truth. | [GPT-3 filtering](https://arxiv.org/abs/2005.14165), [DataComp-LM](https://arxiv.org/abs/2406.11794) | Learned selection is optional, calibrated, and explicit. |
| Data mix and source diversity matter; filtering and mixing are distinct. | [Dolma](https://arxiv.org/abs/2402.00159), [DeepSeek LLM](https://arxiv.org/abs/2401.02954), [ROOTS](https://arxiv.org/abs/2303.03915) | Coverage describes/protects strata; mixture weights are separate from deletion. |
| Benchmark overlap can invalidate evaluation. | [GPT-3 procedure](https://arxiv.org/abs/2005.14165), [PALOMA](https://arxiv.org/abs/2312.10523) | Decontamination is an explicit evaluation firewall, never hidden selector feedback. |
| Curation systems require inspectable operators and repeatable recipes. | [Data-Juicer](https://arxiv.org/abs/2309.02033), [survey](https://arxiv.org/abs/2402.16827) | The product unit is a policy profile, not a single score. |

Dolma warns that labels such as "quality" hide value judgments. This is a
reason to make policy hypotheses visible, not to avoid all selection. FineWeb
and DataComp-LM show stronger filtering can improve results, but they do not
license an uncalibrated weighted formula in another corpus.

## Revised Four Cores

| Core | Operational question | Allowed metrics | Allowed decisions |
| --- | --- | --- | --- |
| **Validity** | Can this record safely and legally enter the candidate pool in a declared form? | provenance, rights, PII/secret/risk detections, normalization and metadata completeness | quarantine, repair, release; never a usefulness rank |
| **Redundancy** | Is content materially repeated at record, span, template, or benchmark-overlap scope? | exact digest, n-gram overlap, template signature, declared benchmark overlap | retain representative, remove confirmed copies, or quarantine evaluation-contaminated data |
| **Quality** | Is there explicit evidence that a unit or separable span has no independent learning payload? | explicit artifact marker, structural test, and payload-preservation check | reason-coded artifact removal or span compaction; no intrinsic score or model-relative runtime selector |
| **Coverage** | Does policy create a preventable hole or shift in content, language, source, time, or style strata? | pre/post token share, stratum distributions, duplicate-family retention | preserve representative, emit drift warning, approve separately declared mixture weights |

Quality is not an intrinsic document score. A policy must name what it
observes: for example, an explicit non-editable generated artifact, a
template-family copy. It may not remove data merely by calling it "low quality."

## Three Stages

```text
Declared raw input + source manifest
  -> Stage A: admission, normalization, safety/rights quarantine
  -> Stage B: local evidence extraction and high-precision hard gates
  -> Stage C: corpus-level deduplication, artifact policies, coverage guard,
              and optional calibrated selection profile
  -> reason-coded curated pool + audit bundle

Frozen curated pool -> External Evaluation Firewall and Training Protocol
```

### Stage A: Admission And Safety

Unit: source record. Normalize deterministically; validate source, collection
time, rights, and text contract; detect secrets/PII and declared hazards; and
preserve source-backed language, content type, artifact context, timestamp, and
license metadata. Stage A must not use benchmark score, downstream loss, or
observed Utility.

An explicit benchmark exclusion list is allowed only in a versioned external
evaluation profile. It is not a default statement about general data validity.

### Stage B: Local Evidence And Hard Gates

Unit: normalized record or chunk. Compute cheap, explainable evidence and apply
only high-precision failures: empty/corrupted text, malformed contract output,
declared minimum-length violations, exact duplicates, and confirmed artifact
types. Language parsing may be diagnostic for any declared language but may
remove only after version-specific false-positive validation.

Stage B can tag uncertainty but should not perform global priority ranking.

### Stage C: Corpus-Level Policy And Materialization

Unit: corpus family, duplicate cluster, or declared stratum. Apply cross-source
deduplication, duplicate-family representative retention, confirmed artifact
removal, coverage preservation, and optional calibrated selector execution.
Emit every selected, rejected, and quarantined record with a reason code.

There are two explicit modes:

1. `safe_structural`: high-precision reason-coded redundancy/artifact policies.
   This is the current v3 mode.
2. `calibrated_selector`: a versioned learned or heuristic selector with declared
   reference data, score direction, threshold or sampling law, held-out
   calibration, false-positive analysis, and external validation plan. It cannot
   read benchmark outcomes from the run it selects for.

Neither mode may hide a token cap or fixed retention fraction as curation. A
separate `mixture_recipe` may specify source/stratum sampling weights for a
training deployment, but it preserves the full curated pool and calls omitted
records unallocated, not rejected.

## Policy Card Contract

Every active or experimental policy needs a registry entry and a policy card:

1. Core, stage, decision unit, and scope.
2. Hypothesis and deployment scope.
3. Exact allowed and forbidden inputs.
4. Metric/model version, threshold or sampling law, and reason codes.
5. Positive, negative, adversarial, and cross-domain fixtures.
6. Labeled false-positive/false-negative audit and known blind spots.
7. Coverage/composition impact and policy interactions.
8. Frozen config hash, rollback behavior, and external validation protocol.

A policy without these fields can remain diagnostic but cannot delete data.

## Policy Families And Gates

| Family | Default status | Gate before removal authority |
| --- | --- | --- |
| Provenance, rights, secrets, PII, corruption | Safe profile | Source-specific held-out risk audit |
| Exact and very-high-confidence near duplicates | Safe profile | Duplicate precision audit and stable representative policy |
| Explicit generated-and-do-not-edit, license-only, structural scaffold | Safe profile | Negative fixtures for real code/documentation |
| Source-declared generated/dependency artifacts | Conditional | Verify label semantics and audit useful examples |
| Repeated boilerplate/template families | Conditional | Family-level labels and cross-source precision audit |
| Minified, lock, binary, vendored paths | Conditional | Source-declared type or content/parser confirmation; paths alone are insufficient |
| Language ID, parser validity, web heuristics | Conditional | Per-stratum calibration and abstain/retain behavior |
| Reference-distribution classifier or LM score | Experimental selector | Frozen reference set, held-out calibration, structural-profile ablation, no benchmark feedback |
| Target domain proportions/token cap | Allocation only | Separate mixture recipe and training ablation; never reason-coded removal |

## Evaluation Design

Evaluate three layers independently.

1. **Policy behavior:** labeled/adversarial fixtures for every removal reason;
   report TP, FP, FN, TN and reason-coded token changes.
2. **Corpus behavior:** clean, raw-like heterogeneous, duplicate-heavy, and
   risk/artifact-heavy scenarios. Report pre/post source/language/content/time
   distributions, duplicate clusters, and abstentions. A clean corpus should
   retain nearly all data.
3. **External LM outcome:** freeze Base, Stage-A release, `safe_structural`, and
   any `calibrated_selector` output before training. The primary comparison is
   natural-token training with matched hyperparameters and at least three seeds.
   Equal-token runs may diagnose selection signal but do not replace natural
   budget evidence.

For Code, use a contamination-controlled multi-benchmark suite. General text
and Math require separate held-out suites. A benchmark cannot tune a selector
and also confirm that selector.

## Migration From Current v3

1. Freeze `safe_structural_v3`; report its 2.1% reduction as a conservative
   baseline, not a universal effectiveness result.
2. Build policy cards and labeled audits for repeated boilerplate,
   source-declared artifact types, and parser/language checks. Do not activate a
   policy because it removes many tokens.
3. Collect a heterogeneous raw corpus with source-backed metadata and a
   separate controlled stress suite. Do not use an artificially dirty corpus as
   the only downstream experiment.
4. Implement `calibrated_selector` only after its reference-data and calibration
   protocol are frozen. It may abstain for unsupported domains.
5. Compare `safe_structural` and `calibrated_selector` against Stage-A release
   under natural-token training. Promote only reproducible policies.

The first implementation step is now present as a non-selecting preflight:
`scripts/preflight_calibrated_selector.py` checks a
`calibrated-selector-contract-v1` for a frozen, hash-verified reference pool,
held-out calibration set, scope audit, external validation plan, and passed
false-positive audit. The example contract is deliberately non-runnable. No
model score is computed and no Stage C output changes until a separate frozen
candidate contract passes this gate.

## Current Code Candidate Reconnaissance

The v3 Stage-A Code release was inspected with the existing diagnostic
alpha-normalized Python template-family inventory. Of 4,697 eligible Python
records, it found 12 duplicate template families containing 75 records. The
largest families are repeated `transformers` model `__init__.py` export
structures. This is evidence for a candidate audit, not removal authority:
valid API/package structures can share the same normalized form. The result is
stored at
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\proxy_removal_forensics\python_template_family_inventory_v3.json`.

The family-level false-positive audit is complete: all 12 observed families
(75 records) are labeled `retain`, with no unlabeled family or sample-path
mismatch. The families cover public API export structures, versioned schema
migrations, distinct model configurations, and bug-regression tests. Therefore
alpha-normalized template similarity alone is explicitly rejected as a new
Stage C removal rule for this corpus. The audit report is stored at
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\proxy_removal_forensics\python_template_family_false_positive_audit_v3.json`.

The next candidate gate must use additional source-backed artifact metadata or
a separately calibrated selector; it cannot reinterpret this template inventory
as an intrinsic-quality ranking.

## Frozen Reference-Distribution Diagnostic

The declared `github_reference_pool` was frozen without entering the current
Stage C policy: 662 source-declared reference records and 4,225 raw-like
records remain disjoint. A deterministic repository-disjoint split keeps 392
reference records for fitting and holds out 270 reference records from
`pytest-dev/pytest` and `scikit-learn/scikit-learn`. The held-out set also
contains 270 balanced raw-like records.

On that frozen split, a character n-gram logistic-regression *diagnostic*
achieved ROC-AUC 0.883018 and average precision 0.842930. The report is at
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\calibrated_selector_v1\reference_distribution_probe_report.json`.
This supports only the narrow observation that the declared reference sources
and the raw-like sources have separable text distributions in the declared
Code scope. The probe reads neither Utility nor benchmark outcomes nor a token
target, scores no candidates for selection, and removes no data. It therefore
does not establish intrinsic Quality, training Utility, or a Stage C removal
policy.

Before any calibrated selector can be activated, the next gates are: audit a
review sample of high-scoring raw-like records for false positives, freeze a
scope/shift audit, define the score direction and action contract, and reserve
an external downstream evaluation set that is never used for selector tuning.

The first review-only sample is frozen at
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\calibrated_selector_v1\reference_distribution_review_sample_top50.json`.
It contains the 50 highest-scoring records from 3,833 raw-like records outside
the diagnostic training sample, together with their preserved dataset,
repository, path, and bounded text excerpt. Every record is explicitly
`unlabeled`; the artifact emits no selection decision and removes no record.
Several leading examples are ordinary public Python source files from distinct
repositories (for example NeMo, TensorFlow, and Composer). That observation is
precisely why a source-role score cannot be converted directly into a removal
rule: high similarity to the reference distribution can identify useful-looking
in-scope code as well as undesirable artifacts.

The Stage-A overlap audit of this sample found that 46 of 50 records satisfy
all existing release, rights, quarantine, hazard, and declared Code-domain
evidence. The remaining four also have Python language confidence 1.0 and no
admission or hazard issue, but their audit-only lexical composition is
`mathematics`. They are retained as scope-audit cases rather than reclassified
or removed: code can legitimately contain mathematical content, and the
composition audit is descriptive rather than a selection authority. The report
is `D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\calibrated_selector_v1\reference_distribution_overlap_audit_top50.json`.

**Policy decision:** `stage_c_reference_distribution_diagnostic` is registered
as diagnostic-only with no reason codes, no selection decision, and no removal
authority. The reference-distribution score is explicitly excluded from the
`calibrated_selector_template_v1` Stage C surface until a new policy card
defines an independent target, an action direction, a labeled false-positive
audit, a scope audit, and downstream validation that does not feed benchmarks
back into the selector.

## Additional Rule Opportunity Result

The Stage-A release stream (4,887 chunks; whitespace token proxy 2,575,171)
was audited for provenance-backed additional Stage C rules. It contains zero
source-declared generated artifacts and zero source-declared dependency copies.
It does contain 19 path-pattern candidates and two text-shape candidates, but
both families are blocked due to known false-positive risk. This is an input
metadata limitation, not justification to infer generated or vendored status
from paths, text shape, source tier, or a learned score. The report is
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\proxy_removal_forensics\core_rule_opportunity_audit_stage_a_v3.json`.

## Source-Declared Artifact Stress Replay

To test the collection contract where upstream artifact metadata is genuinely
available, the framework collected a controlled `gfx-rs/wgpu` corpus using the
repository's `.gitattributes` declaration through `git check-attr`. Of 2,414
UTF-8 tracked files, 130 were explicitly declared `linguist-generated=true`;
the remaining 2,284 were retained as `unknown`, not inferred as authored. The
active A-B-C replay produced 2,342 Stage-A release records and removed zero
records through the explicit-generated-artifact policy. The canonical
opportunity audit correctly finds all 130 declared-generated candidates, but
marks the prospective rule `needs_labeled_false_positive_audit`.

More specifically, zero of those 130 records satisfies the active
generated-and-do-not-edit condition. This controlled counterexample supports
the current policy boundary: source-declared generated status is valuable
provenance for audit, but is not itself a valid removal authority. The replay
uses `protocols/git_attribute_wgpu_stress_curation_contract.json` and writes
artifacts under `D:\UNLV-Research\source_metadata_stress`.

## Claim After Redesign

The defensible claim is: *an auditable framework for constructing LM training
data recipes from heterogeneous raw corpora, with provenance-aware admission,
reason-coded structural curation, coverage audit, and optional calibrated
selection profiles.*

It is not: *a universal intrinsic data-quality detector* or a guarantee that
every curated output improves every language model and domain.
