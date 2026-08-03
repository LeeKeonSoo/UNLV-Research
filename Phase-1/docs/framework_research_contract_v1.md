# Framework Research Contract v1

Status: frozen redesign authority

Frozen on: 2026-08-03

Scope: research objective, architecture, evidence, and claim contract

Implementation status: design target; it does not activate or alter runtime policy

## 1. Authority and Transition Rule

This document is the single design authority for the next framework version.
It fixes what must be implemented and validated before any new policy is
promoted. `framework_consistency_baseline.md` remains the authority for what the
repository executes today.

The two authorities answer different questions:

| Question | Authority |
| --- | --- |
| What does the current runtime do? | `framework_consistency_baseline.md` |
| What must the redesigned framework mean? | this research contract |

Until the implementation block closes, a requirement in this document is a
design target, not an active capability. A later implementation may not change
this contract silently. Contract changes require a versioned decision record,
new evidence hashes, and revalidation of every affected policy.

## 2. Research Problem

The framework implements the curation layer between a collected text corpus
and language-model training. Given an input corpus `D_raw`, it produces:

```text
C_normal = Curate(D_raw, normal, frozen_config)
C_hard   = Curate(D_raw, hard, frozen_config)
```

Each output includes the materialized dataset and a complete decision trace.
The intended use is continued pretraining of small language models, while the
input and output contract remains independent of a particular model vendor,
source, or content domain.

The scientific target is not to assign an intrinsic universal Quality score to
every document. It is to remove data only through reproducible evidence that
the data is unusable, already represented, or insufficiently supported for the
declared LM-training interface, while preventing unexplained support loss.

## 3. Objective and Non-Objectives

### 3.1 Objective

The framework shall compress a corpus without a fixed retention target. A
frozen policy is successful when its smaller natural-token output maintains or
improves preregistered downstream performance relative to the Stage-A baseline
under the same model, tokenizer, optimizer, and seeds.

The framework must be:

- deterministic for identical input, configuration, and provider artifacts;
- source- and domain-independent at the interface boundary;
- auditable at every removal and transformation;
- conservative when required evidence is unavailable or invalid;
- configurable only through versioned, validated policy objects;
- incapable of reading external benchmark outcomes during curation.

### 3.2 Non-objectives

This project does not claim to:

- measure an intrinsic, context-free Quality property of all text;
- guarantee downstream improvement for every corpus or model;
- force a Code, Math, General, language, or source composition;
- meet a fixed token budget or retention percentage;
- treat source reputation, human intuition, or a weighted heuristic as proof;
- provide production-grade PII, secret, copyright, poisoning, or benchmark-
  contamination guarantees until separately validated policies exist;
- include continued pretraining or benchmarks in the runtime framework.

## 4. Core-Metric-Policy-Method Contract

The architecture has four non-interchangeable levels.

| Level | Meaning | May decide removal? |
| --- | --- | ---: |
| Core | scientific responsibility and authority boundary | No |
| Metric | observable, typed evidence with units and provenance | No |
| Policy | versioned decision rule over authorized Metrics | Yes |
| Method | deterministic implementation of a Metric or Policy | Only through its owning Policy |

A Metric is not a decision rule. A Method is not evidence that its output is
valid. A Core is not complete merely because a function bearing its name runs.
Every active Policy must link to one Core, its permitted Metrics, its Method,
fixtures, evidence artifact, lifecycle state, and reason codes.

## 5. Canonical Cores

### 5.1 Validity

**Question:** Can this unit be interpreted and trained on under the declared
input contract without inventing or destroying its payload?

Validity has deletion or quarantine authority only for a closed, observable
failure such as absent text, undecodable content, unrecoverable corruption,
ambiguous declared text fields, or a promoted parser failure under an explicit
language/version/complete-source contract.

Validity must retain partial snippets, unknown formats, and uncertain cases
unless a policy explicitly supports them. It may not equate unfamiliarity,
difficulty, shortness, language, source, or model loss with invalidity.

### 5.2 Redundancy

**Question:** Is the same training information already represented by another
member of a reproducibly formed family?

Redundancy may propose removal of nonrepresentative family members. Exact
identity is the baseline relation. Near-duplicate relations require symmetric,
length-aware evidence so that a one-token difference in a short sample is not
treated like a one-token difference in a long sample. Each family must have a
stable ID, deterministic representative, relation evidence, and member links.

Redundancy may not use Quality, source rank, domain quotas, Utility, or
benchmark outcomes to select the representative.

### 5.3 Quality

**Question:** Is there validated evidence that retaining this otherwise valid,
nonredundant unit is less useful for the declared LM-training interface than
removing it?

Quality is model- and objective-relative. It is represented by a typed evidence
vector, never by an undocumented universal scalar. Evidence may include closed
structural non-payload conditions and calibrated model-driven signals.

The scientific latent target is expected learning contribution per tokenizer
token under a declared training and evaluation distribution. Runtime Metrics
are estimators or structural evidence for that target; they do not redefine it.
No Quality Policy receives deletion authority merely because a score exists.

Quality decisions are:

| Decision | Meaning | Action |
| --- | --- | --- |
| `reject` | a promoted Policy has sufficient negative evidence | remove or transform with trace |
| `keep` | a promoted Policy has sufficient positive evidence | retain with trace |
| `abstain_retain` | neither direction is sufficiently supported | retain without a Quality claim |

### 5.4 Coverage

**Question:** Did the combined removals erase a valid information family or
support region without an authorized explanation?

Coverage is a corpus-level materialization constraint. It selects or verifies
the representative that must survive a removal family and may veto an output
with unexplained extinction. It does not independently score Quality, impose a
target domain mix, restore records by quota, or delete common content because
it is common.

Coverage audits route, script/language, format, structural family, and any
separately validated semantic grouping. Unknown is a valid audit label. Source
identity may be reported for provenance but is not a selection axis.

## 6. Stage Contract

The public stages are fixed as follows:

| Stage | Scope | Primary Core authority | Output |
| --- | --- | --- | --- |
| Stage A | record/chunk hard gate | Validity | valid baseline plus quarantined records |
| Stage B | family and payload decisions | Redundancy and Quality | retained records plus typed removal proposals |
| Stage C | corpus materialization | Coverage | final dataset or fail-closed veto |

Stage A is the common baseline for all development and confirmatory arms.
Stage B must not optimize a target output size. Stage C may not introduce a new
ranking objective or an equal-token selector; it verifies support and writes
the final dataset.

## 7. Runtime Input Boundary

### 7.1 Allowed inputs

An active Policy may use only fields declared in its policy card, including:

- text and deterministic normalized forms;
- stable record, chunk, and family identifiers;
- declared parsing metadata when the parser contract requires it;
- local structural evidence produced by a frozen Method;
- frozen model evidence joined by stable ID when its provider has been
  calibrated and promoted;
- prior-stage reason codes and representative links.

### 7.2 Forbidden inputs

Runtime selection may not read:

- Utility, downstream NLL, or benchmark outcomes;
- target retention fraction, maximum token budget, or desired compression;
- source reputation, source tier, or known-high-quality labels;
- desired domain, language, format, or demographic quota;
- confirmatory results from the corpus being selected;
- unversioned human Quality judgments;
- undeclared metadata or provider outputs.

Optional rights, hazard, and provenance metadata remains audit-only until a
separate validated safety Policy explicitly owns it.

## 8. Normal and Hard Profiles

The only public profiles are `normal` and `hard`. Both profiles expose the same
Core and Policy families. Their difference is a versioned, independently
calibrated operating point for Policies whose Metrics support graded evidence.

`normal` uses the more conservative removal operating point. `hard` uses a
stronger operating point that may remove additional units only where its own
false-positive, Coverage, and external-evidence gates pass. Binary Policies
without a meaningful calibrated strength, including closed Validity failures
and exact identity, behave identically in both profiles.

For the same Stage-A baseline and framework version:

```text
retained(hard) subset-or-equal retained(normal)
```

This monotonicity is a release invariant. Hard is not Normal with an arbitrary
lower score threshold: each operating point has its own immutable provenance
and calibration-artifact hash, and missing calibration forces abstention.
Neither profile has a retention ratio or token budget. If a corpus contains no
supported removal opportunity, either profile may retain nearly all Stage-A
data.

For sensitivity calibration, all candidate arms must reference one identical
Stage-A baseline. That baseline must be record- and source-disjoint from every
arm, and sensitivity arms must be pairwise disjoint. A different baseline per
arm is forbidden because it confounds the policy effect with baseline sampling.
Normal and Hard thresholds may be emitted only after each required route has at
least three ordered empirical effect bins and separately frozen natural-budget
external evidence.

## 9. Model-Driven and Contrastive Quality Evidence

Model-driven evidence is optional and replaceable. It is not a hidden framework
dependency. Each provider declares model revisions, tokenizer, precision,
normalization, context length, truncation behavior, and evidence hashes.

### 9.1 Model roles

| Role | Purpose |
| --- | --- |
| target SLM | represents the model family intended to learn from the corpus |
| quality reference | models the declared desired training distribution |
| broad background | models broad availability rather than the desired distribution |

The quality reference must be trained or adapted on a declared, disjoint,
multi-domain reference pool. A larger generic base model is not automatically a
quality reference.

### 9.2 Directional Metrics

For tokenizer-normalized loss `L`:

```text
LearnabilityGap(x) = L_target(x) - L_quality_reference(x)
AlignmentGap(x)    = L_quality_reference(x) - L_background(x)
```

A high positive LearnabilityGap is evidence that the target can still learn
content captured by the reference. It is therefore keep evidence, not direct
removal evidence. A high positive AlignmentGap is candidate evidence that the
content is better explained by the broad background than by the desired
distribution. It may contribute to a reject Policy only after calibration.

The Metrics remain separate. They may not be collapsed into an arbitrary
weighted sum. Absolute thresholds may not transfer across providers, routes, or
tokenizers.

### 9.3 Provider replacement

Users may replace any model role through configuration. A change in model,
revision, tokenizer, precision, quantization, normalization, context policy, or
reference-pool contract invalidates prior calibration. Until recalibration
passes, the affected Policy must abstain and retain.

The existing Qwen3-4B versus Qwen3-8B audit is diagnostic evidence only. It
does not define either public profile and has no deletion authority.

## 10. Threshold and Evidence Provenance

Every threshold or categorical boundary must record:

- value, unit, and comparison direction;
- derivation procedure and development-corpus identity;
- sample count and supported routes;
- provider and tokenizer identities when model-driven;
- confidence interval or uncertainty procedure where applicable;
- fixture, ablation, and external-evidence artifact hashes;
- lifecycle state: `candidate`, `development_passed`, `promoted`, `blocked`, or
  `retired`;
- invalidation conditions.

A convenient value, inherited constant, or successful single run is not a
valid derivation. Missing provenance forces `abstain_retain` or prevents the
profile from loading.

## 11. Development and Confirmatory Separation

Development and confirmatory data must be disjoint by stable record ID and
normalized-text hash. Where time matters, the temporal boundary must also be
declared.

Every sensitivity audit uses one frozen Stage-A baseline sample `B_A` that is
normalized-text-hash disjoint from the union of its sensitivity-arm samples.
Every arm is compared with that same `B_A`; an arm may not receive its own
baseline. The `B_A` hash and each disjointness result are included in every
sensitivity report.

This audit sample is distinct from the training comparison below. Normal and
Hard training corpora are intentionally derived subsets of the same Stage-A
corpus and therefore are not disjoint from the Base training corpus.

Development may inspect fixtures, compression, reason-code effects, Coverage
impact, and external development evaluations. Confirmatory data and outcomes
remain hidden until the policy, profile, thresholds, and acceptance rule are
frozen. A failed confirmatory result cannot be tuned and reported as the same
confirmatory experiment.

## 12. Promotion Gates

### 12.1 Framework gate

The framework implementation must pass:

- deterministic replay and stable representative selection;
- complete config, policy, provider, input, and output hashes;
- forbidden-input and stage-boundary tests;
- complete reason-code and token-delta accounting;
- Normal/Hard monotonicity;
- fail-closed behavior for missing evidence;
- clean-corpus and high-quality-corpus retention checks;
- Coverage representative and zero-survivor invariants.

### 12.2 Policy gate

Every promoted Policy must pass:

- explicit trigger and non-trigger contracts;
- positive, false-positive, adversarial, and clean-corpus fixtures;
- rule-off versus rule-on development ablation;
- route and corpus transfer analysis for its claimed scope;
- Coverage impact analysis;
- preregistered acceptance criteria and uncertainty reporting;
- benchmark-disjoint natural-budget external evaluation.

No universal numerical false-positive threshold is invented here. Each Policy
must preregister a threshold appropriate to its risk and intended claim before
confirmatory execution.

### 12.3 Evidence gate

External comparison uses exactly three data arms derived from the same Stage-A
corpus:

| Arm | Training data |
| --- | --- |
| Base | all Stage-A survivors, natural token budget |
| Normal | Normal output, natural token budget |
| Hard | Hard output, natural token budget |

Training uses the same model, tokenizer, optimizer contract, and seeds
`101/202/303`. Primary outcomes and non-inferiority margins are preregistered by
domain. Exact tokenizer tokens, examples, updates, and wall-clock compute are
all reported; equal-token comparison is optional diagnosis, not the primary
claim. The untouched pretrained model may be benchmarked as a reference point,
but it is not a fourth curation data arm.

Code, Math, and General evidence are reported independently. Evidence in one
domain does not license a downstream-effectiveness claim in another.

## 13. Success and Claim Boundary

The project may make claims at three levels:

| Level | Required evidence | Permitted claim |
| --- | --- | --- |
| Framework | architecture and framework gate pass | auditable, domain-independent curation interface |
| Policy | named Policy gates pass in stated scopes | validated operation within those scopes |
| Downstream | frozen three-seed natural-budget confirmatory pass | retained performance at lower data cost for the tested model/domain/benchmarks |

`domain-general` describes the framework interface only after Code, Math, and
General inputs execute through the same contracts without domain-specific entry
points. It does not mean every Policy fires in every domain. Downstream
domain-general effectiveness requires independent confirmatory suites in all
claimed domains.

An acceptable positive result is either:

- fewer natural training tokens with statistically supported non-inferior
  benchmark performance; or
- fewer natural training tokens with improved benchmark performance.

Compression alone is not success. Benchmark improvement alone does not prove
the removed records were universally low Quality.

## 14. Abstention and Failure Semantics

The runtime must abstain and retain when evidence is missing, provider identity
does not match calibration, routing is unsupported, uncertainty crosses the
decision boundary, or required trace fields are absent.

The run must fail closed before materialization when configuration hashes do not
match, profile monotonicity is violated, a removal family has no explained
survivor, or audit accounting cannot reconcile input and output.

External release remains blocked when a required confirmatory domain,
retention guardrail, or benchmark result is missing. `blocked` is a scientific
result and may not be converted into `pass` by relaxing a frozen threshold.

## 15. Required Release Artifacts

Every released run must include:

- frozen framework and profile configuration;
- input manifest and stable corpus fingerprint;
- Stage-A baseline manifest;
- Normal and Hard curated JSONL outputs;
- selected, removed, quarantined, transformed-span, and representative-link
  traces;
- reason-code counts and exact tokenizer token deltas;
- Core, route, script/language, format, and family Coverage reports;
- provider manifests and joined-evidence hashes when model-driven;
- fixture and behavior-audit reports;
- development ablation and common-baseline reports;
- seed-level external results and aggregate uncertainty;
- explicit failed, blocked, and retired Policy inventory.

## 16. Block 1 Frozen Decisions

The following decisions may not drift during implementation:

1. The public Cores are Validity, Redundancy, Quality, and Coverage.
2. The hierarchy is Core-Metric-Policy-Method.
3. Stage A is the Validity hard gate, Stage B owns Redundancy and Quality
   proposals, and Stage C owns Coverage and final materialization.
4. Quality is conditional learning contribution, not intrinsic document merit.
5. Coverage preserves explainable support and has veto authority, not quota or
   Quality authority.
6. Public profiles are Normal and Hard. They share the same Policy families,
   use independently calibrated operating points, and require Hard retained
   data to be a subset of Normal retained data.
7. Neither profile uses a fixed token budget or retention fraction.
8. Contrastive Metrics are directional and role-specific; generic model-size
   gaps have no deletion authority.
9. Runtime never consumes Utility, NLL, or benchmark outcomes.
10. External validation is three-seed, natural-budget, common-baseline, and
    benchmark-disjoint.
11. Claims are separated into framework, Policy, and downstream evidence.
12. The existing runtime remains unchanged until the implementation block is
    explicitly completed and revalidated.
