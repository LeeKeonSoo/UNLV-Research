# Aggressive Quality Curation Program

> Historical design note. Current Quality authority is
> `docs/quality_teacher_panel_v1.md` together with
> `configs/quality_teacher_panel_v1.json`. References below to verified
> learnable-payload density or deterministic-only Quality do not override the
> final Q1-Q4 independent Policy contract.

## Decision

The framework will optimize **verified learnable-payload density**, not a
subjective per-document Quality score and not a preset token-retention target.
The operational question is:

> Can this text be shown to contribute no distinct learnable payload under a
> reproducible structural rule, while a coverage guard preserves a comparable
> representative?

Only a yes can authorize deletion. The policy is aggressive because it seeks
the smallest externally non-inferior natural-budget corpus, rather than because
it assigns a harsher arbitrary score.

## Formal Contract

For raw corpus `R`, candidate rule set `P`, curated corpus `C(P, R)`, and
tokenizer count `T`, select a deployable policy set using:

```text
P* = argmin_P T(C(P, R))
     subject to
       FPR(r) <= alpha_r for every r in P,
       coverage_loss(P, R) <= delta,
       benchmark_delta(P, R) >= -epsilon
       for the frozen external protocol.
```

`FPR(r)` is not guessed from corpus output. It is measured from the declared
false-positive and adversarial fixture suite, then supplemented by a sampled
reason-code audit. `coverage_loss` is an audit constraint: it protects observed
structural/content strata from accidental disappearance; it never enforces a
domain quota. `benchmark_delta` is measured only after policy freeze and is
never an input to Stage A, B, or C.

The parameters `alpha_r`, `delta`, and `epsilon` are preregistered with the
policy experiment. They are not inferred from a desired retained-token count.

## Quality Cores

| Core | What counts as positive evidence | Allowed runtime action | What it must not mean |
| --- | --- | --- | --- |
| Validity | Deterministic corruption, empty/too-short normalized content, or a declared-parser failure with a versioned grammar | Quarantine or reject the affected record/chunk | A stylistic preference or source reputation |
| Redundancy | Exact digest, symmetric lexical/semantic duplicate family, or repeated normalized span with a stable representative | Retain one representative and remove confirmed copies | Removing distinct examples merely because their topic is similar |
| Quality | Explicit non-payload artifact, license-only block, or a measured template/boilerplate span whose removal preserves substantive payload | Remove the artifact or compact the repeated span/family | A hand-tuned subjective score |
| Coverage | Pre/post retention of observed content, language/script, format, and structural buckets | Block or flag a policy that makes an observed bucket disappear | Selecting a preferred domain percentage |

**Quality** is the fourth Core. In Normal mode it is limited to explicit
non-payload evidence and payload-preserving structural compaction; it does not
rank surviving texts by intrinsic merit. The future Hard mode may add only
separately validated deterministic structural policies; model-relative
candidate research is not part of either user-facing runtime mode.

## Candidate Rule Ladder

Rules move one direction only: inventory -> fixture -> candidate ablation ->
external validation -> active. No rule becomes active simply because it removes
many tokens.

| Priority | Candidate family | Deterministic/quantitative evidence | Essential false-positive boundary | Expected effect |
| --- | --- | --- | --- | --- |
| 1 | Cross-record and within-record exact repeated spans | Normalized span digest; frequency and copied-token coverage; retain a stable first representative | Repeated API names, short phrases, or a span embedded in distinct substantive context | Remove copied boilerplate without deleting the surrounding payload |
| 2 | Stronger duplicate families | Symmetric 5-shingle containment and a family representative trace; candidate semantic evidence may only propose a family | Same-topic but independently written examples; distinct implementations with shared imports | Reduce repeated examples and memorization pressure |
| 3 | Non-learning structural artifacts | Explicit textual marker plus whole-record or residual-payload check | Authored documentation, generated declaration without “do not edit,” and executable code with a license header | Remove machine-generated or non-executable material only when evidence is explicit |
| 4 | Declared-language invalid or non-executable content | Parser/compiler result bound to declared language and version | Snippets, partial examples, documentation, or mismatched/unknown language version | Reject broken source only inside the declared policy scope |
| 5 | High-repetition boilerplate/template families | Exact normalized-line/span recurrence plus a measurable payload-preservation ratio | Legitimate public APIs, test matrices, configuration variants, and translated content | Compact templates only when the repeated portion, not the substantive variation, is removed |

Priorities 1 and 5 must operate at span level where possible. Deleting an
entire record for a repeated header is unnecessarily destructive and obscures
the quality claim.

## Promotion Gate

For each rule, the repository must create and pass the following artifacts:

1. Registry entry and policy card: Core, Metric, Policy, scope, allowed inputs,
   prohibited inputs, decision unit, and rollback.
2. Fixture suite: positive, false-positive, adversarial, and clean-corpus cases.
3. Reason-code impact audit: removed records/chunks/tokens and retained
   representative linkage.
4. Case-matrix report: clean, duplicate-heavy, boilerplate-heavy, malformed,
   and cross-domain raw-like inputs.
5. Rule-off versus rule-on development ablation.
6. Frozen, three-seed, natural-budget external validation; the policy does not
   see the benchmark it will be judged on.

Failure at any stage means `candidate` or `retired`, never silent activation.

## Implementation Sequence

### Block Q1 - Freeze the Meaning of Quality

Completed by this document and `lm_training_quality_definition.md`. The active
runtime remains unchanged while the rule standard is fixed.

### Block Q2 - Measure Removal Opportunities

Inventory exact repeated spans, duplicate-family sizes, explicit artifact
records, parser failures in declared language scopes, and candidate
payload-preservation rates on raw-like Code, Math, and General corpora. This
is diagnostic only.

### Block Q3 - Build Candidate Rules With Fixtures

Implement the two highest-yield candidates separately: span-level repeated
boilerplate compaction and declared-language validity. Each ships with reason
codes and false-positive fixtures before it can touch a corpus.

Implemented candidate runner: `aggressive_structural_candidate_runner.py`
materializes five isolated text-only arms from a frozen Stage-B snapshot:
active baseline, license-span compaction, repeated-span compaction,
strengthened duplicate-family compaction, and their cumulative combination.
It also emits a 0.90/0.92/0.95 duplicate-threshold sweep. The runner requires
a frozen tokenizer path to count output tokens, but that count is reporting
only and never enters a selector decision.

### Block Q4 - Select an Aggressive Policy Empirically

Run rule-on/off and cumulative candidate ablations. Choose the most compressed
candidate that passes the artifact, coverage, and fixture gates. This becomes a
frozen candidate profile, not yet a production claim.

The Weak development gate is now executable through
`weak_development_gate.py` and its frozen fixture matrix. Its 12 scenario
report covers Code, Math, and General clean, duplicate-heavy,
artifact-heavy, and malformed inputs. In the current fixture run, all clean
arms retained their original token proxy; duplicate-heavy arms removed only
`normalized_exact_duplicate`; malformed arms reported only `payload_absence`;
and the explicit artifact arms removed 8 tokens for a license-only Code chunk,
3 tokens for an empty HTML shell, and 13 tokens for a cookie-control-only
General chunk. This is a structural boundary check, not a corpus-scale or
downstream-performance result.

### Block Q5 - Confirm With Natural-Budget LM Evaluation

On a benchmark-disjoint corpus, compare Raw with the frozen curated output
using identical model and training hyperparameters at each corpus's natural
token budget and three seeds. Promote only policies whose compression is paired
with retained or improved benchmark performance according to the preregistered
non-inferiority criterion.

The first Code development materialization on the frozen 7,383-chunk Stage-B
snapshot is stored outside the repository under
`D:\UNLV-Research\code_5m_corpus_v2\aggressive_structural_candidate_v1`.
With the frozen Qwen3 tokenizer, its active baseline contains 6,358,460 tokens
and the cumulative candidate contains 6,168,129 tokens (2.993% fewer than the
baseline). This is a candidate-development observation, not promotion or a
downstream-performance claim.

## Explicit Exclusions

The runtime will not use a global Quality score, source reputation, human
preference labels, target token fraction, domain quota, Utility, NLL, or
benchmark results. Those may be audit metadata or external evaluation evidence;
none is a hidden deletion signal.
