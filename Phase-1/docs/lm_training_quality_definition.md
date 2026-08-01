# LM Training Quality Definition

## Objective

The framework turns a raw candidate corpus into an auditable corpus that is more suitable for language-model training. It does not claim to assign a universal intrinsic quality score to every text.

For this framework, training-data quality means **verified learnable-payload density under a fixed training recipe**. A corpus is better when it retains distinct, intact learning content and removes evidence-backed content whose marginal training contribution is known to be zero or repeated, while preserving or improving downstream performance at its natural retained-data budget.

This is deliberately a corpus-level definition. The framework does not pretend that a scalar attached to one chunk can capture every form of educational, semantic, or task value.

## What Is Measured

Quality is measured at two different levels.

1. Local operational evidence answers whether a record has a known, reproducible training defect.
2. External evaluation answers whether the frozen curated corpus helps a model more than the corresponding raw corpus.

The first level authorizes runtime actions. The second level validates the framework after curation and never feeds back into the same curation run.

## Operational Dimensions

| Dimension | Question | Runtime action | Evidence required |
| --- | --- | --- | --- |
| Validity | Is this an intended, interpretable text unit under its declared contract? | Quarantine only payload absence, text-contract violation, unrecoverable corruption, or acquisition failure; reject only invalid chunk results. | Deterministic fixture and reason code. |
| Redundancy | Does this chunk add content beyond an observed duplicate family? | Retain one stable representative and remove supported duplicates. | Symmetric duplicate evidence and representative trace. |
| Quality | Does a distinct, valid unit satisfy the active retention-eligibility policies? | `REJECT` only with declared non-payload evidence; otherwise `ABSTAIN_RETAIN`. `KEEP` requires a separately validated positive policy. | Explicit textual evidence, payload-preservation fixture, decision trace, and reason code. |
| Coverage | Did a removal rule accidentally erase an observed structural bucket? | Audit retention effects; do not impose a target domain mix. | Post-hoc composition and retention report. |

These dimensions are deliberately not combined into a weighted score. A weighted score would hide which assumption caused removal and would allow arbitrary trade-offs between unrelated defects. A chunk is removed only by a named policy whose trigger evidence and non-trigger boundary are both executable.

Quality is therefore a retention Core, not a universal document grader.
Redundancy asks whether the payload is distinct; Quality asks whether that
distinct payload has enough evidence to remain eligible. An abstention retains
the unit and explicitly makes no intrinsic-quality claim.

Validity does not treat short snippets, partial files, examples containing an
error, tables, equations, JSON, HTML, Markdown, multilingual text, or an
unfamiliar format as invalid. Declared-language parser failures are candidate
evidence only unless a separately validated complete-artifact adapter is
promoted.

## Aggressive Curation Objective

The objective is not merely to preserve a mostly unchanged corpus. It is to find
the most compact corpus allowed by the evidence and validation constraints.
For a frozen raw corpus `R` and a policy set `P`, let `C(P, R)` be the curated
output and `T(.)` its tokenizer-specific token count. The development objective
is:

```text
minimize    T(C(P, R))
subject to  every active removal has a reason code and executable trigger,
            every rule passes its false-positive/adversarial fixtures,
            observed coverage loss stays within the declared guardrail,
            external natural-budget evaluation is non-inferior to Raw
            under a frozen training and benchmark protocol.
```

Equivalently, the framework maximizes verified compression
`1 - T(C(P, R)) / T(R)`, not a target retention percentage. The runtime never
reads the external benchmark: the benchmark chooses whether a candidate policy
may be promoted *before* a future run is frozen.

This makes a clean corpus a valid outcome: if no rule has evidence, it should
be retained. Conversely, a raw-like corpus with repeated templates, copied
boilerplate, malformed artifacts, or duplicate families should shrink
substantially once the corresponding rules have passed their gates.

## Rule Evidence Standard

An active removal rule must describe a measurable phenomenon rather than a
judgment that text "looks bad." Its card must contain all of the following:

1. A mathematical or deterministic trigger, such as symmetric shingle overlap,
   an exact normalized-span frequency, or an explicit generated-and-noneditable
   declaration.
2. A non-trigger boundary that keeps a nearby useful case, such as distinct
   implementations with shared imports or a file that merely says
   "generated" without a noneditable declaration.
3. A reason code, representative trace where applicable, and token removal
   accounting.
4. Positive, false-positive, and adversarial fixtures.
5. A development ablation and external natural-budget evaluation before the
   rule receives active removal authority.

Candidate families are: exact/repeated span or template compaction with
payload preservation; high-confidence duplicate-family selection; explicit
non-learning artifacts; and declared-language syntax or format failures with a
versioned parser. Model-relative familiarity, source reputation, domain quota,
and hand-tuned weighted scores remain excluded until they satisfy the same
evidence standard.

The active Normal profile additionally recognizes two deliberately narrow
whole-chunk artifacts: a complete HTML shell with no visible lexical payload,
and a cookie-control panel whose every nonblank line belongs to a fixed UI
marker set. These rules do not remove an HTML article, embedded script/style
content, or explanatory prose containing similar words. Placeholder and
separable-boilerplate rules remain candidates, not active deletion policies.

## What Is Not a Runtime Quality Signal

The runtime must not use source identity, domain label, human quality label, model utility, NLL, benchmark outcome, target retention fraction, or a fixed token budget to decide removal.

Metadata can support audit, safety review, and experiment hygiene. It cannot select records in the active text-only profile.

## Claim and Validation

The active framework may claim that it produces a reason-coded, defect-reduced training corpus. It may not claim that every survivor is intrinsically high quality.

The research claim is evaluated externally: train Raw and Curated arms at their respective natural retained-data budgets, with frozen model, seed, tokenizer, and benchmark protocols. Curated performance that is retained or improved with less retained data is evidence that the operational defect policy was useful for that setting. It is not proof of universal quality.

## Normal And Hard Modes

The user-facing framework exposes two modes only. **Normal** is active now:
it applies the high-precision structural rules defined above. **Hard** is an
opt-in stronger structural profile. Its current implementation is restricted to
an explicit `execution_scope: "development"`: it applies Normal plus three
deterministic payload-preserving span compactions, writes every transformation
trace. Its Code/Math/General fixture ablation has passed; it remains fail-closed
for production curation until frozen external validation closes. It will not use a model-relative score, source/domain
metadata, a Quality scalar, or a target retention fraction.

Stronger mode is a containment contract, not a promised deletion rate:
`Hard subset Normal` once Hard is active. A clean corpus can therefore produce
nearly identical outputs. The required difference is a wider *validated*
reason-coded policy set, never forced compression.

Hard v1 is intentionally limited to three span-level candidates listed in
`configs/hard_policy_inventory_v1.json`: explicit prefix-license headers,
self-contained license comment blocks, and long exact repeated template spans.
Each must preserve a Stage-B-valid residual and emit a digest, token delta, and
representative/span trace. Near-duplicate thresholds alone, model-relative
proxies, source-backed dependency labels, and parser adapters are excluded from
this first Hard profile because their removal boundaries are not yet adequate.

## Archived Candidate Research

The earlier Stage C2 proxy, Mid estimator, and token-budget planner remain
candidate research artifacts only. Their known-reference false-positive risk
means they are neither user-facing modes nor runtime inputs. They may inform
future rule research but cannot authorize current Stage C removal.
