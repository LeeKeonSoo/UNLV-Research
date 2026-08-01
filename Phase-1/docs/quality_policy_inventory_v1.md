# Quality Policy Inventory v1

## Purpose

This inventory is the Quality-Core decision record for the active A-B-C
curation runtime. It prevents a rule from receiving Quality authority merely
because it reduces tokens or resembles a historical selector feature.

Quality is the retention-eligibility Core applied after Redundancy. It may
reject a whole unit proven to be explicit non-payload, or compact a
structurally separable non-payload span while the residual remains
Validity-compliant. If no active rule has sufficient rejection evidence, the
decision is `ABSTAIN_RETAIN`; this is not an intrinsic document score or a
claim that the survivor is universally high quality.

## Activation Standard

An active Quality policy must have all of the following:

1. a deterministic text-only trigger;
2. an executable non-trigger and false-positive fixture;
3. a stable reason code and token delta;
4. an explicit whole-unit non-payload finding or a residual-payload check;
5. a Coverage-compatible zero-survivor explanation or span trace.

Development candidates additionally require Code, Math, and General
development ablation before frozen external confirmation. Runtime may never
read a source reputation, domain quota, Quality score, Utility, NLL, benchmark
outcome, or token budget.

## Evidence Boundary

A policy is not treated as a universal statement that removed text can never
help an LM. It is a bounded structural hypothesis: under its declared input
contract, the rule identifies a specified form of non-payload or redundant
text. Before activation, the hypothesis must survive an executable trigger,
false-positive and adversarial boundary fixtures, a reason-code/coverage audit
on Code, Math, and General development corpora, and frozen benchmark-disjoint
external evaluation. A failed gate archives the policy; it is never widened by
a score, model prediction, or retention target.

## Active Quality Policies

| Policy | Decision unit | Trigger | Protected boundary | Reason code | Status |
| --- | --- | --- | --- | --- | --- |
| Explicit generated artifact | Entire source record | The record contains both an in-text generated marker and a do-not-edit marker | A generated declaration alone, authored generated code, and a missing non-editable marker are retained | `explicit_generated_artifact` | Active Normal |
| License-comment-only chunk | Chunk | Every nonblank line is a comment and the chunk has an explicit copyright or license marker | Executable code, documentation, and mixed comment/payload chunks are retained | `license_comment_only_chunk` | Active Normal |
| Empty HTML shell | Chunk | A complete HTML wrapper has no visible lexical payload after tag removal | HTML articles and documents with visible text, scripts, styles, images, or embedded content are retained | `empty_html_shell` | Active Normal |
| Cookie-control-only panel | Chunk | At least four nonblank lines, all from the closed cookie-control UI marker set | A prose discussion of cookies, privacy terms, or a panel with any substantive line is retained | `explicit_web_chrome_only_chunk` | Active Normal |

These rules are intentionally precision-first. Their current role is to
establish a safe Quality floor, not to claim a target compression rate.

## Development Candidates

| Policy | Intended action | Current evidence | Missing promotion evidence | Status |
| --- | --- | --- | --- | --- |
| Prefix license-header compaction | Remove only an explicit prefix license span when the residual meets the Stage-B boundary | Deterministic fixture and Hard runtime trace | Rule-level Code/Math/General ablation and frozen external confirmation | Hard candidate |
| Inline license-comment-block compaction | Remove a self-contained explicit license comment span when the residual is valid | Deterministic fixture and Hard runtime trace | Rule-level Code/Math/General ablation and frozen external confirmation | Hard candidate |
| Exact repeated-template-span compaction | Remove a long exact repeated paragraph only from nonrepresentative chunks | Deterministic fixture, stable span trace, residual check | Rule-level Code/Math/General ablation and frozen external confirmation | Hard candidate |
| Explicit web-control and URL-directory span compaction | Remove contiguous control-line or URL-only runs while preserving a substantive residual | General-web candidate fixture with dialogue protection | Registry card, Code/Math/General matrix, reason-code impact audit, and frozen confirmation | Candidate only; not callable by runtime |
| Explicit error-navigation-only chunk | Reject only a whole chunk composed of one closed error marker and at least three closed navigation labels | Trigger/non-trigger fixture and development-only runtime guard | Independent corpus opportunity, zero-survivor review, and frozen external confirmation | Candidate only |
| URL-directory-only chunk | Reject only a whole chunk containing at least five standalone URL lines and no other payload | Trigger/non-trigger fixture and development-only runtime guard | Independent corpus opportunity, reference-list false-positive review, and frozen external confirmation | Candidate only |

## Quality Retention Candidate Replay

The 2026-07-31 replay applied the same frozen Quality candidate profile to
RedPajama V2 tail, GitHub Code, FineWeb-Edu, and OpenWebMath Stage-B survivor
pools. Counts below use the report's whitespace-token proxy.

| Corpus | Stage-B chunks | Active Quality token removal | Error-navigation candidate | URL-directory whole-chunk candidate | Web/URL span candidate | Cumulative removal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RedPajama V2 tail | 18,719 | 0 | 0 | 0 | 377 tokens, 29 spans in 25 chunks | 377 |
| GitHub Code | 9,732 | 18,795 | 0 | 0 | 0 | 18,795 |
| FineWeb-Edu | 6,811 | 0 | 0 | 0 | 0 | 0 |
| OpenWebMath | 4,356 | 0 | 0 | 0 | 0 | 0 |

Every cumulative arm passed the residual-payload Coverage audit. The two new
whole-chunk rules found no opportunity and therefore remain unpromoted. The
RedPajama span result is development evidence only; its policy still requires
false-positive review on the 25 affected chunks and frozen natural-budget
external validation.
| Repeated closed-set navigation-label block compaction | Remove only the later exact occurrence of a short navigation-marker block within the same chunk | Positive and false-positive fixture; first occurrence and residual are preserved | Code/Math/General matrix, reason-code impact audit, and frozen confirmation | Candidate only; not callable by runtime |

Candidate status means that a rule cannot affect a production materialization.
The existing confirmatory Hard experiment may exercise its declared frozen
license/template candidates only as external evaluation, not as a promotion.

### Q2 Package Decision

`stage_c_explicit_web_control_span_candidate` is now registered in both
`configs/policy_cards.json` and `configs/core_policy_registry.json`. Its
executable fixture proves only the closed-set web-control and URL-directory
cases stated above. The registry explicitly sets
`runtime_authorization: none_candidate_cannot_select_or_remove`.

Generic footer, navigation, and placeholder removal are **not** registered as
candidates yet. Their surface words can occur in a substantive article,
documentation page, dialogue, or reference list, so text-only triggers are not
currently precise enough. Q2 records this as an unresolved rule-design task,
not as a silent gap filled with a heuristic.

The repeated-label-block candidate does not change that conclusion. It is
restricted to a small closed navigation-marker set and retains a document's
first occurrence. It must retain repeated headings, quotations, tables, code,
reference entries, and test matrices. It is registered for auditability, not
because repeated labels generally prove redundancy.

### Q7 Development Result

The frozen-tokenizer matrix found zero candidate spans in all three
development snapshots: Code (8,058 chunks, 6,873,133 tokens), Math (3,632,
2,915,236), and General raw web (707, 940,958). Every Coverage invariant
passed because no chunk was rewritten. This is negative evidence for utility,
not evidence to widen the trigger. The candidate remains non-runtime and is
pending an explicit archive-or-research decision.

The original Code snapshot used the historical flattened long-paragraph
chunker. Its v1 Code count is superseded by the format-preserving rerun: 7,309
chunks, 6,400,945 frozen-tokenizer tokens, zero candidate spans, and zero
token delta. The Math and General v1 snapshots were not affected by this Code
formatting defect.

### Q8 Archive Decision

`stage_c_repeated_label_block_candidate` is retired. It passed its narrow
fixtures but fired zero times in corrected Code, Math, and General development
inputs, so it provides no observed compression opportunity. The remaining
Code one/two-line diagnostic surface is primarily static literals and data
tables; no default deletion authority follows from its line shape or size.
Details are in `docs/r2_static_literal_boundary_decision_v1.md`.

## Reclassified Or Archived Items

| Item | Disposition | Reason |
| --- | --- | --- |
| Structural scaffold family compaction | Reclassify to **Redundancy** | It retains one member of an identical normalized family. Its authority is representative compression, not a claim that scaffold text is intrinsically low Quality. |
| Symmetric near-duplicate compaction | Redundancy candidate only | The frozen Normal and Hard protocols set `candidate_enabled: false`; overlap alone does not prove non-payload. |
| Model-relative representative selector | Archive | Familiarity, novelty, gradient, or semantic proxy evidence is not a stable text-structural Quality authority. |
| Mid Quality estimator / reference-quality score / weighted priority formula | Archive | These are model-relative or weighted proxy mechanisms and violate the no-intrinsic-score runtime contract. |
| Declared dependency-copy policy | Archive | It depends on source metadata, whereas the active curation policy is text-only and source-agnostic. |
| Source reputation, path, domain quota, retention budget | Prohibited | They may appear only in audit sidecars and cannot authorize a Quality deletion. |

## Reconciliation Status

1. **Resolved:** `structural_scaffold` is labeled Redundancy in the registry
   and policy card. Its executable stable-family representative behavior did
   not change.
2. **Resolved:** the README, handoff, active framework document, and a machine
   check state that symmetric near-duplicate compaction is disabled in the
   frozen Normal and Hard protocols.
3. **Resolved:** `general_web_span_compaction.py` has a candidate-only fixture
   and a matching registry entry and policy card; it remains non-runnable.
4. **Resolved:** `archive/historical_contracts/metric_spec_with_citations.md`
   is visibly marked as archive-only historical evidence.
5. Historical validation artifacts can still mention `reference_quality_score`
   because they preserve old experiment results. They are not inputs to the
   active runtime and do not require rewriting.

## Historical Boundary

The archived files remain retained for experiment provenance. In particular,
they must not be used to infer an active selector behavior merely because they
contain historical Quality, Utility, or score vocabulary.

## Q2 Findings That Must Be Carried Forward

1. Generic footer, navigation, and placeholder removal lack a safe closed
   text-only trigger and therefore remain unregistered.
2. `general_web_span_compaction.py` is registered but remains inert until a
   future development matrix establishes more than its local fixture.

## Q2 Entry Criteria

Q2 may add only candidates that satisfy the Activation Standard. It must first
create a policy card, registry entry, reason code, required inputs,
trigger/non-trigger/adversarial fixtures, and a Coverage impact assertion for
each proposed rule. No score formula or training-objective-specific branch is
permitted.
