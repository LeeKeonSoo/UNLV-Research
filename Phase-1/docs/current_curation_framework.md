# Current Curation Framework

> **Status note:** `framework_consistency_baseline.md` is the sole authority for
> current runtime status during the consistency reset. This document preserves
> detailed design context and may include candidate or historical material that
> is not runtime-active.

## Scope

The active implementation materializes an auditable candidate pool from a
domain-agnostic raw JSONL contract. It derives composition metadata from text,
but composition has no selection authority. Its purpose is curation, not
downstream model evaluation.

```text
collect candidate pool
  -> Stage A: source-agnostic text normalization and integrity handling
  -> Stage B: chunk-level hard gate
  -> Stage C: text-only reason-coded duplicate, generated-artifact, license-comment-only, and structural-scaffold compaction, then materialization
```

External training, NLL, and benchmark measurements are not part of this
runtime. They cannot influence a frozen curation output.

The current Code confirmatory input is separated from the development input by
stable record ID and normalized-text hash, each side has a complete frozen
seven-benchmark exclusion audit, and both materializations carry the same
policy fingerprint. The external integrity report is
`D:\UNLV-Research\code_confirmatory_v1\external_validation_integrity_report.json`.
It establishes input separation only; it is not a downstream performance
result and is never read by the runtime.

## Core-Metric-Policy Roles

The active user-facing runtime profile is **Normal** (`normal_structural_v1`), declared in
`configs/policy_profiles.json`. It is deliberately limited to high-precision,
reason-coded text-structural policies. `safe_structural_v3` is retained only to
reproduce historical provenance-and-safety experiments. The historical
score-selector template is retired and cannot run. The required fields for a versioned active policy card are declared
in `configs/policy_card_contract.json` and instantiated in
`configs/policy_cards.json`. `configs/core_policy_registry.json` is the
authoritative lifecycle registry: every active rule is currently
`active_structural`, linked to its required metadata, reason codes,
false-positive fixture, case-matrix scenario, coverage-impact validation, and
promotion requirements. `active_structural` means the executable boundary is
tested; it does not mean the rule has downstream performance validation. The
four canonical Core names are **Validity, Redundancy, Coverage, and Quality**.
Historical artifact identifiers may retain older labels only in the explicit
`legacy_core_aliases` compatibility map; those labels are not public Cores.

| Core | Observable metric | Policy authority in the active implementation |
| --- | --- | --- |
| Validity | closed text-contract evidence | Stage A quarantines only payload absence, declared text-contract violation, unrecoverable corruption, or acquisition failure; Stage B rejects only invalid chunk results |
| Redundancy | lossless exact digest, exact scaffold-family signature, and candidate-only near-duplicate evidence | Stage B exact-duplicate gate and Stage C stable representative retention for identical scaffold families; near-duplicate compaction is disabled in the frozen Normal/Hard protocols |
| Quality | explicit non-payload artifact evidence and payload-preserving structural compaction | Stage C removes only declared generated-and-non-editable artifacts, self-contained license/comment chunks, complete HTML shells with no visible lexical payload, or explicit cookie-control panels with no explanatory prose; never an intrinsic score, weighted priority score, source identity, or target fraction |
| Coverage | representative linkage, residual-payload preservation, and composition audit | Audits that a redundancy removal retains its linked representative and reports drift; it has no selection or removal authority and never enforces a target mix |

Content domain and language/script are separate audit axes. For example,
`Math 10%` and `Code 10%` belong to content-domain composition, while a
multilingual target belongs to language/script composition. The runtime reports
both distributions but does not enforce a target mix by dropping records.

The curation report records these distributions at Raw input, Stage A release,
Stage B pass, and Stage C curated output, along with token-share deltas from
Raw input. It therefore explains a shift such as `Code: 60% -> 70%` without
claiming that either percentage is intrinsically better.

Each report also emits `reason_code_impact_audit`: for every Stage-A quarantine,
Stage-B rejection, and Stage-C compaction reason, it records affected records,
chunks, and token-proxy cost. When one row has multiple Stage-A reasons, its
token cost is deliberately visible under each reason and reason totals are not
additive. `coverage_impact_audit` verifies that each exact-duplicate or
representative-family removal has a linked survivor in the curated pool,
checks that a record with no Stage-C survivor has an explicit non-payload or
representative explanation, and reports multi-rule interactions plus the
raw-to-curated composition delta. It has no metadata-stratum selection logic:
it can fail materialization only when one of these invariants is violated.

Stage C has no weighted operational threshold. It does not access Utility or
benchmarks, and every active policy has executable negative conditions that
bound when it may not trigger. Stage C selects and materializes its output; a fixed-fraction,
token-cap, or priority allocator remains outside the current framework surface.
The active Normal profile may inspect only chunk text. Source identity, source tier,
rights, path, language, composition labels, Utility, and benchmark outcomes
remain unavailable to its selector.
Language-specific parsing is diagnostic unless the input declares the relevant
language version and the corresponding rule declares executable negative
conditions.

Each policy is a bounded empirical hypothesis, not a claim that its removed
text is universally useless for every future model. A candidate can receive
runtime authority only after its closed trigger, false-positive/adversarial
fixtures, reason-code and coverage impact, Code/Math/General development
behavior, and benchmark-disjoint external evaluation are frozen together.
Failure at any gate preserves the original text and archives the candidate;
neither a model score nor an output-size objective may widen a rule.

Short snippets, partial files, tables, equations, JSON, HTML, Markdown,
multilingual text, and unfamiliar formats are not Validity failures. A parser
may produce candidate-only evidence only for a declared complete artifact with
a compatible declared language/version; it has no active deletion authority.
Collectors may supply provenance, rights, language, path, and artifact-context
metadata for traceability or an optional safety review. The adapter preserves
such declarations without inference, but the active v2 Stage-C selector does
not receive them. `scripts/build_rule_opportunity_audit.py` can still report
their availability as a diagnostic; neither a source label nor a path pattern
authorizes removal in the source-agnostic core.

`stage_c_declared_dependency_copy_candidate` remains non-runnable historical
research inventory. It is not part of the source-agnostic framework claim and
cannot be promoted into the active v2 selector merely because a collector
provides an artifact declaration.

The Normal Quality rules are intentionally narrower than a generic web-content
filter. `empty_html_shell` requires a complete HTML wrapper and no visible
lexical token after tags are removed. `web_chrome_only_chunk` requires at
least four nonblank lines and every line must be one of the fixed explicit
cookie-control markers. An HTML article, a script or style payload, and prose
about cookies or consent are retained. Placeholder and separable boilerplate
compaction remain candidate-only until their false-positive and external gates
close.

The Normal development gate runs frozen Code, Math, and General fixtures across
clean, exact-duplicate-heavy, explicit-artifact-heavy, and malformed scenarios.
It compares artifact rules off versus all active Normal Quality rules on, reports
reason-code and token-proxy deltas, and requires the Coverage invariant to
pass. It is local structural evidence only: it does not read a model, NLL,
Utility, or benchmark outcome.

## Normal And Hard Profiles

The active user-facing mode is **Normal**. `curation_mode: "normal"` resolves
to `normal_structural_v1`, which uses only the reason-coded text-structural
policies declared above. `curation_mode: "hard"` is available only with
`execution_scope: "development"`: `hard_structural_v1` runs Normal plus its
frozen span policies and writes a transformation JSONL plus residual-payload
audit. N4 fixture ablation has validated its deterministic behavior, but it
remains fail-closed for production curation until confirmatory external
evaluation closes. Once promoted, Hard will be a strict containment
profile (`Hard subset Normal`) made of Normal plus validated structural
policies; it will not use model-relative scores, source or domain metadata,
Quality scalars, or retention targets.

The frozen Hard v1 inventory is deliberately small: prefix-license header
spans, self-contained license comment-block spans, and long exact repeated
template spans. Each requires an explicit trigger, a useful non-trigger
fixture, a Stage-B-valid residual, a reason code/token delta, and a
representative or span trace. Near-duplicate threshold changes, model-relative
proxies, source metadata rules, and parser adapters are not in the initial Hard
surface.

The earlier Stage C2 proxy, Mid estimator, token-budget planner, and three-arm
materializer remain archived candidate research. They have no user-facing mode
or Stage C authority. Their files are retained for traceability and possible
future policy research, not as a path around the Hard promotion gate.

A frozen reference-distribution probe is also diagnostic-only. It may compare
candidate text with a source-declared reference pool for scope diagnostics, but
it emits neither a Stage C reason code nor a selection
decision. The current policy profile explicitly excludes its score from Stage C
until a separate policy card passes independent target-definition, executable
negative-condition, scope, and external-validation gates. Similarity to a reference
source is not an intrinsic Quality measurement and cannot authorize removal.

A `declared_generation=generated` label alone is likewise not an automatic
removal rule: generated code can be a useful, authored-for-distribution API or
client implementation. The active generated-artifact rule still requires
in-text generated-and-do-not-edit evidence, and the Core behavior audit keeps a
declared-generated negative fixture to prevent that boundary from drifting.

This boundary was replayed on a controlled `gfx-rs/wgpu` source corpus. Its
repository-declared `.gitattributes` labels 130 Stage-A release records as
`linguist-generated=true`; none of those 130 records matches the active
generated-and-do-not-edit condition. The replay therefore records zero
explicit-generated-artifact removals. This is evidence against promoting a
source-declared generated label into a label-only removal rule, not evidence
that generated code is intrinsically poor. The collector uses `git check-attr`
to preserve the repository declaration and leaves every undeclared file as
`unknown`.

The license-comment-only rule uses text only: every nonempty line must be a
comment line and the block must contain an explicit copyright, SPDX, or license
marker. It does not remove ordinary comments, documentation, code with a
license header, or numeric tables that merely have few line breaks. It is a
precision rule, not a claim of material compression.

`validation/core_behavior_audit_v3.py` executes labeled positive,
false-positive, metamorphic, and adversarial fixtures through the real A/B/C
runtime. It verifies more than reason-code presence: Validity must issue the
declared action, Redundancy removals must link to a surviving representative,
Quality removals must carry the typed deletion-authority trace, and Coverage
must detect missing representatives while remaining audit-only. It also fails
when Registry and Case Matrix Core ownership disagree. The resulting
TP/FP/FN/TN and invariant counts are a regression gate for constructed
boundaries, not a human Quality label or an estimate of corpus-wide precision,
recall, intrinsic text quality, or downstream effectiveness.
`validation/fixtures/policy_fixture_contract_v1.json` additionally binds every
active registry entry to positive, false-positive, and, where needed,
adversarial case IDs. A policy cannot remain active without an executable
non-trigger boundary; these fixtures are structural contracts, not human text
quality annotations.

When provided, provenance, rights, and PII context are stored in the optional
audit sidecar. They do not decide whether a text-only v2 candidate is selected.
PII context is an optional safety-review setting, not a selection score. For
example, the `technical_math` context retains high-confidence formatted or
contact-context phone detections while suppressing unformatted formula,
citation, and numerical-identifier sequences. Its use must be declared in the
frozen source contract and assessed through executable negative conditions; it cannot
be inferred from a composition label or changed after observing benchmark
results.

### Legacy Source Replay

A legacy collector output is not automatically eligible for the active runtime.
It must first be translated into the canonical input contract with a source
manifest that preserves upstream identity, collection time, and rights state.
The adapter cannot promote `unknown` rights to `allowed`.

The retained 5M-token mathematics raw pool from
`brando/small-open-web-math-dataset-v2` has a replay manifest at
`protocols/math_5m_legacy_source_manifest.json`. Its upstream card did not
declare reusable rights when inspected on 2026-07-27. The current replay is
therefore a Stage-A quarantine check, not a materialized training dataset. A
future replay requires an authoritative source/license declaration; it will
then use the same frozen A-B-C policy and no-budget behavior.

## Active Files

| Role | File |
| --- | --- |
| A-B-C materialization CLI | `run_curation.py` |
| Development-only aggressive five-arm ablation | `aggressive_structural_candidate_runner.py` |
| Composition audit | `composition_audit.py` |
| JSON and artifact utilities | `curation_artifacts.py` |
| Raw-record adapter | `ingestion/input_adapter.py` |
| Stage A candidate processing | `ingestion/candidate_processing.py` |
| Candidate release contract | `ingestion/candidate_contract.py` |
| Core behavior audit | `validation/core_behavior_audit_v3.py` |
| Core behavior executors | `validation/core_behavior_executors.py` |
| Core behavior contracts | `validation/core_behavior_contracts.py` |
| Labeled Core behavior fixtures | `validation/fixtures/core_behavior_audit_v3_cases.json` |
| Official Code benchmark snapshot builder | `scripts/build_code_benchmark_snapshots.py` |
| Retired-proxy forensic report builder | `scripts/build_historical_proxy_forensics.py` |
| Additional-rule evidence audit | `scripts/build_rule_opportunity_audit.py` |
| Git-attribute source collector | `scripts/collect_git_attribute_candidate_pool.py` |
| Hugging Face text source collector | `scripts/collect_huggingface_text_candidate_pool.py` |
| Legacy-source contract adapter | `scripts/adapt_legacy_candidate_pool.py` |
| Framework contract | `configs/curation_contract.json` |
| Materialization contract example | `configs/curation_run_contract.example.json` |
| Pretraining-eligible Code replay contract | `protocols/code_7m_pretraining_eligible_curation_v3_contract.json` |
| License-comment candidate replay contract | `protocols/code_7m_pretraining_eligible_curation_v4_license_comment_contract.json` |
| Frozen Code external protocol | `protocols/code_evaluation_protocol.json` |
| Active record/text-disjoint confirmatory protocol | `protocols/code_record_disjoint_confirmatory_evaluation_protocol.json` |
| Confirmatory training-input contract | `protocols/code_record_disjoint_confirmatory_training_materialization.json` |
| Confirmatory external preflight | `external_evaluation/preflight_record_disjoint_confirmatory.py` |

For the frozen v3 Code replay, the opportunity audit is stored at
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\proxy_removal_forensics\rule_opportunity_audit_v3.json`.
It found 7,997 curated chunks with path, content-type, and source-name metadata
preserved for all chunks, but no declared generation or dependency-copy context.
Its path/text-shape candidates are therefore explicitly blocked from becoming
selection rules without new source-backed metadata and executable negative
conditions.

The historical v1 Stage-A release audit over 4,887 chunks found zero rows with
artifact declarations and a small number of path/text-shape candidates. These
counts remain diagnostic only and do not alter the active v2 text-only policy.
Its report is
`D:\UNLV-Research\code_5m_corpus_v2\final_replay_v1\proxy_removal_forensics\core_rule_opportunity_audit_stage_a_v3.json`.

The controlled source-metadata stress contract is
`protocols/git_attribute_wgpu_stress_curation_contract.json`. Its curation
output and canonical metadata audit are stored under
`D:\UNLV-Research\source_metadata_stress\abc_curation_wgpu_git_attribute_v1`.

The preserved mathematics pool contains 4,292 legacy source rows and a
5,000,112 collection-time token proxy. Its historical current-contract replay
artifacts are under `D:\UNLV-Research\cross_domain_stress\abc_curation_math_5m_current_v1`.
The Normal text-only profile does not quarantine rows because rights are absent;
rights resolution belongs to the optional safety review, not A-B-C compression.

The source-declared-license OpenWebMath cross-domain replay uses
`protocols/openwebmath_5m_current_curation_contract.json` and its
`technical_math` context sensitivity replay uses
`protocols/openwebmath_5m_technical_math_curation_contract.json`. These are
controlled runtime artifacts, not a claim that a mathematics-oriented source
proves universal curation effectiveness.


## Commands

```powershell
conda run --no-capture-output -n research python run_curation.py --config C:\path\to\curation-run-contract.json
```

The command writes `curation_report.json` and the stage artifacts to the
configured output directory. Domain, source, and collection details remain
input metadata for audit; they do not control a runtime branch.

The aggressive candidate runner accepts a frozen Stage-B JSONL snapshot and a
frozen tokenizer through `configs/aggressive_structural_candidate_v1.example.json`.
It is not callable from `run_curation.py`, cannot change an active policy, and
must pass the preregistered promotion gates in
`configs/aggressive_structural_candidate_ablation_preregistration.json` before
any candidate rule receives runtime authority.
