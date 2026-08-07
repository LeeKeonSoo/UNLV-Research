# Framework Consistency Baseline

Status: current authority after retirement of the Contrastive Quality
candidate and registration of the three-teacher Quality Ranker candidate.

## Scientific Claim Boundary

The framework curates LM training corpora through auditable, typed decisions.
It does not claim to measure an intrinsic universal document Quality value.
It does not use downstream outcomes while selecting data. External training
and benchmarks validate a frozen curation policy after the framework emits its
dataset.

## Fixed Core and Stage Ownership

| Core | Stage | Question | Decision authority |
|---|---|---|---|
| Validity | A | Is the unit trainable under the declared input contract? | Closed failure quarantine or removal |
| Redundancy | B | Is equivalent training information already represented? | Stable-family nonrepresentative removal |
| Quality | B | Does an independently validated Policy support rejection or retention? | Promoted Policy decision only |
| Coverage | C | Did combined removals erase support without authorization? | Materialization veto only |

Coverage cannot rank data, impose a domain mix, delete data, or restore by
quota. Quality cannot use Utility, NLL, benchmark outcomes, source reputation,
domain quotas, or a forced token budget.

## Current Object Authority

The authority chain is:

1. `configs/curation_framework_v1.json`
2. `configs/framework_objects_v1.json`
3. `configs/framework_profiles_v1.json`
4. `configs/framework_runtime_bridge_v1.json`
5. `configs/framework_release_validation_v1.json`

Every file is hash-linked. The runtime bridge preserves legacy-compatible
output and prevents blocked Policies from becoming active merely because a
candidate file exists.

## Current Policy Status

| Policy | Lifecycle | Current meaning |
|---|---|---|
| `validity.interpretable_text` | candidate | Closed observable input failures only |
| `redundancy.exact_text_family` | development_passed | Exact family representatives validated in development |
| `redundancy.symmetric_near_duplicate_candidate` | candidate | Active runtime experiment; empirical promotion evidence is incomplete |
| `quality.explicit_nonpayload` | candidate | Closed deterministic non-payload cases |
| `quality.teacher_panel_v2` | candidate | Active runtime experiment; Q1-Q4 scientific qualification is incomplete |
| `coverage.representative_guard` | candidate | Semantic v3 explicit veto/rematerialization is implemented; empirical promotion gates remain open |

Normal and Hard share these Policy families. Their operating points are not
calibrated, and both profiles are release-disabled. `Hard subset-or-equal
Normal` is a mandatory materialized-output invariant.

The Redundancy implementation separates similarity retrieval from deletion
authority. Normal and Hard execute the same witness families: exact-text,
formatting, bounded near-substitute, exact token containment, and
token-preserving prose reflow. Hard uses broader frozen changed-token bounds
and may consume a versioned declared equivalence verifier.
Numeric, operator, negation, answer-label, code-identifier, and named-entity
differences retain by default. Every proposal records one stable family,
representative, witness kind, evidence hash, reason code, and token delta.
These are Stage-B proposals only: Stage C must apply its Coverage veto before
membership can change. Runtime activation does not imply promotion;
protected-fixture and development-disjoint evidence remain required.

## Semantic Coverage Candidate

Stage C now has an explicit candidate implementation rather than a silent
post-selection repair. It combines two independent frozen embedding views with
deterministic route, script, format, and Redundancy-family evidence. Stable
local support groups require reciprocal-neighbor evidence shared by both
providers. Remaining provider-specific neighborhoods become overlapping
uncertainty groups. Similarity defines support evidence but never authorizes
deletion; Stage C acts only when a support group would otherwise have no
survivor.

A Coverage veto emits typed `required_retain_uids`, rematerializes the candidate
corpus, and reruns the complete invariant set. The Redundancy-selected
directional representative has precedence; deterministic facility-location is
only a fallback. Normal and Hard share these exact Coverage invariants.

The primary candidate provider is Qwen3-Embedding-0.6B and the independent audit
provider is BGE-M3. Qwen is authorized only as a development/confirmatory
runtime experiment; BGE-M3 is audit-only. Scientific activation still requires
protected false-veto bounds and independent multilingual and multidomain
confirmatory evidence. The current blockwise exact-neighbor implementation is
adequate for the 8,024-chunk candidate run but is not a production-scale ANN
claim.
Raw-to-Curated route and language composition files are explanation artifacts
only and cannot enforce quotas or feed selection.

## Quality Ranker Candidate

The current candidate uses three independent model organizations and four
independent fail gates. It does not average dimensions into a weighted score.
Each teacher returns `pass`, `fail`, or `abstain` plus reason codes for Q1-Q4.
A 2-of-3 first pass requires the same majority and at least two stable teachers
on a blinded second pass. All other outcomes abstain.

The first hosted/local panel is frozen as a failed candidate after 143/512
behavior tasks. A historical all-hosted candidate (Mistral Medium 3.5,
Nemotron 3 Ultra, and DeepSeek V4 Pro) matched 64/64 development cells, but
seven unavailable provider traces were observed.

The active runtime-experiment panel is GLM-5.2, Nemotron 3 Ultra, and MiniMax
M3. GLM-5.2 accepted the production Q1-Q4 JSON request and returned a
schema-valid response, but endpoint probes observed 237.971 and 409.265 second
successes plus one 300.508 second timeout. Its transport timeout is therefore
600 seconds with one retry. This is endpoint and schema evidence only. The new
provider identity cannot inherit either historical panel's behavior or
false-removal evidence, so Quality promotion remains blocked.

The candidate cannot delete data by itself. Promotion requires:

- a fresh, development-disjoint 512-item behavior fixture matrix;
- at least 800 protected fixtures;
- one-sided 95% exact false-removal upper bounds no greater than 0.5% for
  Normal and 2.0% for Hard;
- stable consensus and schema behavior;
- a frozen provider identity and operating point;
- Coverage compatibility;
- independent external evaluation after curation.

The initial language scope is English. External NVIDIA teachers may receive
only public, license-compatible calibration samples. Private or undeclared
corpora remain local.

## Contrastive Retirement

The previous target/reference loss-gap design is retired as a current Quality
candidate. Its observed evidence did not support a stable route-general
deletion threshold. All implementation and evidence files are preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/` for
provenance, but no active manifest, profile, provider registry, runtime
fingerprint, or release gate references that candidate.

## Release State

`validation/frozen_contracts/framework_release_validation_v1.json` reports:

- implementation integrity: `passed`;
- framework release: `blocked`;
- blockers: unpromoted policies, uncalibrated profile operating points,
  candidate Near-duplicate and Quality policies, and unqualified semantic
  Coverage providers.

This is a fail-closed scientific state, not evidence that the active
legacy-compatible curation output is broken.

The final-test readiness states are deliberately separate:

| State | Meaning | Current state |
|---|---|---|
| Framework materialization | Frozen A-B-C code emits audited Normal/Hard datasets | Implementation ready; final all-policy run pending |
| External confirmatory | Exact tokenizer inputs exist for Raw/Normal/Hard | Pending final all-policy outputs |
| Paper claim | Promoted Policies plus confirmatory results support the stated claim | Blocked |
| Production release | All active Policies and providers are promoted | Blocked |

The previous pre-all-policy Qwen3-4B stream-token counts were 6,984,438 Raw,
6,961,249 Normal, and 6,747,888 Hard. They are historical diagnostics only and
must not be reused as the final Quality/Redundancy/Coverage result. The final
counts will be read only from `D:/UNLV-Research/final_all_policy_v1/` after both
profiles complete.

## Next Exit Gates

1. Execute the implemented 512 behavior tasks and 3,200 protected Policy tasks.
2. Freeze the qualification report and promote only modes that pass their exact
   one-sided false-removal bound.
3. Execute the witness-based Redundancy behavior and protected false-removal gates.
4. Complete multilingual provider-agreement, provider-bias, extinction-recall,
   protected false-veto, and corpus-scale ANN Coverage validation.
5. Run admitted corpus-scale Base, Normal, and Hard curation audits.
6. Run external three-seed natural-budget evaluation of frozen outputs.

Only after those gates pass may the release claim or paper claim be expanded.
