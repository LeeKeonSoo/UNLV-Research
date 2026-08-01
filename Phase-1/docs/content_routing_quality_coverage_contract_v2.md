# Content Routing, Quality, And Coverage Contract V2

## Status

This is the frozen successor design to
`configs/positive_quality_coverage_contract_v1.json`. It does not activate a
new runtime policy and does not change any historical curation result. The
active boundary remains `configs/curation_contract.json` until every v2
activation gate passes and the contract, runtime, tests, and documentation are
switched together.

## Responsibility Boundary

The four Cores remain **Validity, Redundancy, Coverage, and Quality**. Content
routing is a shared method, not a fifth Core. It emits observable metadata that
Quality and Coverage consume for different purposes.

| Component | Owns | Must not do |
| --- | --- | --- |
| Content Router | Multi-label format, structural-state, language/script, and semantic-domain evidence | Select, remove, rank, assign importance, or infer a training objective |
| Validity | Integrity, structural interpretability, reversible repair, rechunking, quarantine, and unrecoverable rejection | Judge usefulness or intrinsic Quality |
| Redundancy | Repeated-payload families and deterministic representative linkage | Treat domain membership as duplication |
| Quality | Route-conditioned positive retention eligibility under calibrated evidence | Use a route label alone, a global weighted score, Utility, a benchmark, or a target fraction |
| Coverage | Raw-to-Curated representation and loss audit over router labels | Set a quota, alter a threshold, select a record, or rescue a Quality reject |

This resolves the earlier ambiguous wording that Coverage performs domain
classification and therefore controls Quality. Coverage does not control
Quality. The same frozen router output is supplied independently to both.

## Router Contract

Routing is multi-label. A chunk can be mathematical prose, technical
documentation, and English at the same time. Mixed content is routed at the
smallest stable chunk or span that preserves context; it is not forced into one
exclusive domain.

The four ordered axes are:

1. `content_format`: prose, source code, mathematical notation, table,
   dialogue, question-answer, instruction, log, markup, mixed, or unknown.
2. `structural_state`: complete document/artifact, snippet, template,
   generated artifact, partial, mixed, or unknown.
3. `language_script`: observable scripts and multilingual mixtures.
4. `semantic_domain`: Code, Math, Science, Medicine, Law, Finance/Economics,
   History/Humanities, General Knowledge, mixed, or unknown.

Quality routing prioritizes format and structural state. Language/script is a
compatibility condition for some validators. Semantic domain may select an
additional registered validator, but never authorizes retention or removal by
itself. Source name, dataset identity, path, source reputation, a human Quality
label, Utility, NLL, benchmark outcomes, quotas, and target retention fractions
are forbidden router inputs.

Every routing result records labels, status, calibrated confidence, evidence
codes, and router version. Routing confidence is not a Quality score.

## Route-Conditioned Quality

Quality is positive retention eligibility, not universal document goodness.
Each registered route first requires a routing precondition and then two
independent Quality evidence heads:

1. `route_confidence`: routing precondition only; it is not Quality evidence and
   cannot authorize retention or removal.
2. `substantive_payload`: the unit contains payload rather than only a closed
   non-payload artifact.
3. `route_specific_evidence`: evidence validated for the relevant content
   format and route.

The two Quality heads are conjunctive and cannot be added into a weighted
global score. A route label is only a dispatch condition and cannot substitute
for either head.
Topic labels are weaker than format and structural-state evidence.

Closed domain-independent non-payload policies may operate on every route.
Domain-specific policies require a registered scope, a positive and negative
fixture boundary, source- and dataset-disjoint calibration, an OOD behavior,
and a frozen threshold. When a unit is unknown, mixed, OOD, or lacks complete
evidence, only common closed rules may run; the remaining decision is
`ABSTAIN_RETAIN`.

`route_quality_evidence_candidates.py` materializes the frozen candidate
availability registry without granting runtime authority. Blocked evidence
emits `indeterminate`; unsupported evidence emits `missing`. Only a future
artifact that passes integrity, disjoint calibration, transfer, and stress gates
may emit `pass` or a separately named `negative` boundary.

| Route | Substantive payload | Route-specific evidence | Current result |
| --- | --- | --- | --- |
| General prose | Missing | Indeterminate | `ABSTAIN_RETAIN` |
| Code artifact | Missing | Indeterminate | `ABSTAIN_RETAIN` |
| Mathematical content | Indeterminate | Indeterminate | `ABSTAIN_RETAIN` |
| Technical documentation | Missing | Missing | `ABSTAIN_RETAIN` |
| Conversation | Missing | Missing | `ABSTAIN_RETAIN` |
| Instruction | Missing | Missing | `ABSTAIN_RETAIN` |
| Table/structured data | Missing | Missing | `ABSTAIN_RETAIN` |

Normal retains abstentions. Hard may become stricter only for a separately
calibrated route. Hard does not turn unknown or mixed content into automatic
deletions.

## Validity Correction

Structural coherence and integrity belong to Validity. Forbidden control
characters, unambiguous encoding corruption, unclosed registered structures,
dangling fences, and trailing delimiter corruption are not evidence that an
otherwise useful document has low Quality.

The action order is:

```text
reversible repair -> context-preserving rechunk -> quarantine -> reject
```

The original text and transformation trace are preserved for repair and
rechunk actions. Rejection is allowed only when no interpretable payload
remains after declared recovery options are exhausted. The existing explicit
structural coherence artifacts remain immutable development evidence, but the
policy is reassigned to candidate Validity and is removed from the prospective
Quality evidence bundle.

## Coverage Contract

Coverage consumes the same router labels and reports document/token shares,
stratum retention, semantic-cluster survival, tail loss, unknown/mixed/OOD
rates, and Redundancy representative survival. It can block a release when the
audit or representative linkage is missing. It cannot change curated
membership to make the report look balanced.

In particular, Coverage does not enforce statements such as "Code must be 10%"
or "rare content must always survive." Raw-to-Curated changes are reported for
transparency and failure analysis only.

## Prospective Stage Order

```text
Stage A: normalization, reversible repair, source-record integrity
Stage B: chunk Validity hard gate and exact Redundancy gate
Stage C: shared routing
         -> route-conditioned Quality
         -> non-exact Redundancy representative resolution
         -> Coverage pre/post loss audit
         -> reason-coded materialization
External evaluation: frozen natural-budget validation outside runtime
```

No runtime policy may read benchmark results or Utility. External evaluation
can validate a frozen profile, but cannot retune it in the same confirmatory
cycle.

## Frozen Route Evidence Gate

`configs/quality_route_evidence_gate_v2.json` normalizes the existing General,
Code, and Math development artifacts under the current routing-precondition-
plus-two-head Quality contract.
It requires frozen artifacts, stable-ID and normalized-text-hash disjointness,
strict source transfer, adversarial and format fixtures, provider-bias stress,
route-holdout stress, and hidden external results. Passing this gate creates
only an evidence-ready candidate; it never grants runtime authority.

| Route | Decision | Blocking evidence |
| --- | --- | --- |
| General prose | `blocked_source_transfer` | On 1,200 source-balanced controls, Regulations produced 266/400 q0 leave-one-source-out failures (`70.26%` Wilson upper bound); no Normal or Hard profile passed. |
| Code artifact | `blocked_source_transfer` | The safest Stack-Edu boundary exceeds the Hard leave-one-repository-out bound and separates only one candidate record. |
| Mathematical content | `blocked_source_transfer` | Held-out sources still fail substantive-payload or route-specific heads after structural integrity was reassigned to Validity. |

No former four-head result is silently reinterpreted as a v2 pass. Unsupported
routes and every blocked route remain `ABSTAIN_RETAIN`.

The General route was additionally tested with two independent scalar
providers on the same controls and stress pairs. DCLM fastText failed
Regulations LOSO at 17/400 and semantic destruction at 181/450; FineWeb-Edu
failed at 18/400 and 139/450. Their HTML and Markdown decisions were stable,
but both measure only a route-specific preference and both fail the frozen
behavioral gates. `configs/general_provider_candidate_decision_v2.json`
therefore rejects both and explicitly forbids ensemble activation.

## Implementation Gates

Current status: the four-axis Router and its closed fixture matrix are
implemented at candidate/audit scope. Quality and Coverage consumer integration
remains intentionally disabled until their later gates pass. The routing
precondition and two-head Quality decision gate are fixture-validated at
candidate scope, but no route has
a promoted v2 evidence bundle. The Validity
recovery decision layer is also fixture-validated at candidate scope; its
payload-preserving rechunk materializer and active-runtime migration remain
closed activation gates.

1. **Complete:** implement and fixture-test all four router axes, including
   mixed, unknown, and OOD cases.
2. **Candidate decision complete:** separate Validity outcomes into repair,
   rechunk, quarantine, and reject with original-text and transformation
   traces. Rechunk materialization remains pending.
3. **Candidate evidence audit complete:** the routing precondition, two-head
   Quality decision, and common
   route-evidence gate are implemented. General, Code, and Math are all frozen
   as blocked; unsupported routes abstain.
4. Prove that route labels cannot independently cause removal and Coverage
   output cannot leak into selection.
5. Run source- and dataset-disjoint route calibration and Normal/Hard
   development ablations.
6. Freeze one candidate before three-seed natural-budget external evaluation.
7. Activate only by an atomic contract, runtime, test, and documentation
   switch.
