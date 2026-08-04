# Quality Teacher Panel v1

## Status

This panel is a `candidate_qualification` artifact. It is not active in the
curation runtime, and no teacher response may delete a unit by itself.

## Frozen panel

| Slot | Model | Location | Purpose |
|---|---|---|---|
| Teacher A | `google/gemma-4-31b-it` | NVIDIA Build | Independent broad judgment over code, math, structured data, and prose |
| Teacher B | `meta/llama-3.1-8b-instruct` | NVIDIA Build | Independent model-family judgment and structured-output cross-check |
| Teacher C | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Local | Reproducible and private local judgment |

The local teacher uses bitsandbytes int8 inference on GPU 0, the RTX 4060 Ti.
The frozen revision was downloaded to `D:\hf_cache\hub` and all 16 Hub files
passed checksum verification. The common adapter loads the model in about 11.2
GiB of VRAM in the observed text-only smoke path.

## Quality policies

The panel evaluates four independent Stage-B policies. It does not emit an
overall Quality score.

| Policy | Decision question | Fail boundary |
|---|---|---|
| Q1 Correctness Evidence | Is correctness supported by local or attached verifier evidence under the declared context? | Only reproducible contradiction, impossible derivation, failed declared verifier, or locally checkable incorrect result |
| Q2 Semantic Coherence | Do the parts form a consistent and recoverable semantic unit? | Only incompatible fragments, broken dependencies, or internal contradiction that prevents coherent interpretation |
| Q3 Substantive Payload | Does substantive content remain after observable navigation, metadata, boilerplate, and empty templates are excluded? | Only when no substantive residual payload remains |
| Q4 Learnable Relations | Is at least one relation recoverable among entities, operations, claims, conditions, or outcomes? | Only an unconnected token, label, or fragment set with no recoverable relation |

Each policy returns `pass`, `fail`, or `abstain`. Missing external knowledge,
undeclared execution assumptions, uncertain specialized notation, and possible
missing context must produce `abstain`.

Q1 first consumes a typed declared-verifier result when one is present. The
verifier identity, binary status, and evidence SHA-256 are required. A declared
`pass` or `fail` is authoritative and bypasses all teacher generation; teachers
evaluate Q1 only when no declared verifier is attached. Q2-Q4 always follow the
panel path. A verifier result is evidence for Q1 only and cannot bypass another
Policy.

## Response and consensus contract

Each teacher must return one JSON object with a decision enum and a non-empty
reason-code list drawn from the closed vocabulary for that Policy and decision.
Lower-snake-case syntax alone is insufficient. One schema-only retry is
allowed. A second invalid response or transport unavailability becomes an
audited `abstain` and cannot contribute a pass or fail label.

First-pass unanimity is accepted. A 2-of-3 result triggers a blinded second
pass using the same teachers. It is accepted only when the same decision and at
least two of the same teachers remain stable. All other outcomes abstain.

## Qualification and promotion

The 512-item controlled fixture matrix contains four Policies, four routes,
four fixture classes, and eight samples per cell. Labels come from deterministic
constructions and attached local verifiers rather than subjective document
ratings. Exact behavior on all 512 tasks is required.

Normal activation requires at least 800 protected fixtures and a one-sided 95%
exact false-removal upper bound no greater than 0.5%. Hard uses the same frozen
protected evaluation with a 2.0% upper bound. Even zero observed errors in only
512 samples leaves the Normal upper bound above 0.5%, so the smoke suite cannot
activate runtime policy.

Fixture labels must come from controlled transformations and attached
verifiers where possible, rather than subjective document-quality annotation.
The final protected set remains disjoint from teacher prompt development and
student-ranker training.

Normal and Hard use the same four Policies. Normal removes only a first-pass
3-of-3 Policy FAIL. Hard additionally admits a 2-of-3 FAIL only when the same
decision and at least two of the same teachers survive a blinded second pass.
An abstention always retains. Stage C may veto either mode to preserve Coverage.

## Data and runtime boundary

Only public, license-compatible calibration samples may be sent to NVIDIA
Build. The initial language scope is English. Benchmark outcomes, NLL, Utility,
source reputation, domain quota, target retention, maximum token budget, and
confirmatory data are forbidden teacher and runtime inputs.

The current hosted/local adapter completed a public Q3 smoke run on 2026-08-04.
Gemma, Llama, and local Qwen all returned first-pass `pass` with the closed
`substantive_payload_present` reason code. Observed generation latency was
5.67, 0.48, and 20.16 seconds respectively. This proves current connectivity,
dispatch, schema, and consensus behavior, not teacher qualification. The
hosted endpoint is
`https://integrate.api.nvidia.com/v1`; model IDs and raw-response hashes are
recorded because a hosted endpoint does not expose immutable weight artifacts
like the local Hugging Face revision.

The first eight Q1 behavior fixtures also completed under the v2 observation
contract: all eight used declared-verifier precedence, all passed, and no model
generation occurred. Precedence-free v1 diagnostic observations are excluded,
and the resumable runner rejects any observation schema other than v2.
