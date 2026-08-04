# Quality Teacher Panel v1

## Status

This panel is a `candidate_qualification` artifact. It is not active in the
curation runtime, and no teacher response may delete a unit by itself.

## Frozen panel

| Slot | Model | Location | Purpose |
|---|---|---|---|
| Teacher A | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA Build | Broad reasoning over code, math, technical text, and general prose |
| Teacher B | `z-ai/glm-5.2` | NVIDIA Build | Independent model-family judgment and structured-output cross-check |
| Teacher C | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Local | Reproducible and private local judgment |

The local teacher uses bitsandbytes int8 inference on GPU 0, the RTX 4060 Ti.
The frozen revision was downloaded to `D:\hf_cache\hub` and all 16 Hub files
passed checksum verification. An observed text-only smoke test used 10.76 GiB
maximum allocated VRAM and returned the required enum JSON after the schema was
made explicit.

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

## Response and consensus contract

Each teacher must return one JSON object with a decision enum and non-empty
reason-code list. One schema-only retry is allowed. A second invalid response
becomes `abstain` and cannot contribute a pass or fail label.

First-pass unanimity is accepted. A 2-of-3 result triggers a blinded second
pass using the same teachers. It is accepted only when the same decision and at
least two of the same teachers remain stable. All other outcomes abstain.

## Qualification and promotion

The 512-item controlled fixture matrix contains four policies, four routes,
four fixture classes, and eight samples per cell. It is a smoke qualification
suite, not activation evidence.

Normal activation requires at least 800 protected fixtures and a one-sided 95%
exact false-removal upper bound no greater than 0.5%. Hard uses the same frozen
protected evaluation with a 2.0% upper bound. Even zero observed errors in only
512 samples leaves the Normal upper bound above 0.5%, so the smoke suite cannot
activate runtime policy.

Fixture labels must come from controlled transformations and attached
verifiers where possible, rather than subjective document-quality annotation.
The final protected set remains disjoint from teacher prompt development and
student-ranker training.

## Data and runtime boundary

Only public, license-compatible calibration samples may be sent to NVIDIA
Build. The initial language scope is English. Benchmark outcomes, NLL, Utility,
source reputation, domain quota, target retention, maximum token budget, and
confirmatory data are forbidden teacher and runtime inputs.

The NVIDIA API key was detected and both hosted endpoints completed a public
smoke request on 2026-08-04. GLM-5.2 returned JSON inside a Markdown fence, and
Nemotron returned an out-of-contract decision enum on the first prompt. This
confirms connectivity, not schema qualification. The production adapter must
issue one schema-only retry and convert a second invalid response to
`abstain`. The hosted endpoint is `https://integrate.api.nvidia.com/v1`; model
IDs and raw response hashes must be frozen because a hosted endpoint does not
expose immutable weight artifacts like the local Hugging Face revision.
