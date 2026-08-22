# Evidence-Bound LM Data Curation

This package turns JSONL text records into an auditable language-model training
corpus. The runtime ends at curated data and decision traces. Training loss,
Utility, benchmark results, source reputation, domain quotas, and fixed token
budgets never enter membership decisions.

## Runtime

The public entry point is `run_curation.py`. It exposes one budget-free mode,
`framework`. A qualified local Q1-Q4 `FAIL` removes the chunk. Otherwise, Q2,
Q3, and Q4 must all `PASS` to provide positive retention support; Q1 remains
a qualified-failure veto because correctness may be unverifiable in isolation.
`ABSTAIN`, out-of-distribution, low-confidence, and other unsupported local
outcomes are sent to the frozen GPT-5.6 Luna Batch fallback. A completed
fallback retains only the same positive-support pattern; a confirmed failure
or no positive support removes the chunk. Missing or invalid provider evidence
stops materialization after writing the required request artifact, so an API
failure is never interpreted as poor data. There is no hidden Normal/Hard
selector.

| Stage | Core | Runtime authority |
|---|---|---|
| A | Validity | Normalize losslessly and reject closed record/chunk failures. |
| B | Redundancy | Remove only linked nonrepresentatives and exact repeated sentence spans. |
| B | Quality | Retain conjunctive Q2+Q3+Q4 support; reject qualified failures and completed fallback cases without that support. |
| C | Coverage | Restore support needed to prevent unexplained representative or semantic-support extinction. |

The exact Core-Policy-Method inventory is in
`configs/runtime_policy_registry_v1.json`. The exact deployable file set is in
`configs/deployment_surface_v1.json`.

## Quick Start

1. Install the runtime dependencies.

```powershell
conda activate research
pip install -r requirements.txt
```

2. Copy `configs/curation_run_contract.example.json` to a run-specific path and
   set the input, output, model-cache, and frozen Quality-ranker manifest paths.

3. Run curation from this directory.

```powershell
python run_curation.py --config C:\path\to\run.json
```

If local evidence is unsupported, this command writes
`quality_teacher_requests.jsonl` and stops. Materialize those requests with the
frozen Luna Batch support workflow, set `teacher_observation_path` to the
completed hash-bound observation JSONL, and rerun the same config. The final
run does not make an online request and cannot silently continue with missing
evidence.

Automatic artifact mode embeds the Stage-B universe with the frozen Qwen3
primary provider and BGE-M3 audit provider, builds the semantic Coverage graph,
applies the frozen Quality ranker, and materializes Stage C. Precomputed
embedding and Coverage artifacts may be supplied instead, but automatic and
precomputed modes cannot be mixed in one run.

## Outputs

Every completed run writes:

- normalized input and Stage-A record quarantine JSONL;
- Stage-A pass/rejection and Stage-B pass/rejection JSONL;
- Stage-B proposed survivors and typed non-selection proposals;
- complete local Quality evidence and the exact Luna fallback request subset;
- Stage-C final curated and not-selected JSONL;
- Coverage restoration traces and reason-code impact audit;
- route, language/script, and morphology composition reports;
- hashes for every active policy config and imported runtime module.

Composition is descriptive only. It never enforces a target domain mix.

## Release Status

The software surface is packaged as a **beta research release**. Runtime
contracts and fixtures are machine-checked, but the current model providers and
operating points are not claimed to be universally production-validated. The
supported scientific claim remains bounded by the external Code and Math
experiments; those experiments are not part of this runtime package.

See `docs/deployment_runtime.md` for the detailed contract and verification
commands.
