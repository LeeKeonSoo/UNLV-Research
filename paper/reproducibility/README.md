# Evidence-Bound Curation: Manifest Artifact

This package contains the small, non-payload artifacts used to verify the paper's frozen Code-domain experiment. It is an evidence and replay manifest, not a redistributable copy of the training corpus or trained adapters.

## Contents

- `confirmatory_protocol.json`: frozen natural-budget training protocol and Qwen3-4B-Base snapshot identity.
- `materialization_protocol.json`: tokenizer, EOS, packing, and arm-source contract.
- `analysis_amendment.json`: timestamped primary/secondary benchmark hierarchy amendment.
- `normal_curation_config.json`, `hard_curation_config.json`: final runtime profiles.
- `normal_curation_report.json`, `hard_curation_report.json`: membership counts, reason codes, hashes, and invariant outcomes.
- `normal_coverage_audit.json`, `hard_coverage_audit.json`: Stage-C required-retain evidence.
- `training_inputs_report.json`: exact stream tokens, packed tokens, blocks, steps, and hashes.
- `runs/*.json`: the nine completed QLoRA run manifests.
- `benchmark_results.json`: complete Base/Raw/Normal/Hard benchmark matrix.
- `benchmark_provenance_audit.json`: recomputation evidence for all 60 model-benchmark cells and 42,820 task judgments.
- `source/`: the training, generation, platform-compatibility, collection, and provenance-audit entry points used for the reported matrix.
- `SHA256SUMS.txt`: package-local SHA-256 identities.

## Frozen Execution Contract

- Model/tokenizer: `Qwen/Qwen3-4B-Base`, snapshot `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Natural-budget arms: one complete packed pass for Raw, Normal, and Hard; no equal-token resampling or target retention budget.
- Training seeds: `101`, `202`, `303`.
- Optimizer: AdamW with `lr=5e-5`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.1`; no scheduler or warmup.
- QLoRA: rank 32, alpha 64, dropout 0.05, all linear targets; 4-bit NF4 with double quantization and bfloat16 compute.
- Generation: greedy pass@1 with EOS stopping; maximum 512 new tokens for EvalPlus, 1,024 for BigCodeBench/DS-1000, and 256 for CRUXEval.
- Scorers: EvalPlus 0.3.1; BigCodeBench v0.1.4 Complete public evaluator; CRUXEval commit `190faf16d175b5847b0af05d937872b1fb395942`; DS-1000 commit `b39aab71da6d23ef8d3cac59a7c5f834516ab334`.

## Verification Order

1. Verify every file against `SHA256SUMS.txt`.
2. Confirm the protocol hash and packed-input hash in each `runs/*.json` file.
3. Recompute arm summaries from `benchmark_results.json`.
4. Confirm `expected_cells=verified_cells=60` and `all_verified=true` in `benchmark_provenance_audit.json`.
5. Inspect the curation and Coverage reports for survivor links, required-retain IDs, final recheck status, and the Hard-subset-of-Normal invariant.

## Boundaries

The raw corpus, curated payload JSONL files, model snapshot, adapters, and task-level generated programs are omitted because of size and redistribution constraints. The manifest can verify the reported identities and outcomes but cannot alone rerun training. BigCodeBench was scored through its public evaluator; that remote service was not commit-pinned, so scorer outputs and task-level result hashes are retained instead. Public corpus release additionally requires the record-level rights-metadata repair disclosed in the paper.
