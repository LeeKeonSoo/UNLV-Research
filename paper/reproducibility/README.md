# Evidence-Bound Curation: Reproducibility Manifest

This package contains the compact, non-payload artifacts for the final frozen
Code and Math experiments. It verifies identities, curation transitions, and
reported outcomes; it does not redistribute corpora, model weights, adapters,
or the full generated-response payloads.

## Final Manuscript Scope

- Model/tokenizer: `Qwen/Qwen3-4B-Base`, revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Arms: `Base`, `Raw`, `Ours`, `Data-Juicer`, `NeMo Curator`.
- Trained-arm seeds: `101`, `202`, `303` in both Code and Math.
- Training budget: one complete pass over each arm's naturally retained packed
  tokens; no equal-token resampling and no target retention budget.
- The final framework policy is packaged as `beta_release`. Its Code and Math
  Coverage materializations pass the complete recheck but do not claim
  production or scientific promotion.

## Canonical Artifacts

- `final_code_curation_report.json`, `final_math_curation_report.json`: final
  runtime membership counts, reason codes, profile status, and hashes.
- `final_code_coverage_audit.json`, `final_math_coverage_audit.json`: final
  Coverage restoration traces and complete recheck outcomes.
- `final_code_training_inputs_report.json`,
  `final_math_training_inputs_report.json`: exact tokenizer stream, packed
  tokens, dropped tails, blocks, optimizer steps, and input hashes.
- `framework_code_results.json`: final Ours Code seed-level scores and source
  hashes.
- `framework_math_results.json`: final Ours Math seed-level scores and source
  hashes.
- `bigcodebench_common_cache_results.json`: authoritative BigCodeBench values
  for every displayed arm and seed using one common evaluator and ground-truth
  cache.
- `same_corpus_baseline_results.json`: Data-Juicer token counts and aggregate
  Code/Math results. Its matched-random section is a retained historical
  diagnostic and is excluded from the final manuscript.
- `nemo_curator_baseline_results.json`: three-seed Code NeMo result matrix.
- `nemo_curator_math_baseline_results.json`: three-seed Math NeMo result matrix,
  including seed 303 and its source hashes.
- `nemo_curator_math_seed303_extension_protocol.json` and
  `nemo_curator_math_seed303_evaluation_protocol.json`: frozen protocol records
  for the third NeMo Math seed added without changing membership or settings.
- `seed_robustness_summary.json`: final-arm cross-task mean seed variability.
- `execution_footprint.json`: historical Normal/Hard development-run workload;
  it is not the final manuscript count authority.
- `protocols_final/`: frozen Ours, Data-Juicer, and NeMo Curator curation,
  materialization, training, and evaluation protocols used by the final result
  surface.
- `source/`: evaluator, collection, training, and audit entry points.
- `SHA256SUMS.txt`: package-local file identities.

## Final Curation Counts

| Domain | Raw packed tokens | Ours packed tokens | Retention | Stage-A chunks | Final chunks | Coverage restores |
|---|---:|---:|---:|---:|---:|---:|
| Code | 6,979,584 | 6,242,304 | 89.44% | 8,026 | 7,270 | 104 |
| Math | 6,979,584 | 5,767,168 | 82.63% | 6,619 | 5,528 | 143 |

Both final Coverage audits have `complete_recheck_passed=true`,
`rematerialization_applied=true`, `may_create_new_removal=false`, and
`scientific_promotion_claimed=false`.

## Evaluation Contract

- Optimizer: AdamW, `lr=5e-5`, `betas=(0.9, 0.999)`, `eps=1e-8`,
  `weight_decay=0.1`; no scheduler or warmup.
- QLoRA: rank 32, alpha 64, dropout 0.05, all linear targets; 4-bit NF4 with
  double quantization and bfloat16 compute.
- Generation: greedy pass@1 with EOS stopping; 512 maximum new tokens for
  EvalPlus, 1,024 for BigCodeBench/DS-1000, and 256 for CRUXEval.
- Code scorers: EvalPlus 0.3.1; BigCodeBench Complete dataset v0.1.4;
  CRUXEval commit `190faf16d175b5847b0af05d937872b1fb395942`;
  DS-1000 commit `b39aab71da6d23ef8d3cac59a7c5f834516ab334`.
- Math scorer: lm-evaluation-harness 0.4.12 on frozen GSM8K and MATH-500
  snapshots. GSM8K normalized is a post-hoc common-parser sensitivity audit.
- BigCodeBench uses source commit
  `8653fc7f5e4f8c268d84e24d48f3548648a267d0`, dataset hash
  `acf4f1debe64a10c1a0bf6d4906245b2`, and cache SHA-256
  `46b1b99945e5754c533d4b828252e0ce473ddf3afc2e1271bf13e5da61c12766`
  for every arm and seed.

## Verification Order

1. Verify packaged files against `SHA256SUMS.txt`.
2. Check the final Code/Math curation reports and Coverage audits.
3. Recompute Ours means and sample SDs from `framework_code_results.json` and
   `framework_math_results.json`.
4. Use only `bigcodebench_common_cache_results.json` for the manuscript's
   BigCodeBench row.
5. Recompute Data-Juicer and NeMo summaries from their canonical result files.
6. Confirm that every trained arm has seeds 101, 202, and 303.
7. Cross-check the manuscript table against the four canonical result sources.

## Historical Artifacts

`benchmark_results.json`, `math_transfer_results.json`, the Normal/Hard reports,
and matched-random outputs document earlier development experiments.
`benchmark_results.json` is explicitly marked `superseded_for_final_manuscript`.
These files must not override the final Ours arm or common-cache BigCodeBench
scores.

## Boundary

The manifest can verify reported identities and summary outcomes but cannot by
itself rerun training because large payloads and adapters are omitted. Public
corpus release also requires the record-level rights-metadata repair disclosed
in the manuscript.
