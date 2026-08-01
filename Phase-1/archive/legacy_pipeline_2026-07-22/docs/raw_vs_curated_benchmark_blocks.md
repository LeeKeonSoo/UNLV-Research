# Raw-vs-Curated Benchmark Blocks

## Goal

Prove the framework through Stage-C model evidence:

Curated equal-token fine-tuning should outperform raw-random and Stage-A-random
equal-token fine-tuning on domain benchmarks, while benchmark outcomes remain
forbidden from Stage-B selection.

## Block 1: Claim and Benchmark Contract

Status: completed.

Completed:

- Frozen protocol: `configs/paper_multidomain_benchmark_protocol_v1.json`
- Contract test: `validation/test_paper_multidomain_benchmark_protocol.py`
- Benchmark scope: Stage C only
- Training arms: base, raw random, Stage-A random, curated, high-quality reference
- Reporting requirement: pre/post curation record, chunk, and token counts

## Block 2: Dataset Composition and Readiness

Status: code-domain ready, non-code domains pending.

Completed for code:

- Raw-random payload exists.
- Stage-A-random payload exists.
- Curated Stage-B payload exists.
- Known high-quality reference payload exists.
- Packed token blocks exist for all four trainable arms.
- Equal packed token count is frozen at 325,632 tokens per arm.
- Current code-domain Stage-C NLL evidence is available.

Pending:

- Freeze code benchmark execution commands for EvalPlus/HumanEval+/MBPP+.
- Decide whether SWE-bench Lite runs locally or through cloud execution.
- Acquire raw mixed math corpus.
- Acquire high-quality math reference corpus.
- Materialize math equal-token arms before GSM8K/MATH evaluation.

Readiness report:

- `outputs/validation/raw_vs_curated_benchmark_readiness_report.json`
- `outputs/validation/raw_vs_curated_benchmark_readiness_report.md`

## Block 3: Code Benchmark Execution

Status: EvalPlus completed; SWE-bench capstone pending.

Decision:

- Run EvalPlus/HumanEval+/MBPP+ first as the lightweight code-generation
  benchmark family.
- Keep SWE-bench Lite or Verified as a capstone benchmark only after a
  feasibility gate passes.
- Treat all benchmark outcomes as Stage-C evidence only; they remain forbidden
  from Stage-B selector objectives.

Completed evidence:

- Block 3 freeze: `configs/code_domain_block3_benchmark_execution_freeze_v1.json`
- EvalPlus guardrail report:
  `outputs/validation/code_domain_v2_evalplus_confirmatory_guardrail_report.json`
- Block 3 benchmark report:
  `outputs/validation/code_domain_block3_benchmark_report.json`
- Curated v2 beats Stage-A random on EvalPlus macro pass-rate by more than the
  frozen 0.01 absolute margin.

Frozen execution contract:

- Use the frozen command templates in
  `configs/code_domain_block3_benchmark_execution_freeze_v1.json`.
- Use the frozen confirmatory arms and seed set from
  `configs/code_domain_v2_confirmatory_protocol_qwen3_4b.json`.
- Applied frozen EvalPlus practical margin: curated must beat Stage-A random
  by at least 0.01 absolute macro pass-rate or the result is inconclusive.
- Benchmark output remains forbidden from Stage B.

Recommended order:

1. EvalPlus/HumanEval+/MBPP+ lightweight benchmark.
2. Held-out code NLL rerun only if payloads change.
3. SWE-bench Lite feasibility check.
4. SWE-bench Lite confirmatory run if feasible.

## Block 4: Math Domain Acquisition

Status: completed; Stage-C protocol frozen before math outcomes.

Completed:

- Raw mixed math pool with noisy/problem/solution/explanation records.
- Known high-quality math reference pool.
- Frozen acquisition contract:
  `configs/math_domain_block4_acquisition_freeze_v1.json`
- Acquisition report:
  `outputs/validation/math_domain_block4_acquisition_report.json`
- Stage 0/A/B materialization.
- Equal-token payloads for raw, Stage-A random, curated, and reference arms.
- Equal-token materialization report:
  `outputs/validation/math_domain_equal_token_arms_report.json`
- Frozen Stage-C protocol:
  `configs/math_domain_stage_c_protocol_qwen3_4b_v1.json`
- Held-out math NLL slice:
  `outputs/math_domain_stage_c_qwen3_4b/heldouts/math_nll_heldout.jsonl`

Materialized arms:

- `raw_random_equal_budget`
- `stageA_random_equal_budget`
- `curated_math_equal_budget`
- `known_high_quality_equal_budget`

Pending:

- Run equal-token math fine-tuning.
- Run held-out math NLL.
- Implement or run the frozen GSM8K/MATH benchmark evaluator.

Stage-C targets:

- Held-out math NLL.
- GSM8K accuracy.
- MATH accuracy.

## Block 5: Fine-Tuning and Benchmark Comparison

Status: pending after Blocks 3 and 4.

Required:

- Same base model.
- Same token budget.
- Same seeds.
- Same training recipe.
- Same benchmark splits.
- No post-hoc selector tuning.

## Block 6: Paper Integration

Status: partially complete.

Completed:

- Novelty reframed as disposition, budget allocation, and downstream validation separation.
- Results now explain ablation mechanism.
- Next evidence tier is documented without claiming incomplete benchmark results.

Pending:

- Insert actual benchmark result figures.
- Add dataset curation funnel figure.
- Replace planned benchmark text with completed benchmark evidence when available.
