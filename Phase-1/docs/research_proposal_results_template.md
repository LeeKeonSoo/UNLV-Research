# Code-Domain External Evaluation: Results Template

**Status:** Pending completion of curated seeds 23 and 37, then external evaluation.

## Experiment Design

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Base |
| Adaptation | QLoRA continued pretraining (not SFT or DPO) |
| Arms | Base (no update), Raw-safe natural, Curated natural |
| Seeds | 11, 23, 37 for Raw-safe and Curated |
| Comparison | Natural dataset budget per arm |
| Raw-safe effective training tokens | 7,028,736 |
| Curated effective training tokens | 3,637,248 |
| Curated token reduction vs. Raw-safe | 48.25% |

The curation runtime does not read benchmark outcomes, NLL, or utility. The values below are external validation only.

## Final Results

Fill Raw-safe and Curated cells as `mean +/- standard deviation` over seeds 11, 23, and 37. Base is a single no-update checkpoint and therefore has no seed variance.

| Metric | Base | Raw-safe natural (3 seeds) | Curated natural (3 seeds) | Curated vs. Raw-safe |
|---|---:|---:|---:|---:|
| Effective training tokens | 0 | 7,028,736 | 3,637,248 | -48.25% |
| HumanEval+ pass@1 | Pending | Pending | Pending | Pending |
| MBPP+ pass@1 | Pending | Pending | Pending | Pending |
| LiveCodeBench v6 code_generation_lite pass@1 | Pending | Pending | Pending | Pending |
| BigCodeBench Complete pass@1 | Pending | Pending | Pending | Pending |
| DS-1000 pass@1 | Pending | Pending | Pending | Pending |
| OJBench pass@1 | Pending | Pending | Pending | Pending |

## Interpretation Record

| Question | Rule for the final wording | Result |
|---|---|---|
| Data efficiency | Does Curated match or exceed Raw-safe on a benchmark with fewer natural training tokens? | Pending |
| Seed consistency | Is the Raw-safe vs. Curated direction stable across the three seeds? | Pending |
| Scope | On which benchmarks are improvements, preservation, or degradation observed? | Pending |
| Claim boundary | State only the benchmark-specific evidence supported by the table and the reason-coded curation audit. | Pending |

## Reporting Constraints

- Report benchmark results separately; do not average scores across suites.
- Primary benchmark set: HumanEval+, MBPP+, LiveCodeBench v6 code_generation_lite, BigCodeBench Complete, DS-1000, and OJBench.
- SWE-bench Lite is a secondary agent-scaffold case study, not part of the primary model-only result table.
- Do not state that curation improves or preserves performance until all planned evaluations have completed.
