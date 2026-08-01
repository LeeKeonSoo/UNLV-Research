# Code External Evaluation Protocol

## Purpose

This protocol evaluates frozen Raw and Curated Code training arms. It is not
part of the curation runtime and cannot alter a curation output.

## Fixed Arms

For each seed `101`, `202`, and `303`, evaluate the base model, the model
continued on the Raw Code corpus, and the model continued on the Curated Code
corpus. Each arm uses its natural dataset budget. Report the raw and curated
token totals and the curation composition shift beside every benchmark table.

## Primary Matrix

| Family | Benchmark | Why it is included |
| --- | --- | --- |
| Function correctness | HumanEval+ | Augmented unit tests for function synthesis |
| Function correctness | MBPP+ | A broader set of entry-level function tasks |
| Temporal contest generation | LiveCodeBench code_generation_lite | Post-cutoff generation under a frozen temporal split |
| Software API generation | BigCodeBench Complete | Library and software-engineering-oriented programming tasks |
| Code reasoning | CRUXEval-I, CRUXEval-O | Input and output prediction rather than synthesis only |
| Data-science code | DS-1000 | Python data-science library use |

SWE-bench Lite is secondary. It requires a frozen agent scaffold and container
environment, so it is not interpreted as a model-only curation result.

## Freeze Rules

1. Freeze every evaluator version, benchmark revision, prompt template, and
   decoding setting in a run manifest before first generation.
2. Run benchmark-exclusion and temporal audits before training. For
   LiveCodeBench, use only problems later than both the base-model cutoff and
   raw-corpus snapshot end.
3. Report every benchmark separately. Do not create one cross-metric score.
4. Report seed variation and paired task-level differences where evaluator
   outputs permit it.

## Active Execution Contract

`protocols/code_record_disjoint_confirmatory_evaluation_protocol.json` is the
active confirmatory execution contract. It freezes Base, confirmatory
Stage-A-release, and confirmatory Curated natural-budget arms for seeds 101,
202, and 303. Its preflight requires a complete benchmark-exclusion audit,
the record/text-disjoint integrity gate, exact curation artifacts, tokenizer
materialization, and frozen benchmark snapshots.

For the frozen Qwen3 tokenizer and 2,048-token packing, the confirmatory
Stage-A-release arm has 14,581,760 tokens (890 optimizer steps) and the
Curated arm has 14,417,920 tokens (880 steps). This 1.12% reduction is an
observed result of the no-budget policy on this corpus, not a selected target.

The current first-pass matrix contains HumanEval+, MBPP+, BigCodeBench,
CRUXEval-I, CRUXEval-O, and DS-1000. LiveCodeBench remains blocked until
auditable base-model cutoff and raw-corpus snapshot-end dates are declared and
a post-cutoff task subset is materialized. The public LiveCodeBench snapshot is
only a contamination audit surface; it does not establish absence of overlap in
unavailable private tests.

`protocols/code_7benchmark_pretraining_eligible_v3_execution.json` and older
six-benchmark execution files are historical artifacts. They use different
curation evidence or training inputs and must not be used for the confirmatory
result.

## Sources

- EvalPlus: https://github.com/evalplus/evalplus
- LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
- BigCodeBench: https://github.com/bigcode-project/bigcodebench
- CRUXEval: https://github.com/facebookresearch/cruxeval
- DS-1000: https://ds1000-code-gen.github.io/
- SWE-bench: https://github.com/SWE-bench/SWE-bench
