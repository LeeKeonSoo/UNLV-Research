# Final Experiment Readiness

Status: executable candidate protocol, not a release claim.

## Readiness Levels

| Level | Required evidence | Authority |
|---|---|---|
| Framework materialization | Normal and Hard complete A-B-C, Stage C consumes the frozen semantic graph, full Coverage recheck passes, Hard is subset-or-equal Normal | The framework can emit candidate datasets |
| External confirmatory | Framework materialization plus exact Qwen3-4B tokenizer counts and natural-budget training blocks for Raw, Normal, and Hard | External training may start |
| Paper claim | Confirmatory results plus promoted Policies within the stated domain | The evidence may support a bounded paper claim |
| Production release | Every enabled Policy/provider passes release gates and operational checks | The package may be described as release-ready |

`final_experiment_preflight.py` computes these states independently. A passing
lower level never implies a passing higher level.

## Frozen Candidate

- Input: audited benchmark-excluded Code corpus at
  `D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/audited_release_candidates_v2.jsonl`.
- Modes: Raw, Normal, Hard.
- Runtime boundary: Stage A Validity, Stage B Redundancy/Quality proposals,
  Stage C Coverage veto and materialization.
- External boundary: Qwen3-4B-Base continued pretraining with seeds 101, 202,
  and 303 under each arm's natural token budget.
- Mandatory external suites: HumanEval+, MBPP+, BigCodeBench Complete,
  CRUXEval-I, CRUXEval-O, and DS-1000.
- Amended primary reasoning suite: BigCodeBench Complete, CRUXEval-I,
  CRUXEval-O, and DS-1000.
- Mandatory secondary short-function diagnostics: HumanEval+ and MBPP+.

The analysis hierarchy is timestamped in
`protocols/code_reasoning_primary_amendment_v1.json`. It was frozen after
partial EvalPlus observation but before any non-Base reasoning-suite result.
All six benchmarks, all arms, and all seeds must still be reported.

No equal-token resampling, target retention fraction, maximum token budget,
Utility, NLL, or benchmark result is visible to curation.

## Current Boundary

The candidate A-B-C implementation is internally consistent. Semantic Coverage
can veto Stage-B proposals, emits required retain IDs, explicitly
rematerializes, and reruns the complete contract. Scientific release is still
blocked by unpromoted Quality and near-duplicate Policies and by missing
protected false-veto and independent multidomain Coverage evidence.

The current preflight result is:

| Field | Result |
|---|---:|
| Framework materialization ready | true |
| External confirmatory ready | true |
| Paper claim ready | false |
| Production release ready | false |

| Arm | Stream tokens | Packed tokens | Optimizer steps |
|---|---:|---:|---:|
| Raw | 6,984,438 | 6,979,584 | 426 |
| Normal | 6,961,249 | 6,946,816 | 424 |
| Hard | 6,747,888 | 6,733,824 | 411 |

Normal removes 0.33% and Hard removes 3.39% of the Raw token stream. This
already curated Code corpus is therefore a low-opportunity compression case;
the external experiment tests whether those smaller natural budgets preserve
or improve performance, but that result cannot tune the frozen runtime.
