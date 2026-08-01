# Code-Domain Training Validation Protocol

## Decision

The next main validation should test the framework as a training-data curation
system, not as an executable-task mining system.

The primary experiment is:

```text
raw permissive Python code corpus
-> framework curation
-> equal-budget continued pretraining
-> external and heldout code evaluation
```

The retrospective strict-E2 pipeline is retained as secondary executable
evidence. It is no longer the main blocker for starting the raw-vs-curated
training validation.

## Why This Replaces The E2-First Track

Strict E2 tasks are useful because they provide objective executable evidence,
but acquiring them is expensive and filters for a narrow property:

```text
Can this repository patch be verified by parent/merge test behavior?
```

The paper's central claim is broader:

```text
Can the framework turn a raw candidate corpus into a better LM-training
dataset than equal-budget random baselines?
```

Therefore the primary evidence should come from equal-token target-model
updates. E2 evidence should support the code-domain evaluation story, not
define the entire research path.

## Primary Domain

Use Python code, with a preferred focus on general utility and data-science
code.

Reasons:

- Python has mature open code corpora and evaluation benchmarks.
- Real collected code contains clear curation hazards: duplicates, generated
  files, vendored code, boilerplate, broken snippets, low-signal files,
  licenses, and benchmark contamination.
- Functional and benchmark-based evaluation is more objective than subjective
  domain quality review.
- Medical or other high-stakes domains introduce safety, privacy, and
  specialist-claim burdens that can obscure the curation-framework claim.

## Candidate Corpus

The candidate corpus should be raw-like and provenance-rich.

Preferred sources:

- permissively licensed Python files from a fixed GitHub or Software Heritage
  style snapshot
- The Stack v2 / StarCoder2-style source metadata where available
- repository/file metadata sufficient for license, contamination, duplicate,
  and source-slice audits

Avoid using only already curated high-quality code as the candidate corpus.
High-quality sources are useful as reference arms, not as the sole raw input.

## Model Choice

Primary target model:

```text
Qwen/Qwen3-4B-Base
```

Reason: it is a modern 4B-class base model and is closer to the intended
small-language-model setting than the earlier 0.5B pilot.

Secondary model-transfer check:

```text
Qwen/Qwen2.5-Coder-3B-Base or Qwen/Qwen2.5-Coder-7B-Base
```

Reason: a code-specialized model tests whether the curation effect transfers
to a stronger code prior. If local memory is limiting, use 3B first.

## Training Method

Use continued pretraining / domain-adaptive language-model training, not SFT,
for the primary corpus experiment.

Frozen training requirements:

- causal LM next-token objective
- equal target-token budget for every primary arm
- same optimizer, steps, sequence length, packing, LoRA/QLoRA settings, and
  seed schedule for every arm
- no Stage-C Utility, benchmark, or human/LLM judgment in Stage-B selection
- development runs before untouched confirmation

The first local method can be QLoRA. A full-parameter replication may be
reported later, but QLoRA evidence must be labeled as parameter-efficient
continued pretraining.

## Required Arms

| Arm | Role |
| --- | --- |
| `base_no_update` | No-training reference and regression baseline |
| `raw_random_equal_budget` | Raw collected data under equal tokens |
| `stageA_random_equal_budget` | Usable-data random baseline after hard gates |
| `curated_equal_budget` | Framework-selected primary treatment |
| `known_high_quality_equal_budget` | Reference upper-style baseline, not the raw input |

Optional arms:

- `stageA_all_reference`
- `raw_all_reference`
- Quality-only Stage-B ablation
- no-Coverage-support Stage-B ablation

## Evaluation

Primary evaluation should combine external benchmark and internal heldout
signals:

- EvalPlus HumanEval+
- EvalPlus MBPP+
- LiveCodeBench on a frozen, temporally disjoint task slice
- BigCodeBench or BigCodeBench-Hard
- DS-1000 for data-science Python
- disjoint heldout code/documentation NLL
- contamination audit against benchmark tasks and solutions
- general capability/forgetting checks where feasible

Repository-patch strict E2 tasks are secondary executable evidence. They can
be reported as an additional heldout slice if enough tasks are available, but
they should not block the primary raw-vs-curated training comparison.

## Primary Success Rule

The framework supports a scoped positive training claim only if:

1. `curated_equal_budget` beats `raw_random_equal_budget` on the predeclared
   external-code aggregate.
2. `curated_equal_budget` beats or matches `stageA_random_equal_budget`.
3. The result is stable across at least two development seeds, with three seeds
   preferred for a paper claim.
4. Heldout NLL and general checks do not show a predeclared major regression.
5. The benchmark-contamination audit passes.

If `curated_equal_budget` fails, the framework can still publish an abstention
or negative finding, as long as Stage B was frozen before target-model
outcomes.

## Independent LiveCodeBench Pilot

The first frozen pilot uses 48 2025 tasks, seed 101, Qwen3-4B-Base, and the
natural-budget raw and curated adapters. Base, raw, and curated each score
`9/48` (`18.75%`) pass@1. The generated programs differ across arms, but the
48 correctness outcomes are identical. This is neutral transfer evidence and
does not authorize selector tuning. A paper-level independent-benchmark claim
requires a pre-registered multi-seed evaluation with enough medium/hard task
power to distinguish the arms.

## Relationship To Existing E2 Work

Current strict-E2 retrospective status:

```text
825 execution attempts
167 task-valid E2 tasks
```

This is valuable engineering and executable-evidence progress. It should be
reported as a secondary validation asset. It should not force the project to
delay the main raw-corpus training validation until the earlier 542 valid-E2
development target is reached.

## References

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Qwen2.5-Coder Technical Report: https://arxiv.org/abs/2409.12186
- The Stack: https://arxiv.org/abs/2211.15533
- StarCoder2 and The Stack v2: https://arxiv.org/abs/2402.19173
- EvalPlus: https://arxiv.org/abs/2305.01210
- BigCodeBench: https://arxiv.org/abs/2406.15877
- DS-1000: https://arxiv.org/abs/2211.11501
