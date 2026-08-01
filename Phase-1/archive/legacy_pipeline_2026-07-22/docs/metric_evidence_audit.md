# Metric Evidence Audit

## Purpose

This audit separates literature support, implementation correctness, automated
metric behavior, and causal curation evidence. A citation does not certify the
project's implementation, thresholds, weights, or downstream benefit.

## Evidence Classes

| Class | Meaning | Claim allowed |
| --- | --- | --- |
| `paper_backed_method` | The implementation directly follows a documented method with materially matching inputs and decision rule | The method is reproduced within documented differences |
| `paper_aligned_principle` | Literature supports the general principle, but this project defines its own implementation or parameters | The design is literature-aligned, not literature-validated |
| `project_hypothesis_frozen` | A project-specific rule frozen before target-model outcomes | The rule is a preregistered hypothesis |
| `engineering_diagnostic` | A heuristic or diagnostic used to inspect behavior, not support a release claim | Engineering or explanatory evidence only |

## Current Temporal-Code Classification

| Component | Evidence class | Current evidence | Missing proof |
| --- | --- | --- | --- |
| Exact duplicate rejection | `paper_backed_method` | Deterministic hashes, split-local rejection tests | Large-corpus operational error audit |
| Hard near-duplicate rejection | `paper_aligned_principle` | SimHash shortlist plus verified shingle overlap; labeled fixture precision `1.0`, recall `0.5`; silver calibration precision `1.0`, recall `0.626667` | Broader repository-disjoint calibration and recall improvement without dropout |
| Python AST chunking and parseability | `paper_aligned_principle` | Syntax-aware chunking and bounded real-corpus smoke | Broader language/repository coverage |
| Selection Value Evidence and redundancy as Stage-B budget signals | `paper_aligned_principle` | Literature supports filtering, selection, and deduplication | Current proxy construction is not reproduced from a paper and has no hard-reject authority |
| Python selection-value formula (`Quality` legacy field) | `project_hypothesis_frozen` | Property fixtures, retain-all semantics, and frozen smoke behavior | Target-model ablation and confirmatory evidence |
| Documentation selection-value formula (`Quality` legacy field) | `project_hypothesis_frozen` | Property fixtures, retain-all semantics, and frozen smoke behavior | Target-model ablation and confirmatory evidence |
| Stage-B objective weights `0.8 / 0.2` | `project_hypothesis_frozen` | Frozen before temporal-code Stage-B smoke | Development ablation and untouched confirmation |
| Structural duplicate risk `0.85` | `project_hypothesis_frozen` | Identifier-rename checks and binary template-saturation diagnostic; known gap: not saturation-magnitude-sensitive | Count-sensitive Stage-B evidence and target-model guardrails without promotion leakage |
| Coverage relative-token-share floor `0.5` | `project_hypothesis_frozen` | Prevents bounded-smoke distribution collapse | No-Coverage ablation and target-model evidence |
| Indexed redundancy search | `engineering_diagnostic` | Exact equality to all-pairs on all 318 smoke chunks | Larger-scale runtime and equality checks |
| Human or LLM review | `engineering_diagnostic` | Optional score-hidden review tooling | Never required for policy approval or Stage-C entry |
| Stage-C equal-budget Utility | `paper_aligned_principle` | Common disjoint Stage-A baseline contract and earlier text experiments | Temporal-code 4B development and untouched confirmatory results |

## Primary Validation Rule

The framework intentionally avoids requiring human intuition as a selector
ground truth. Stage-B policies are validated by automated property tests and
then by downstream causal comparisons:

```text
frozen Stage-B policy
vs disjoint Stage-A random equal-budget
vs raw random and frozen ablations
under matched model, tokens, compute, seeds, and evaluation
```

Human review and LLM judging may explain failures, but they cannot approve,
reject, tune, or block a Stage-B policy.

## Required Proof Sequence

1. Verify stage ownership, determinism, isolation, and contamination contracts.
2. Expand automated metamorphic tests for each Stage-B proxy.
3. Freeze full-selector and ablation policies before target-model outcomes.
4. Run the policy on a provenance-rich raw corpus.
5. Compare equal-budget training arms on development evidence.
6. Freeze the candidate and release recipe.
7. Evaluate on untouched confirmatory evidence.
8. Claim benefit, negative finding, or abstention exactly as supported.

The frozen temporal-code ablation contract is
`configs/temporal_code_stage_b_ablation_protocol_v1.json`.
