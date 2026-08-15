# Professor Review Package

## Suggested Email

**Subject:** IEEE BigData 2026 Draft for Review: Evidence-Bound Curation

Dear Professor Arifuzzaman,

I have completed a full draft of our paper, **“Evidence-Bound Curation: Auditable Membership Decisions for Language-Model Pretraining Data.”** The paper frames curation as a corpus-membership decision problem: evidence producers may score, retrieve, or compare data, but only typed and auditable policies can remove data. The runtime separates Validity, Redundancy, Quality, and Coverage authority across three stages, records every membership transition, and excludes benchmark results, training loss, utility, source reputation, target mixtures, and token budgets from runtime decisions.

The final code-domain experiment uses a 6.98M-token audited Python corpus and Qwen3-4B-Base. Raw, Normal, and Hard are trained with their own natural token budgets across seeds 101, 202, and 303. Normal retains 87.70% of Raw tokens and Hard retains 72.05%. Across BigCodeBench Complete, CRUXEval-I, CRUXEval-O, and DS-1000, the primary macro scores are 18.38 for Base, 20.86 for Raw, 20.59 for Normal, and 20.80 for Hard. Thus, Hard uses 27.95% fewer tokens and finishes 0.06 percentage points below Raw on the primary macro. HumanEval+ and MBPP+ are retained as mandatory secondary diagnostics; their mixed pattern is reported rather than omitted.

The intended contribution is not a universal quality score or a claim that curation always improves downstream performance. It is an auditable framework that makes the evidence-to-membership boundary executable through existing runtime checks: complete reason-coded traces, representative survival, Coverage zero-survivor restoration, forbidden-input isolation, and identity-bound deterministic replay. Unsupported similarity edges cannot delete data, Coverage can veto provisional removals, identity mutations invalidate inherited evidence, and external evaluation cannot retroactively change corpus membership. All 60 model-benchmark cells were recomputed from 42,820 task-level judgments and passed the provenance audit.

I would appreciate your feedback on three points: (1) whether the systems contribution and claim boundary are sufficiently clear, (2) whether the compression-with-near-retention result is presented with appropriate caution, and (3) which limitation should receive the most emphasis in the final revision. The current draft is 10 pages including references and compiles in IEEE conference format.

Sincerely,
Keonsoo Lee

## Evidence Snapshot

| Arm | Natural stream tokens | Retention vs. Raw | Primary macro, mean +/- sample SD |
|---|---:|---:|---:|
| Base | No update | N/A | 18.38 |
| Raw | 6,984,438 | 100.00% | 20.86 +/- 0.35 |
| Normal | 6,125,213 | 87.70% | 20.59 +/- 0.83 |
| Hard | 5,032,400 | 72.05% | 20.80 +/- 0.44 |

The primary macro is the unweighted mean over BigCodeBench Complete, CRUXEval-I, CRUXEval-O, and DS-1000 after seed aggregation. HumanEval+ and MBPP+ remain visible secondary diagnostics.

## Complete Benchmark Matrix (Table V)

All values are percentages. Trained-arm values preserve seed order `101 / 202 / 303`; Base is evaluated once as the no-update reference.

| Benchmark | Base | Raw 101 | Raw 202 | Raw 303 | Normal 101 | Normal 202 | Normal 303 | Hard 101 | Hard 202 | Hard 303 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval+ | 31.10 | 30.49 | 28.66 | 27.44 | 32.93 | 32.93 | 29.27 | 29.88 | 32.93 | 29.27 |
| MBPP+ | 58.47 | 45.50 | 50.00 | 51.32 | 54.76 | 52.65 | 53.70 | 54.76 | 54.76 | 53.97 |
| BigCodeBench Complete | 39.91 | 40.35 | 41.84 | 40.79 | 40.18 | 38.95 | 39.39 | 39.12 | 40.09 | 40.35 |
| CRUXEval-I | 3.50 | 6.38 | 7.63 | 7.50 | 8.75 | 5.50 | 4.75 | 5.63 | 6.25 | 6.63 |
| CRUXEval-O | 2.13 | 2.88 | 2.00 | 1.38 | 5.63 | 3.25 | 5.00 | 5.38 | 5.25 | 4.13 |
| DS-1000 | 28.00 | 33.10 | 33.60 | 32.90 | 31.60 | 32.20 | 31.90 | 31.10 | 33.00 | 32.70 |
| **Primary macro** | **18.38** | **20.68** | **21.27** | **20.64** | **21.54** | **19.97** | **20.26** | **20.31** | **21.15** | **20.95** |

| Summary | Base | Raw | Normal | Hard |
|---|---:|---:|---:|---:|
| **Primary macro mean +/- sample SD** | **18.38** | **20.86 +/- 0.35** | **20.59 +/- 0.83** | **20.80 +/- 0.44** |

The primary macro excludes HumanEval+ and MBPP+ by the timestamped analysis hierarchy. No seed or secondary diagnostic is omitted from the reported matrix.

## What the Draft Supports

- The framework implements separate, typed decision authority for Validity, Redundancy, Quality, and Coverage.
- Its authority contract is stated through five machine-checkable runtime invariants and a contract-checked-materialization proposition with a transition-induction proof sketch.
- Every final membership decision is traceable to versioned evidence, a policy, a reason code, and, when applicable, a representative.
- Stage C materially affected the output by restoring 99 Normal and 429 Hard chunks before final invariant checks.
- On this frozen Code corpus and training recipe, Hard removed 27.95% of Raw natural tokens while nearly retaining the primary macro.
- The complete 3-seed, 4-arm, 6-benchmark matrix is present and provenance-audited.

## What the Draft Does Not Claim

- It does not measure a universal or intrinsic notion of data quality.
- It does not show statistical non-inferiority; no benchmark-specific margin and paired decision rule were frozen for this final matrix.
- It does not establish consistent curated-over-Raw superiority.
- It does not establish cross-domain effectiveness or production readiness.
- It does not attribute downstream changes to an individual Quality or Coverage policy.

## Remaining Scientific Decisions

These are genuine evidence limits, not wording defects:

1. **Boundary-control experiment:** a Chunked-All control is needed to separate corpus-membership effects from whole-record versus chunk-boundary effects.
2. **Cross-domain evidence:** an independently frozen Math or general-text corpus and external suite are needed before claiming downstream effectiveness beyond Code.
3. **Quality reproducibility:** the fitted Quality gate includes proprietary-teacher observations and targeted synthetic enrichment; independent teachers or a releasable annotation artifact would strengthen reproducibility.
4. **Runtime-version closure:** the frozen models were trained on pre-fix artifacts affected by the 112-token source-record acquisition rule. The corrected runtime no longer makes that error, but a strict final-release claim would require retraining from corrected artifacts.
5. **Metadata closure:** 662 reference records lack complete record-level rights metadata. Rights are not selector inputs, but release requires a self-contained metadata repair.

The current paper discloses all five issues. They do not invalidate the decision-authority mechanism, but they bound the empirical and release claims.

## Submission Checklist

- [x] Complete IEEE conference-format manuscript
- [x] Ten pages including references
- [x] Full 60-cell benchmark table with all seed values
- [x] Compression and benchmark-delta figure
- [x] Artifact-level provenance audit for all benchmark cells
- [x] Compact reproducibility manifest with frozen protocols, configurations, run manifests, source entry points, and SHA-256 hashes
- [x] Explicit distinction between framework, policy, and external evaluation
- [x] Explicit disclosure of the post-EvalPlus analysis-hierarchy amendment
- [x] Balanced final-page reference columns
- [ ] Professor review and author approval
- [ ] Final proofreading after comments
- [ ] Overleaf clean build from the upload bundle
- [ ] Submission-system metadata and final PDF compliance check
