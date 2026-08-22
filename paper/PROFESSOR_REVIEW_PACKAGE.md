# Professor Review Package

## Suggested Email

**Subject:** IEEE BigData 2026 Draft for Review: Evidence-Bound Curation

Dear Professor Arifuzzaman,

I have completed the current draft of **“Evidence-Bound Curation: A Traceable
Framework for Language-Model Training Data.”** The paper introduces a curation
framework that separates evidence generation from the authority to change
corpus membership. Validity, Redundancy, Quality, and Coverage therefore have
different permissions instead of acting as interchangeable keep/drop votes.
Each accepted action records its reason, policy version, content identity, and
any representative or restoration link.

The final frozen policy was evaluated on similarly sized Code and Math corpora
with Qwen3-4B-Base. Every trained arm used all naturally retained packed tokens
for one pass and seeds 101, 202, and 303. The same source corpora were also
processed with frozen Data-Juicer and NeMo Curator recipes. BigCodeBench was
rescored for every arm and seed with one common evaluator source, dataset hash,
and frozen ground-truth cache.

The three most important empirical results are:

1. Ours retained 6,242,304 of 6,979,584 Code packed tokens (89.44%) and
   5,767,168 of 6,979,584 Math packed tokens (82.63%). Final Coverage restored
   104 Code and 143 Math units, after which both complete rechecks passed.
2. In Code, Ours exceeded both executable curation baselines on three of six
   benchmarks: HumanEval+ (31.50), MBPP+ (54.67), and CRUXEval-O (2.83).
   Data-Juicer and NeMo were stronger on BigCodeBench, CRUXEval-I, and DS-1000.
3. In Math, Ours improved over the no-update Base on all four reported scores,
   but trailed both curation baselines on all four. This result limits the
   downstream claim while preserving the paper's auditable decision-system
   contribution.

The intended contribution is not a universal quality score or a claim that one
frozen policy always improves training. It is an executable authority model in
which unsupported similarity cannot delete data, a removed duplicate must name
a surviving representative, Coverage can restore but cannot delete, stale
evidence is rejected by content identity, and benchmark outcomes cannot feed
back into the frozen selector.

I would appreciate your feedback on whether the technical contribution and its
claim boundary are clear, and whether the mixed Code/Math evidence is explained
appropriately.

Sincerely,  
Keonsoo Lee

## Final Token Exposure

| Domain and arm | Packed training tokens | Retention vs. Raw |
|---|---:|---:|
| Code Raw | 6,979,584 | 100.00% |
| Code Ours | 6,242,304 | 89.44% |
| Code Data-Juicer | 5,505,024 | 78.87% |
| Code NeMo Curator | 6,029,312 | 86.38% |
| Math Raw | 6,979,584 | 100.00% |
| Math Ours | 5,767,168 | 82.63% |
| Math Data-Juicer | 4,603,904 | 65.96% |
| Math NeMo Curator | 5,881,856 | 84.27% |

## Final Benchmark Matrix

All trained-arm values are three-seed means in percent. Base is the no-update
reference.

| Code benchmark | Base | Raw | Ours | Data-Juicer | NeMo Curator |
|---|---:|---:|---:|---:|---:|
| HumanEval+ | 31.10 | 28.86 | 31.50 | 28.25 | 30.08 |
| MBPP+ | 58.47 | 48.94 | 54.67 | 41.27 | 50.97 |
| BigCodeBench Complete | 39.65 | 40.61 | 39.12 | 40.35 | 40.41 |
| CRUXEval-I | 3.50 | 7.17 | 5.79 | 9.79 | 7.42 |
| CRUXEval-O | 2.13 | 2.08 | 2.83 | 1.83 | 2.25 |
| DS-1000 | 28.00 | 33.20 | 34.13 | 34.23 | 34.63 |

| Math benchmark | Base | Raw | Ours | Data-Juicer | NeMo Curator |
|---|---:|---:|---:|---:|---:|
| GSM8K strict | 75.13 | 77.79 | 77.53 | 78.72 | 78.29 |
| GSM8K flexible | 57.54 | 71.19 | 63.08 | 67.60 | 64.87 |
| GSM8K normalized | 76.42 | 80.11 | 79.05 | 79.78 | 80.34 |
| MATH-500 | 3.80 | 8.67 | 7.07 | 8.07 | 8.07 |

## Final Framework Evidence

- Code: Stage A 8,026 chunks; final 7,270; Quality not-selected 816;
  Redundancy removals 3; Coverage restores 104.
- Math: Stage A 6,619 chunks; final 5,528; Quality not-selected 1,223;
  Redundancy removals 6; Coverage restores 143.
- Both Coverage audits report `complete_recheck_passed=true` and
  `may_create_new_removal=false`.
- The policy is packaged as `beta_release`; neither Coverage audit claims
  production or scientific promotion.

## What the Draft Supports

- A single frozen, domain-neutral decision contract can be run on Code and Math
  without domain quotas or benchmark feedback.
- Every membership change is tied to typed evidence and a reason-coded trace.
- Coverage materially changed both final corpora and passed complete rechecks.
- Ours is competitive on selected Code capabilities while using fewer tokens
  than Raw.
- The unfavorable Math comparison is reported directly and bounds the
  downstream-effectiveness claim.

## What the Draft Does Not Claim

- It does not measure universal intrinsic data quality.
- It does not show that the frozen policy improves every benchmark or domain.
- It does not claim production promotion from the two confirmatory corpora.
- It does not claim that the tested Data-Juicer or NeMo recipe represents every
  configuration available in those systems.

## Submission Checklist

- [x] Final five-arm, three-seed Code and Math result surface
- [x] Common-cache BigCodeBench recheck for every displayed arm and seed
- [x] Final Code and Math curation and Coverage reports packaged
- [x] Q1 veto and Q2--Q4 positive-selection rule stated consistently
- [x] Historical Normal/Hard and matched-random artifacts excluded from final claims
- [x] Reproducibility manifest and SHA-256 identities
- [ ] Professor review and author approval
- [ ] Final Overleaf clean build and submission-system PDF check
