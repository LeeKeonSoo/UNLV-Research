# Code 7M Corpus Provenance

This document freezes the provenance of the Raw corpus used by the final
Code-7M curation, continued-pretraining, and benchmark experiment. Counts in
this document were re-audited on 2026-08-09 against the final JSONL rather than
inferred from directory names or collection targets.

## Frozen Experiment Input

- Final Raw JSONL:
  `D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/audited_release_candidates_v2.jsonl`
- SHA-256: `804dc90e35b360ae257fba99cdb1835d4b72ebd174528650dcdd20d9621a58e7`
- Final records: 4,890
- Qwen3-4B stream tokens, including one EOS token per record: 6,984,438
- Materialized packed training tokens: 6,979,584
- Dropped incomplete final packing tail: 4,854 tokens

The `code_5m_corpus_v2` directory name refers to the original raw-like
collection target. It does not describe the final mixed corpus size. The final
Raw arm is approximately 7M tokens because a GitHub reference pool was added
before the release and benchmark-contamination audits.

## Final Source Composition

| Source dataset | Source-pool role | Records | Qwen3-4B stream tokens | Share |
|---|---|---:|---:|---:|
| `bigcode/the-stack-dedup` | raw-like Python code | 4,228 | 4,723,925 | 67.64% |
| `github_reference_pool` | established-project reference code | 662 | 2,260,513 | 32.36% |
| **Total** | | **4,890** | **6,984,438** | **100.00%** |

The first source is `bigcode/the-stack-dedup`, not The Stack v2. The second
source contains fixed snapshots of eight public Python repositories:

| Repository | Snapshot commit | Declared license | Final records | Final stream tokens |
|---|---|---|---:|---:|
| `huggingface/transformers` | `7ea2320c76117e6742364808a666ef6f2fb40a67` | Apache-2.0 | 240 | 814,820 |
| `scikit-learn/scikit-learn` | `6d7f1f12c792bb17dd92b4bfbc86b9392e0f5ef6` | BSD-3-Clause | 164 | 811,376 |
| `pytest-dev/pytest` | `67a174fcee355334c53588be2eeba8df702477e9` | MIT | 106 | 262,734 |
| `pydantic/pydantic` | `428b0dba8924c8c3c588458928fb69c9eb203d3d` | MIT | 50 | 195,415 |
| `python-poetry/poetry` | `f46702336862f30050d5c641d5ed6f7568ded793` | MIT | 78 | 93,050 |
| `pallets/flask` | `36e4a824f340fdee7ed50937ba8e7f6bc7d17f81` | BSD-3-Clause | 10 | 35,744 |
| `pallets/click` | `333c28d79cd982990ee98eef61ec20ab1a4f38ba` | BSD-3-Clause | 7 | 34,663 |
| `psf/requests` | `f361ead047be5cb873174218582f7d8b9fcd9f49` | Apache-2.0 | 7 | 12,711 |
| **Total** | | | **662** | **2,260,513** |

## Acquisition-to-Experiment Reconciliation

The initial raw-like collection accepted 4,624 records from 541,527 scanned
records across 4,559 repositories and stopped at exactly 5,505,024 tokenizer
tokens. Collection applied path, file-size, license-allowlist, and repository
concentration filters. The reference-pool sampler then selected 695 records and
2,368,900 tokens from the eight fixed GitHub snapshots. Subsequent framework
release processing and benchmark exclusion produced the final composition
reported above.

The benchmark audit received 4,902 release candidates and excluded 12 records:
11 from `bigcode/the-stack-dedup` and one from the scikit-learn snapshot. The
audit covered the frozen HumanEval+, MBPP+, LiveCodeBench Code Generation Lite,
BigCodeBench Complete, CRUXEval Input/Output Prediction, and DS-1000 snapshots.
It retained 4,890 records for training. This audit only establishes the declared
snapshot boundary; it does not prove absence of contamination against every
possible benchmark or private test set.

## Claim Boundary

`github_reference_pool` and the historical internal label
`known_high_quality_reference` identify source provenance only. They are not an
intrinsic Quality score, a human label of every record, or evidence that every
reference record is useful to a language model. Source identity, source tier,
license, benchmark outcomes, and downstream Utility were not exposed as
selector features. The corpus should therefore be described as:

> A mixed Python-code corpus comprising a raw-like sample from
> `bigcode/the-stack-dedup` and version-pinned snapshots from eight established
> open-source GitHub repositories.

Do not describe the full 6.98M-token input as a purely raw The Stack corpus or
as an independently validated high-quality corpus.

## Authoritative Artifacts

- Raw-like collection report:
  `D:/UNLV-Research/code_5m_corpus_v2/raw_like_collection_report.json`
- Frozen mixed-input report:
  `D:/UNLV-Research/code_5m_corpus_v2/stage0_input/stage0_input_report.json`
- Reference shard A report:
  `D:/UNLV-Research/code_5m_corpus_v2/reference_pool_shard_a_v2/known_high_quality_reference_pool_report.json`
- Reference shard B report:
  `D:/UNLV-Research/code_5m_corpus_v2/reference_pool_shard_b_v2/known_high_quality_reference_pool_report.json`
- Benchmark exclusion audit:
  `D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/benchmark_exclusion_audit_v2.json`
- Final training-input report:
  `D:/UNLV-Research/final_all_policy_v1/luna_final_v1/training_inputs_v1/training_inputs_report.json`

Historical artifact names containing `stage0` belong to the acquisition run and
do not redefine the current public Stage A-B-C framework contract.
