# Multi-Domain Validation Plan for the Curation-Stage Paper

This plan addresses the main reviewer risk: the current paper has strong
framework structure, but the current completed training evidence is
domain-mixed. Code is a positive domain result; Math is a v3 repair-only
abstain result.

## Claim to Strengthen

The framework is not a generic quality detector. Its novelty is the separation
of three decisions that are often collapsed in LM data filtering:

1. record disposition: quarantine, Stage-A fail, Stage-A pass, retain-all;
2. budget allocation: optional Stage-B selection only when token budget binds;
3. downstream validation: Stage-C utility and guardrails after subset creation.

The expanded validation must show that this separation remains useful when the
input corpus contains both high-quality data and raw mixed data.

## Required Data Setting

Each domain should build two candidate pools:

- raw mixed pool: collected data with useful examples, duplicates, malformed
  records, generated files, noisy fragments, and benchmark-risk records;
- known high-quality reference pool: curated reference data used for context,
  not as selector labels or an oracle.

The next code-domain retest uses Hugging Face datasets as source pools rather
than a single monolithic dataset. The frozen Qwen3-4B protocol is
`configs/hf_mixed_corpus_retest_protocol_qwen3_4b_v1.json`; validate it with
`223_build_hf_mixed_corpus_retest_protocol.py` and
`validation/test_hf_mixed_corpus_retest_protocol.py`.

Primary mixture:

- `70%` raw-like Python code sources
- `30%` known-high-quality reference sources

Stress mixture:

- `90%` raw-like Python code sources
- `10%` known-high-quality reference sources

Planned Hugging Face source pools:

- raw-like: `bigcode/the-stack-v2`, `codeparrot/github-code`
- known-high-quality reference: `irds/codesearchnet`,
  `Nan-Do/code-search-net-python`

The mixture may be materialized as one candidate corpus, but source provenance
must not be flattened away. Every record must preserve source dataset, config,
split, tier, license family, repository or origin, content type, and token
proxy. Source tier and dataset identity are audit fields only and are forbidden
as Stage-B selector features.

Every report must include:

- records, chunks, and token proxy before curation;
- Stage-0 quarantined records;
- Stage-A passing chunks and tokens;
- Stage-B selected chunks and tokens;
- retained record fraction and token fraction;
- equal-token training payload hashes.

## Training Arms

Use equal-token budgets for all fine-tuning arms:

- base no-update model;
- raw-random equal-token fine-tuning;
- Stage-A-random equal-token fine-tuning;
- curated Stage-B equal-token fine-tuning;
- known high-quality reference equal-token fine-tuning.

The curated arm should be compared first against Stage-A random. Raw random is a
supporting baseline. Known high-quality reference is context, not an oracle.

## Benchmark Families

Code:

- held-out code NLL;
- EvalPlus as a code-retention guardrail;
- SWE-bench Lite or SWE-bench Verified when compute allows.

Math:

- held-out math NLL;
- GSM8K accuracy;
- MATH accuracy.

General text or instruction:

- held-out general-text NLL;
- instruction-following rubric or win-rate when available;
- code and general-task retention guardrails.

## Success Interpretation

If only code succeeds, the paper remains a code-domain framework validation.

Current state: historical Code natural-budget evidence is positive, but the
current Stage-A implementation requires a rerun before Code can be treated as
current evidence. Math v3 repairs v2 over-filtering but does not beat raw and
lacks GSM8K/MATH guardrails. Therefore the current multi-domain decision
remains `abstain`, not failure of the framework structure and not all-domain
success.

If code plus at least one non-code domain succeeds, the paper can claim
multi-domain evidence.

If a high-quality raw corpus passes Stage 0 and Stage A and fits the budget, a
retain-all outcome is correct. The framework should not force removal just to
look active.

If Stage-C evidence is missing or mixed, the correct outcome is abstain rather
than tuning the selector against benchmark outcomes.

## Near-Term Execution Order

1. Freeze dataset composition and equal-token budgets.
2. Materialize raw mixed, Stage-A random, curated, and reference payloads.
3. Report pre/post curation size and token counts.
4. Run lightweight NLL smoke tests.
5. Run full domain benchmarks only after the payloads and margins are frozen.
6. Add coverage and redundancy shift figures to the paper.
