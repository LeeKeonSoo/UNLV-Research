# Quality Rule Development Matrix v1

## Scope

This is development-only structural evidence. It independently materializes
the active Quality rules and the registered web-control span candidate on
frozen Stage-B-pass snapshots. It does not read benchmark results, NLL,
Utility, source identity, domain labels, composition labels, or a retention
budget. It does not promote any candidate or establish downstream benefit.

The reports are stored outside the Git working tree:

- `D:/UNLV-Research/quality_rule_development_matrix_v1/code_report.json`
- `D:/UNLV-Research/quality_rule_development_matrix_v1/math_report.json`
- `D:/UNLV-Research/quality_rule_development_matrix_v1/general_report.json`

Token counts below are whitespace proxies. Frozen-tokenizer counts belong to
the later confirmatory materialization, not this rule-isolation audit.

## Inputs

| Development corpus | Frozen Stage-B input | Chunks |
| --- | --- | ---: |
| Code | `code_5m_corpus_v2/final_replay_v1/abc_curation_development_current_policy_v1/stage_b_pass_chunks.jsonl` | 8,058 |
| Math | `cross_domain_stress/abc_curation_openwebmath_5m_v1/stage_b_pass_chunks.jsonl` | 3,632 |
| General raw web | `general_raw_like_development_v1/commoncrawl/normal_curation/stage_b_pass_chunks.jsonl` | 707 |

## Rule-Isolated Results

Values are deltas from the same corpus baseline.

| Corpus | Arm | Chunk delta | Token delta | Triggered reason |
| --- | --- | ---: | ---: | --- |
| Code | Explicit generated artifact | -42 | -7,798 | `explicit_generated_artifact` |
| Code | License-comment-only | -1 | -8 | `license_comment_only_chunk` |
| Code | Empty HTML shell | 0 | 0 | none |
| Code | Cookie-control-only | 0 | 0 | none |
| Code | All active Quality | -43 | -7,806 | generated artifact + license-comment-only |
| Code | Web-control span candidate | 0 | -3 | one `url_directory_span_removed` |
| Math | Every active Quality arm | 0 | 0 | none |
| Math | Web-control span candidate | 0 | 0 | none |
| General raw web | Every active Quality arm | 0 | 0 | none |
| General raw web | Web-control span candidate | 0 | -6 | one `url_directory_span_removed` |

All 21 corpus-arm combinations passed the residual/whole-unit Coverage audit.
The candidate remained `runtime_active: false` in every report.

## Interpretation

1. The current active Quality floor is demonstrably narrow: it detects clear
   generated and license-only non-payload structure in Code, while preserving
   the observed Math and General chunks.
2. The web-control candidate fires only on a closed URL-directory pattern in
   these snapshots. It has no evidence yet for broader footer, navigation, or
   placeholder removal.
3. Zero removals in Math and General must not be reinterpreted as evidence that
   all text is high quality. It only says the current explicit rules found no
   authorized deletion event.
4. The next work is to build stronger candidates from closed structural
   patterns and adversarial fixtures, then repeat this matrix before frozen
   external evaluation.
