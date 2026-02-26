# Analysis Artifact Contract (Schema v2)

## 1. Scope
This document defines the required contract for Phase-1 analysis outputs:
1. `outputs/khan_analysis.jsonl`
2. `outputs/tiny_textbooks_analysis.jsonl`
3. `outputs/run_manifest.json`

## 2. Record-Level JSONL Contract
Each JSONL record must include:

### 2.1 Required Top-Level Fields
1. `schema_version` (string): must be `"v2"`.
2. `source` (string): `khan_academy` or `tiny_textbooks`.
3. `doc_id` (string): stable document identifier.
4. `chunk_id` (integer): 0-based chunk index.
5. `text` (string): chunk text.
6. `word_count` (integer).
7. `domain_labels` (object[string -> number]).
8. `educational_markers` (object).
9. `quality_score` (number, [0, 1]).
10. `difficulty` (object).
11. `redundancy` (object).
12. `perplexity` (object).
13. `metric_tier` (object).
14. `validity_flags` (object).

### 2.2 `metric_tier` Object (Required Fixed Keys)
```json
{
  "domain": "core",
  "quality": "core",
  "difficulty": "core",
  "redundancy": "exploratory",
  "perplexity": "exploratory"
}
```
Allowed values: `"core"`, `"exploratory"`.

### 2.3 `validity_flags` Object (Required Fixed Keys)
```json
{
  "domain_valid": true,
  "quality_valid": true,
  "difficulty_valid": true,
  "redundancy_valid": false,
  "perplexity_valid": false
}
```
All values must be boolean.

### 2.4 Metric Sub-Objects
`redundancy` requires:
1. `exact_duplicate` (bool)
2. `near_duplicate_score` (number in [0, 1])
3. `semantic_duplicate_score` (number in [0, 1])
4. `n_gram_overlap_3` (number in [0, 1])
5. `n_gram_overlap_5` (number in [0, 1])

`perplexity` requires:
1. `gpt2` (number or null)
2. `token_level_variance` (number or null)
3. `sentence_level_mean` (number or null)
4. `max_sentence_perplexity` (number or null)

## 3. Run Manifest Contract (`outputs/run_manifest.json`)
Required top-level fields:
1. `schema_version` (string)
2. `phase` (string)
3. `objective_mode` (string)
4. `generated_by` (string)
5. `code_commit_hash` (string)
6. `dataset_versions_and_counts` (object)
7. `threshold_config` (object)
8. `metric_tier` (object; same fixed keys as record-level)
9. `reliability_gate_outcomes` (object)
10. `perplexity_fallback_behavior` (object)
11. `redundancy_runtime_status` (object)

### 3.1 `threshold_config`
Must include:
1. `TOP_K_DOMAINS`
2. `MIN_SIMILARITY`
3. `CHUNK_SIZE`

### 3.2 `reliability_gate_outcomes`
Must include metric-level gate objects for:
1. `domain`
2. `quality`
3. `difficulty`
4. `redundancy`
5. `perplexity`

Each gate entry should expose:
1. threshold
2. measured value (or null when manual validation pending)
3. pass/fail/null
4. status string

## 4. Dashboard/CSV Compatibility Requirements
The dashboard and CSV exporter must safely consume:
1. `schema_version`
2. `metric_tier.*`
3. `validity_flags.*`

Interpretation policy:
1. Show Core/Exploratory badges.
2. Hide non-claimable metrics from headline comparison.
3. Keep exploratory/non-claimable metrics visible as diagnostic tables.

## 5. Minimal Valid v2 Record Example
```json
{
  "schema_version": "v2",
  "source": "khan_academy",
  "doc_id": "cosmopedia://khanacademy/algebra_1/abcd1234",
  "chunk_id": 0,
  "text": "In this lesson, we solve linear equations...",
  "word_count": 145,
  "domain_labels": {"Math::Algebra 1": 1.0},
  "educational_markers": {"has_examples": true, "has_explanation": true, "has_structure": false},
  "quality_score": 0.6667,
  "difficulty": {
    "flesch_kincaid_grade": 8.2,
    "flesch_reading_ease": 65.3,
    "smog_index": 8.9,
    "avg_sentence_length": 15.3,
    "avg_word_length": 4.6,
    "rare_words_pct": 0.08,
    "lexical_diversity": 0.68
  },
  "redundancy": {
    "exact_duplicate": false,
    "near_duplicate_score": 0.23,
    "semantic_duplicate_score": 0.31,
    "n_gram_overlap_3": 0.15,
    "n_gram_overlap_5": 0.08
  },
  "perplexity": {
    "gpt2": 42.3,
    "token_level_variance": 6.2,
    "sentence_level_mean": 39.8,
    "max_sentence_perplexity": 58.7
  },
  "metric_tier": {
    "domain": "core",
    "quality": "core",
    "difficulty": "core",
    "redundancy": "exploratory",
    "perplexity": "exploratory"
  },
  "validity_flags": {
    "domain_valid": true,
    "quality_valid": true,
    "difficulty_valid": true,
    "redundancy_valid": true,
    "perplexity_valid": true
  }
}
```
