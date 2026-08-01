# Phase-1 Core Metrics Overview

## 1. What Phase-1 is trying to do

Phase-1 is not a generic text-quality project.

The goal is:

- identify dataset-construction criteria for language-model pretraining
- keep useful information
- reduce redundancy and obvious waste
- preserve coverage
- eventually validate whether the selected data improves learning utility

In short:

- `Phase-1 = pretraining corpus construction framework`

## 2. What the framework analyzes

The framework analyzes training chunks through three layers:

1. `Hard filters`
   - remove unusable chunks before ranking
2. `Selection signals`
   - rank usable chunks by validated structural and redundancy signals
3. `Outcome validator`
   - check whether the ranking policy is aligned with actual learning utility

Current canonical split:

- `structural_validity_gate` = canonical Validity hard filter
- `reference_quality_score` = canonical quality signal
- `exact_duplicate_indicator` = hard filter
- `shingle_near_duplicate_risk_score` = canonical redundancy signal
- `shingle_near_duplicate_indicator` = canonical dedup filter
- `subset_coverage_retention_score` = canonical subset-level coverage validator
- `small_lm_probe_gain_score` = canonical subset-level utility validator
- `structural_validity_score` = diagnostic validity audit score
- `explanatory_quality_proxy` = diagnostic proxy
- `tail_cluster_rarity_proxy` = diagnostic proxy
- `predictive_utility_proxy` = diagnostic utility proxy
- `fixed_token_probe_gain_score` = deprecated diagnostic utility fallback

## 3. Core metrics

### 3.1 `structural_validity_gate`

What it analyzes:

- whether the chunk is minimally usable as language data
- whether the text is hard-invalid before quality or redundancy analysis

Current hard-fail signals:

- empty or too-short units
- encoding/control-character corruption
- non-language fragments
- excessive symbol noise
- markup or extraction residue
- hard broken repetition runs

Interpretation:

- `1.0` = structurally usable enough for later scoring
- `0.0` = hard-invalid and excluded before selection ranking

Current status:

- `paper_aligned`

### 3.2 `structural_validity_score`

What it analyzes:

- diagnostic support for the binary Validity gate
- which hard-failure or warning rules explain the gate decision

Current signals:

- `word_count`
- `sentence_count`
- `alpha_ratio`
- hard-failure rule counts
- warning rule counts

Interpretation:

- high score = clean structural audit surface
- low score = structural usability risk
- pass/fail still comes from `structural_validity_gate`

Current status:

- `diagnostic`

### 3.3 `reference_quality_score`

What it analyzes:

- whether the chunk is informative, coherent, and useful after it already passes Validity
- whether reference-model quality evidence is being distorted by style or length bias

Current signals:

- reference-trained clean-vs-corrupted quality model
- structural hygiene and information-support calibration
- style bucket, length bucket, lexical diversity, and boilerplate checks as calibration context

Interpretation:

- high score = coherent information-bearing text, including short technical/reference text when evidence supports it
- low score = boilerplate, shallow, noisy, or low-information text
- selection policy uses a soft top-quality cap so `0.99+` chunks do not dominate without forcing exact quality-band matching

Current status:

- `paper_aligned_style_length_normalized_v2`

### 3.4 `explanatory_quality_proxy`

What it analyzes:

- whether the chunk looks explanatory, information-bearing, and concept-rich
- whether the chunk is closer to useful instructional/expository text than to boilerplate or shallow procedural text

Current signals:

- positive prototype similarity
- negative prototype similarity
- `info_density`
- `explanatory_signal`
- `definition_signal`
- `structure_signal`
- penalties for:
  - procedural text
  - glossary-like text
  - conclusion boilerplate
  - excessive bullet structure
  - unnatural sentence shape

Interpretation:

- high score = coherent explanatory content with real information
- low score = shallow, repetitive, list-like, or procedural filler

Current status:

- `paper_informed_proxy`

### 3.5 `exact_duplicate_indicator`

What it analyzes:

- whether the exact same chunk appears multiple times

Current signals:

- `text_hash`
- `hash_counts` lookup from the index

Interpretation:

- `1.0` = exact duplicate exists
- `0.0` = unique at exact-hash level

Current status:

- `paper_aligned_basic`

### 3.6 `shingle_near_duplicate_indicator`

What it analyzes:

- whether the chunk sits in a dense repeated-content neighborhood even if the exact text is not identical

Current signals:

- `simhash_prefix_bucket_count`
- row-weighted quantile calibration of prefix bucket size
- log-scaled redundancy score from local SimHash bucket density

Important implementation note:

- semantic `cluster_size` is **not** used anymore for near-dup scoring
- KMeans `cluster_size` is treated as a coverage structure signal, not a duplication signal

Interpretation:

- high score = chunk likely belongs to a dense near-duplicate neighborhood
- low score = chunk is relatively isolated at the local SimHash-prefix level

Current status:

- `operational_proxy`

### 3.7 `shingle_near_duplicate_risk_score`

What it analyzes:

- whether the chunk carries harmful duplicate burden
- whether repeated phrasing is wasteful duplication or useful recurrence such as definitions, examples, exercises, or technical-reference patterns

Current signals:

- refined SimHash shortlist
- maximum token 3-gram Jaccard overlap within local candidate neighborhoods
- prefix collision pressure
- intra-chunk repetition pressure
- useful-recurrence relief when verified overlap is low

Interpretation:

- high score = harmful repeated overlap or wasteful repetition
- low score = unique text or useful structured recurrence without verified duplicate burden

Current status:

- `paper_aligned_harmful_redundancy_v1`

### 3.8 `tail_cluster_rarity_proxy`

What it analyzes:

- whether the chunk contributes to tail coverage rather than to already dominant head regions

Current signals:

- semantic `cluster_size` from index clustering
- rarity proxy derived from cluster size

Subset-level coverage report also uses:

- `distribution_similarity`
- `rare_cluster_retention`
- `source_coverage_support`
- `style_coverage_support`
- `semantic_coverage_support`

Interpretation:

- high score = chunk is rarer and potentially important for coverage retention
- low score = chunk comes from a common head cluster
- explicit domain coverage is only claimed when domain metadata exists; otherwise source-bucket support is reported as fallback

Current status:

- `paper_aligned_source_style_semantic_support`

### 3.9 `predictive_utility_proxy`

What it analyzes:

- whether the chunk is likely to help learning utility
- whether the chunk looks aligned with useful reasoning, explanation, or knowledge patterns

Current signals:

- utility prototype similarity
- negative utility prototype similarity
- `quality_support`
- `explanatory_signal`
- `definition_signal`
- `qa_signal`
- `concept_signal`
- penalties for:
  - procedural text
  - glossary-like text
  - conclusion boilerplate
  - bullet-heavy/list-heavy text

Optional mode:

- heuristic only
- `probe_calibrated` predictor if gate conditions pass

Important interpretation note:

- this metric is currently **not** a canonical selection gate
- it is kept only as a diagnostic/development signal
- canonical Utility is `small_lm_probe_gain_score`, not this proxy

Current status:

- `diagnostic_only`

## 4. What we are claiming right now

Safe claim:

- the framework has an explicit, literature-linked metric contract
- the framework distinguishes filtering, ranking, and utility validation
- the framework can generate reproducible scored outputs and subset summaries

Not yet safe claim:

- every metric is already a fully validated paper-faithful implementation
- the current selector is already optimal for pretraining utility

Current blockers:

1. `small_lm_probe_gain_score` remains the hardest certification axis and must show stable positive learning signal against the multi-matched Stage-A baseline.
2. explicit semantic domain coverage still requires real domain metadata; without it, the framework reports source-bucket fallback rather than claiming true domain coverage.

## 5. Short presentation version

Use this if a slide needs only one summary block:

> We evaluate training data with five core axes: Validity, Quality, Redundancy, Coverage, and Utility. Stage A filters structurally invalid and duplicate chunks, Stage B selects among usable chunks using calibrated Quality and harmful-Redundancy risk, and Stage C validates the selected subset for source/style/semantic Coverage and small-LM Utility. Utility remains an outcome validator, not a selector objective.

## 6. Recommended slide labels

If you need short slide headers:

- `Validity`
- `Quality`
- `Exact Duplication`
- `Near-Duplicate Redundancy`
- `Coverage Retention`
- `Small-LM Utility`

If you need one line under each:

- `Validity`: Is the text structurally usable?
- `Quality`: Is the text explanatory and information-bearing?
- `Exact Duplication`: Is the chunk repeated verbatim?
- `Near-Duplicate Redundancy`: Is the chunk in a dense repeated-content neighborhood?
- `Coverage Retention`: Does the subset preserve source, style, and semantic coverage?
- `Small-LM Utility`: Does the selected subset improve held-out learning outcome against a fair baseline?
