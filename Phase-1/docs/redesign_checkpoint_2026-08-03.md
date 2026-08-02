# Redesign Checkpoint - 2026-08-03

## Purpose

This checkpoint preserves the last empirically observed state before the
Core-Policy-Metric-Method redesign. It is an audit boundary, not a runtime
promotion. The preceding repository commit is `471a08d`.

## Frozen Framework State

- The public Cores remain Validity, Redundancy, Quality, and Coverage.
- Runtime stages remain Stage A, Stage B, and Stage C.
- External training, NLL measurement, and benchmark evaluation remain outside
  the curation runtime.
- Utility, benchmark outcomes, source reputation, domain quotas, and forced
  token budgets remain forbidden runtime inputs.
- `normal_structural_v1` remains the only active curation profile at this
  checkpoint. No contrastive observation has deletion authority.
- Development selection remains fail-closed on `quality_gate_not_ready` and
  `coverage_gate_not_ready`.

## E3b Observation Preserved

The first replaceable contrastive provider used Qwen3-4B-Base as the target
model and Qwen3-8B-Base as the reference model. The target ran in BF16 and the
reference ran in unvalidated INT8. Both arms used one frozen Qwen3 tokenizer.

| Item | Frozen value |
|---|---:|
| Balanced development records | 1,650 |
| Scored records | 1,500 |
| Omitted empty-payload records | 150 |
| Context-truncated records | 215 |
| Exact-copy observations checked | 300 |
| Exact-copy score mismatches | 0 |

The strict join required identical provider, scoring contract, tokenizer,
input artifact, record, route, token hash, and scored-token count identities.
The four native tokenizer files present in both Qwen snapshots were byte
identical.

| Route | Target NLL | Reference NLL | Excess NLL |
|---|---:|---:|---:|
| Code | 0.9339 | 0.8843 | 0.0496 |
| Math | 1.4331 | 1.3459 | 0.0871 |
| General | 2.5535 | 2.4372 | 0.1163 |

The observation did not establish a Quality threshold. Explicit boilerplate
wrappers often reduced both models' NLL and entropy, while excess-NLL changes
were small and route dependent. Absolute familiarity can therefore reward
easy repetitive chrome, and a generic larger-versus-smaller base-model gap is
not sufficient evidence for removal.

## Frozen Evidence Identity

| Artifact | SHA-256 |
|---|---|
| Contrastive audit file | `ff4508edaf8cb7a4ddc8a17b5b083f9dcca1f65a803ebabac61efecf73a65f30` |
| Contrastive audit logical report | `d86974720d89984567c5ad5ac79310f4975cc86b08f7b64dea0112ba447df419` |
| Joined evidence bundle | `a329d43ff58ef2d7d15edf8ed8eb9a04a3475a0af77812407b94df574e873dfe` |
| Development sample | `2e7f620c9eea7fc730cc3eeca8a56a8a8641160b9008145632cd0257eaf8316e` |
| Tokenizer compatibility file | `d203b3bba1cda8748d9098ef1f8d971f4dda8f9c0090e6772d15f3226d908150` |
| Qwen3-4B snapshot manifest file | `812fedfcca267eb90d3fbc21157d806abe830e3ba0d50afbc7d2fc7e3c2d1105` |
| Qwen3-8B snapshot manifest file | `c92a1d2978a137439c469281adda2b1a65fb7e797155562382690531283c9be0` |

Large model scores and joined observation records remain generated artifacts
on `D:/UNLV-Research/contrastive_quality_v1/development/` and are not tracked
by Git.

## Explicit Blockers

The frozen E3b audit status is `blocked` with nine blocker codes:

1. `reference_quantization_unvalidated`
2. `provider_training_disjointness_unverified`
3. `common_baseline_missing`
4. `insufficient_source_groups:code_artifact`
5. `empirical_effect_bins_missing:code_artifact`
6. `insufficient_source_groups:mathematical_content`
7. `empirical_effect_bins_missing:mathematical_content`
8. `insufficient_source_groups:general_prose`
9. `empirical_effect_bins_missing:general_prose`

These blockers cannot be overridden by a readiness boolean, a fixture result,
or a successful model-scoring run.

## Redesign Boundary

The next design must distinguish model roles before assigning score direction:

- learnability gap: target loss minus quality-reference loss;
- alignment gap: quality-reference loss minus background loss.

A high learnability gap is potential keep evidence, not automatic rejection. A
high alignment gap may become a rejection candidate only after source-disjoint
calibration and natural-budget validation. All provider replacements invalidate
their inherited calibration.

No implementation after this checkpoint may be described as active until the
new research contract, single authoritative runtime configuration, object
contracts, Core permissions, and Normal/Hard promotion rules are frozen.
