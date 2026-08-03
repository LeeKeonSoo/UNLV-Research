# Configuration Status

This directory contains several classes of JSON artifacts. File presence alone
does not grant runtime authority.

## Redesign Root

`curation_framework_v1.json` is the single machine-readable root for the next
framework version. Its schema is
`schemas/curation_framework_v1.schema.json`. It is design-only until Block 7;
`framework_objects_v1.json`, `contrastive_quality_protocol_v2.json`, and
`framework_profiles_v1.json` implement its typed registries. They are not
Policy promotion: Block 7 integrates their identity and Stage permissions into
the runtime, while both new profiles remain release-disabled until their Policy
gates close.

`framework_release_validation_v1.json` is the frozen Block 8 integrity
protocol. It binds the Core behavior fixtures, negative fail-closed scenarios,
blocked Policy inventory, and curated-output equivalence hash. A passing Block
8 report validates implementation integrity only; it does not enable a profile
or promote a Policy.

## Runtime Contracts

The runtime contract surface is anchored by:

- `curation_contract.json`
- `curation_run_contract.example.json`
- `core_policy_registry.json`
- `policy_card_contract.json`
- `policy_cards.json`
- `policy_profiles.json`

These files are checked against the observed call path documented in
`../docs/framework_consistency_baseline.md`. Normal/Hard profile identity now
determines all policy switches; a run contract may provide operational paths
and sizes but cannot override Stage-A/B/C policy decisions.

## Candidate and Evidence Artifacts

Files named with `candidate`, `development`, `calibration`, `evidence`,
`provider`, `quality`, `router`, or a versioned experimental profile generally
describe candidate behavior, frozen evidence, or a promotion decision. They do
not become active merely by being loaded by an audit or test.

## External Experiment Inputs

Files describing benchmark suites, training arms, clean controls,
confirmatory runs, or preregistration belong to development or external
evaluation. They must remain unavailable to the runtime selector.

## Legacy Material

`code_7m_text_only_baseline_v1.json` is an experiment baseline, not a universal
runtime profile. Historical metric specifications and obsolete pipeline
configuration belong under `../archive/`.

Any future activation must update the registry, policy card, profile,
behavioral fixtures, audit trace, and baseline documentation together.
