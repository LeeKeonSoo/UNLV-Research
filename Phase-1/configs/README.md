# Configuration Status

This directory contains several classes of JSON artifacts. File presence alone
does not grant runtime authority.

## Runtime Contracts

The runtime contract surface is anchored by:

- `curation_contract.json`
- `curation_run_contract.example.json`
- `core_policy_registry.json`
- `policy_card_contract.json`
- `policy_cards.json`
- `policy_profiles.json`

Even these files must be checked against the observed call path documented in
`../docs/framework_consistency_baseline.md`; profile identity does not yet
fully determine all run-time rule switches.

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
