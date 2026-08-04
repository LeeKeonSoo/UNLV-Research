# Protocol Status

Protocols in this directory govern development, ablation, data materialization,
training, or downstream evaluation. They are not curation policy and are not
selector-visible inputs.

## Development Protocols

Development and calibration protocols may compare candidate rules, fixtures,
compression, representation shifts, and false-positive behavior. Their results
can nominate a frozen policy but cannot silently change runtime behavior.

## Confirmatory Protocols

Confirmatory and benchmark protocols operate on already-curated artifacts.
Natural-budget training arms, NLL, EvalPlus, and other benchmark results are
external evidence. They validate a frozen policy after selection and must be
record-disjoint from development data where the protocol requires it.

## Historical Protocols

Protocols for obsolete selectors, forced token budgets, retired Stage-C
Utility experiments, or superseded profiles remain reproducibility records
only. Their presence does not authorize reuse in current claims.

Before running any protocol, verify its input hashes, model revision, dataset
snapshot, seed set, output location, and status against
`../docs/framework_consistency_baseline.md`.

The retired loss-gap source-pool protocols are preserved under
`../archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.
They are not current development inputs.
