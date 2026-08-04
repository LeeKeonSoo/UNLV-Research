# Retired Contrastive Quality Candidate

This directory preserves the target/reference loss-gap Quality experiment as
historical negative evidence. The experiment did not identify a stable,
route-general deletion boundary and was retired on 2026-08-04 when the current
Quality direction was fixed to the three-teacher Q1-Q4 Quality Ranker.

The files here are provenance only:

- `src/` contains the retired scorer, audit, gate, and source-pool modules.
- `scripts/` contains retired collection and scoring entry points.
- `configs/` and `protocols/` contain the frozen candidate contracts.
- `validation/` contains candidate tests and frozen negative evidence.
- `docs/` contains the superseded completion plan and redesign checkpoint.

Current runtime, profiles, provider registries, and release gates must not
import or reference these files. Revival requires a new candidate ID, current
contract, new fixtures, and an independent promotion decision.
