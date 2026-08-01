# Archive Boundary

Everything under this directory is historical material preserved for provenance
and forensic comparison. It is not part of the current runtime surface and must
not be imported by current curation modules.

- `legacy_pipeline_2026-07-22/` contains the pre-reduction pipeline and related
  utilities.
- `historical_contracts/` contains superseded dataset and metric contracts that
  are not referenced by the current runtime.
- Documentation snapshots preserve earlier claims and terminology.
- Historical outputs may explain prior results but cannot establish current
  behavior without an explicit replay protocol.

Do not repair or modernize archived code in place. If an old idea is revived,
write a current candidate contract, tests, reason-code audit, and promotion
decision outside the archive.
