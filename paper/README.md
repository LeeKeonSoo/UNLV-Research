# Paper working notes

Target: IEEE BigData 2026 full-length paper, at most 10 pages in IEEE
two-column format with references counted in the limit.

## Current snapshot

- Working title: **Evidence-Bound Curation: Auditable Membership Decisions for
  Language-Model Pretraining Data**.
- The draft follows a systems/data-curation structure rather than an RQ
  structure.
- `IEEEtran` is used in `conference` mode at the class-default 10 pt size, with
  Times-compatible TeX Gyre Termes/NewTX text and math fonts.
- The compiled manuscript is US Letter, two-column, and 10 pages including
  references. It has no manual margin, font-size, or line-spacing overrides.
- The submission is single-blind, so author names and affiliations remain.
- Local QA currently reports no overfull boxes, undefined references, or
  undefined citations. A final IEEE PDF Checker/PDF eXpress pass is still
  required when the conference exposes its checker or conference ID.

## Central claim and boundary

The paper's central claim is that heterogeneous filter signals should not own
corpus membership directly. Evidence producers retrieve candidates, compare
units, or estimate bounded properties; only a typed, versioned policy may
authorize a membership transition. The implemented roles are closed Validity,
witnessed Redundancy, positive-selection Quality, and veto-only Coverage.

The curation runtime does not read benchmark outcomes, pretraining loss, NLL,
utility estimates, target source mixtures, or a target token budget. Raw,
Normal, and Hard are frozen first and are then evaluated externally by
natural-token continued pretraining. The evidence supports implemented
authority separation and one frozen corpus materialization. It does not support
a universal data-quality claim or a production-scale comparison. Downstream
results qualify only the frozen code-corpus profiles under the declared model,
recipe, and benchmark hierarchy.

## Frozen evidence map

| Paper content | Repository authority |
|---|---|
| Offline Q1--Q4 calibration | `Phase-1/configs/quality_teacher_luna_single_v1.json` |
| Local four-head Quality ranker | `Phase-1/configs/quality_ranker_v1.json` |
| Normal/Hard policy thresholds | `Phase-1/configs/policy_profiles.json` |
| MinHash retrieval and typed redundancy bounds | `Phase-1/configs/redundancy_v2.json` |
| Quality/Coverage formal contract | `Phase-1/docs/quality_coverage_formal_definition.md` |
| Corpus sources and token materialization | `Phase-1/docs/code_7m_corpus_provenance.md` |
| Corpus, policy, and integrity audit | `Phase-1/docs/code_7m_dataset_integrity_audit.md` |
| Frozen external evaluation protocol | `Phase-1/protocols/code_reasoning_primary_amendment_v1.json` |

The manuscript reports frozen facts only: 4,890 Raw records; 7,147 Normal
chunks; 5,859 Hard chunks; 87.70%/72.05% natural-token retention; two/three
witnessed redundancy exclusions; 99/429 Coverage restores; and zero unexplained
extinctions after the complete recheck. MinHash is candidate retrieval only;
the frozen corpus contains no MinHash-only deletion.

The authority contract is expressed through the runtime's existing checks:
complete reason-coded traces, representative survival, Coverage zero-survivor
restoration, forbidden-input isolation, and identity-bound deterministic
replay. The accompanying transition-induction argument applies to accepted
materializations under the stated implementation assumptions; it is not a
claim that arbitrary software defects are impossible.

## Final result surface

The final block delimited by `FINAL_RESULTS_BLOCK_BEGIN` and
`FINAL_RESULTS_BLOCK_END` in `draft.tex` contains the frozen Table V
(`tab:downstream`). It includes:

- all six official benchmark scores in seed order 101/202/303;
- three-seed mean plus sample standard deviation for each trained arm;
- the unweighted primary macro over BigCodeBench Complete, CRUXEval-I,
  CRUXEval-O, and DS-1000 after seed aggregation; and
- HumanEval+ and MBPP+ as mandatory secondary diagnostics.

The frozen machine-readable summary can be regenerated with:

    conda run -n research python Phase-1\external_evaluation\collect_confirmatory_benchmark_results.py --seeds 101 202 303

Use `confirmatory_benchmark_results.json` as the numeric authority and its
Markdown companion as a cross-check. Do not select a best seed, omit a weak
benchmark, alter a curation profile, or add result-direction-dependent prose.
All 60 cells in the final table were recomputed from 42,820 task-level
judgments and passed the artifact-provenance audit.

## Release artifacts

- `Evidence_Bound_Curation_Draft.pdf`: compiled 10-page manuscript.
- `Evidence_Bound_Curation_Overleaf.zip`: self-contained TeX source and figure.
- `Evidence_Bound_Curation_Reproducibility.zip`: compact manifest package with
  frozen protocols, configurations, run manifests, relevant source entry
  points, and SHA-256 hashes. It intentionally excludes corpora, model weights,
  adapters, and task-level generations for size and distribution-rights reasons.
