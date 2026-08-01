# Paper Claim Redefinition

## Claim To Avoid

Do not frame the work as:

- a universal data-quality framework
- an intrinsic semantic quality measurement system
- a production-ready safety, PII, license, or contamination filter
- a guarantee that curation improves every corpus or domain
- a selector that uses downstream Utility or benchmark outcomes

Those claims are not supported by the current evidence.

## Claim To Make

Frame the work as a deployment-conditioned LM training-data curation framework.
The framework receives a candidate corpus and a deployment contract, separates
Core responsibilities from Metric implementations and Policy decisions, and
returns one of three outcomes:

- accept a curated training release
- reject or quarantine unusable/risky data
- abstain when Stage-C evidence is insufficient or negative

The contribution is the operational control structure:

- Stage 0 performs ingestion, normalization, and risk quarantine
- Stage A applies chunk-level hard gates
- Stage B performs optional budget allocation using pre-outcome evidence only
- Stage C validates downstream training behavior
- Utility is measured only in Stage C and is never a Stage-B objective

## Current Evidence

- Code is a historical positive natural-budget case: curated v2 reduces packed
  training tokens by 60.8%, improves heldout NLL from `1.210000` to `1.201043`,
  and improves same-protocol EvalPlus macro pass rate from `51.0649%` to
  `57.8682%`. Because these artifacts predate the current Stage-A implementation
  fingerprint, they require a current-framework rerun before confirmatory use.
- Math is the negative natural-budget case: selector v2 reduces packed training
  tokens by 44.1% but worsens heldout NLL from `1.495650` to `1.527065`.
- Production release is still blocked by incomplete guardrails.

## Suggested Paper Wording

This paper proposes a curation-stage control framework for language-model
training data. The framework does not claim to measure intrinsic data quality
or guarantee all-domain improvement. Instead, it defines auditable stage
boundaries for risk quarantine, hard usability gating, optional budgeted
selection, and downstream validation. The current evidence contains one
positive code-domain validation and one negative math-domain validation,
supporting the claim that the framework can produce bounded accept/reject/
abstain decisions while exposing when a domain-specific selector is not yet
validated.
