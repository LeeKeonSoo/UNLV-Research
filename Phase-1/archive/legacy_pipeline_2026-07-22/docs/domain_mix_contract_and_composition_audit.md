# Domain Mix Contract and Composition Audit

## Purpose

Domain composition is now treated as deployment-contract evidence, not as a
universal quality score. The framework can report how a candidate corpus changes
from raw input to curated training arms, but it must not claim a fixed universal
ratio such as code/math/science percentages for every language-model training
run.

## Contract Rule

A deployment contract may declare:

- target domain or capability mix;
- allowed drift from that target;
- minimum retention by domain or source family;
- whether `retain_all` is allowed when no training budget binds;
- required Stage-C benchmarks and guardrails for each domain.

If no target mix is declared, the framework reports observed composition only.
It must not convert observed proportions into a recommendation.

## Current Paper Evidence

The current Block-2 artifact is:

```text
configs/domain_mix_contract_v1.json
-> 219_build_domain_composition_audit.py
-> outputs/validation/domain_composition_audit_report.json
-> outputs/validation/domain_composition_audit_report.md
```

Block 3 connects this composition evidence to Coverage:

```text
configs/coverage_domain_mix_contract_v1.json
-> 220_build_coverage_domain_mix_audit.py
-> outputs/validation/coverage_domain_mix_audit_report.json
-> outputs/validation/coverage_domain_mix_audit_report.md
```

The current contract mode is:

```text
observed_paper_domain_arm_composition
```

This means the report compares paper-facing Code and Math domain arms. It is
not a single joint production mixture and not a certified target mix.

| Domain | Raw packed tokens | Curated packed tokens | Token reduction | Raw share | Curated-arm share | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Code | 980,992 | 385,024 | 60.75% | 46.69% | 27.29% | current five-seed curation-stage pass |
| Math | 1,120,256 | 1,026,048 | 8.41% | 53.31% | 72.71% | repair-only abstain |

## Claim Boundary

Allowed:

```text
The framework audits how curation changes domain-arm composition under a
declared deployment contract.
```

Forbidden:

```text
The framework discovers a universal optimal domain mixture.
```

```text
The current Code/Math arms certify a joint production training mix.
```

```text
Domain mix is an intrinsic data-quality measurement.
```

Utility remains Stage C only and must never become a selector objective.

## Coverage Connection

Coverage now has two distinct responsibilities:

- collapse prevention over observable source/style/path/content/cluster axes;
- domain or capability mix drift auditing when metadata and a declared
  Deployment Contract support the claim.

For the current paper evidence, no target domain mix is declared. Therefore the
Coverage/domain-mix audit allows only observed-composition reporting. It does
not allow target-mix satisfaction, Utility, intrinsic-quality, or universal
domain-ratio claims.
