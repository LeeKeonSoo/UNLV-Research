# Code 5M Corpus Collection Protocol

## Purpose

This is a future natural-budget validation corpus for Qwen3-4B continued pretraining. The target Raw arm has 5,242,880 packed tokens and 5,210,112 complete gradient-accumulation training tokens. It replaces neither the current 0.69M-token pilot nor its recorded results.

## Why These Criteria

The GPT-2 WebText `karma >= 3` rule was a web-page discovery heuristic, not a universal quality definition. It is not transferred to source-code inclusion. In code, source admission is limited to rights, provenance, file safety, deduplication, concentration, and benchmark-contamination controls.

The Stack reports that permissive licensing and near-deduplication can support code-model evaluation, and that near-deduplication improves downstream performance. SantaCoder reports that a GitHub stars cutoff can harm code-model performance. Therefore stars are preserved only as an optional post-collection audit stratum and cannot control admission or Stage B.

## Frozen Source Rules

- Primary Raw-like source: `bigcode/the-stack-v2`, Python, revision `7408bfbcfd48e5833d62fd3dba48afd20d109473`.
- Primary source admission: permissive SPDX license, non-generated source file, 256 bytes to 256 KiB, per-repository cap, exact and near-duplicate removal, PII/secrets quarantine, and contamination/uncertainty quarantine.
- Reference context: immutable-commit snapshots from a permissively licensed Python repository allowlist. Its source tier remains audit-only and is blinded from Stage B.
- Mixture before Stage A: 70% Raw-like and 30% reference context. The post-Stage-A/B composition is observed and reported; it is not forced to retain this ratio.

## Access Gate

The Stack v2 is gated on Hugging Face. Collection may begin only after the authenticated account has accepted its dataset access terms. The frozen protocol records the requested revision and does not silently substitute another source.

## Required Evidence Before Training

1. Immutable source and transformation manifests with hashes.
2. License/provenance, deduplication, repository concentration, PII/secrets, and contamination audits.
3. Source-tier and benchmark/Utility isolation audit for Stage B.
4. Frozen train, development, and confirmatory splits before Stage C outcomes.

## Planned Stage C Design

- Three paired seeds: `11`, `23`, and `37`.
- Each seed trains the Raw natural arm on its complete frozen token budget and the Curated natural arm on its own complete frozen token budget.
- Code benchmarks are the primary decision evidence. Held-out code NLL is diagnostic only and does not determine the final pass/fail decision.
- Stage-B policy is frozen before any training or benchmark outcome is read.
