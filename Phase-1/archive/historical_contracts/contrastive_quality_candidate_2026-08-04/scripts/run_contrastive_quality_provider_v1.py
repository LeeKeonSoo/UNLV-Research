#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from contrastive_quality_provider import (
    ContrastiveEvidenceBundle,
    ContrastiveProviderError,
    FrozenModelSnapshotManifest,
    FrozenTokenizerCompatibilityManifest,
    ModelRole,
    ModelScoreBundle,
    ModelScoreObservation,
    NativeTokenizerSnapshot,
    Precision,
    TokenizerCompatibilityRequest,
    build_model_snapshot_manifest,
    build_tokenizer_compatibility_manifest,
    combine_model_score_bundles,
    load_contrastive_provider,
    score_token_ids,
)
from contrastive_quality_audit import ContrastiveAuditInputs, build_contrastive_quality_audit
from model_provider_contract import ProviderRole, load_provider_registry


ROUTE_BY_DOMAIN = {
    "code": "code_artifact",
    "math": "mathematical_content",
    "general": "general_prose",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_hash(token_ids: tuple[int, ...]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_snapshot(path: Path, revision: str, kind: str) -> None:
    if not path.is_dir():
        raise ContrastiveProviderError(f"contrastive_{kind}_snapshot_missing")
    if path.name != revision:
        raise ContrastiveProviderError(f"contrastive_{kind}_revision_mismatch")


def _validate_frozen_manifest(
    manifest_path: Path | None,
    *,
    expected_artifact_sha256: str | None,
    expected_model_id: str,
    expected_revision: str,
) -> None:
    if expected_artifact_sha256 is None:
        return
    if manifest_path is None or not manifest_path.is_file():
        raise ContrastiveProviderError("contrastive_snapshot_manifest_missing")
    manifest = FrozenModelSnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.model_id != expected_model_id
        or manifest.revision != expected_revision
        or manifest.artifact_sha256 != expected_artifact_sha256
    ):
        raise ContrastiveProviderError("contrastive_snapshot_manifest_identity_mismatch")


def _load_model(path: Path, precision: Precision, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    common: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    if precision is Precision.INT8:
        common["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        common["device_map"] = {"": device}
    elif precision is Precision.INT4:
        common["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        common["device_map"] = {"": device}
    else:
        dtype = {
            Precision.BFLOAT16: torch.bfloat16,
            Precision.FLOAT16: torch.float16,
            Precision.FLOAT32: torch.float32,
        }[precision]
        common["dtype"] = dtype
        common["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(path, **common)
    model.eval()
    return model


def _score(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    provider = load_contrastive_provider(args.provider)
    registry = load_provider_registry(args.provider_registry)
    registry_provider = next(
        (item for item in registry.providers if item.provider_id == provider.provider_id),
        None,
    )
    if registry_provider is None or registry_provider.role is not ProviderRole.QUALITY:
        raise ContrastiveProviderError("contrastive_registry_provider_missing")
    if registry_provider.implementation_contract_identity_sha256 != provider.identity_sha256():
        raise ContrastiveProviderError("contrastive_registry_contract_identity_mismatch")
    role = ModelRole(args.role)
    spec = provider.target if role is ModelRole.TARGET else provider.reference
    model_path: Path = args.model_path
    tokenizer_path: Path = args.tokenizer_path
    input_path: Path = args.input
    _validate_snapshot(model_path, spec.revision, "model")
    _validate_snapshot(tokenizer_path, provider.tokenizer.revision, "tokenizer")
    _validate_frozen_manifest(
        args.model_snapshot_manifest,
        expected_artifact_sha256=spec.artifact_sha256,
        expected_model_id=spec.model_id,
        expected_revision=spec.revision,
    )
    _validate_frozen_manifest(
        args.tokenizer_snapshot_manifest,
        expected_artifact_sha256=provider.tokenizer.artifact_sha256,
        expected_model_id=provider.tokenizer.tokenizer_id,
        expected_revision=provider.tokenizer.revision,
    )
    if (args.route is None) == (args.route_field is None):
        raise ContrastiveProviderError("contrastive_route_source_ambiguous")
    if args.route is not None and args.route not in provider.supported_routes:
        raise ContrastiveProviderError("contrastive_route_unsupported")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ContrastiveProviderError("contrastive_tokenizer_eos_missing")
    model = _load_model(model_path, spec.precision, args.device)
    records: list[ModelScoreObservation] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row[args.id_field])
            text = str(row[args.text_field])
            route = args.route if args.route is not None else str(row[args.route_field])
            if route not in provider.supported_routes:
                raise ContrastiveProviderError(f"contrastive_route_unsupported:{uid}")
            token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            truncated = len(token_ids) + 1 > provider.scoring.maximum_context_tokens
            token_ids = token_ids[: provider.scoring.maximum_context_tokens - 1] + (tokenizer.eos_token_id,)
            if len(token_ids) - 1 < provider.scoring.minimum_scored_tokens:
                continue
            score = score_token_ids(
                model,
                token_ids,
                chunk_tokens=provider.scoring.inference_chunk_tokens,
                device=args.device,
            )
            records.append(
                ModelScoreObservation(
                    record_uid=uid,
                    route=route,
                    token_ids_sha256=_token_hash(token_ids),
                    scored_token_count=score.scored_token_count,
                    mean_nll=score.mean_nll,
                    mean_entropy=score.mean_entropy,
                    truncated=truncated,
                )
            )
            if args.progress_every and len(records) % args.progress_every == 0:
                print(f"[contrastive-score] role={role.value} records={len(records)} line={line_number}", flush=True)
            if args.max_records is not None and len(records) >= args.max_records:
                break
    bundle = ModelScoreBundle.create(
        provider_identity_sha256=registry_provider.identity_sha256(),
        scoring_contract_identity_sha256=provider.identity_sha256(),
        role=role,
        model_identity_sha256=spec.identity_sha256(),
        tokenizer_identity_sha256=provider.tokenizer.identity_sha256(),
        input_artifact_sha256=_sha256_file(input_path),
        records=tuple(records),
        quantization_validation_artifact_sha256=spec.quantization_validation_artifact_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json() + "\n", encoding="utf-8")
    print(f"[contrastive-score] role={role.value} records={len(records)} output={args.output}")
    return 0


def _combine(args: argparse.Namespace) -> int:
    target = ModelScoreBundle.model_validate_json(args.target.read_text(encoding="utf-8"))
    reference = ModelScoreBundle.model_validate_json(args.reference.read_text(encoding="utf-8"))
    combined = combine_model_score_bundles(target, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(combined.model_dump_json() + "\n", encoding="utf-8")
    print(f"[contrastive-combine] records={len(combined.records)} output={args.output}")
    return 0


def _sample_key(seed: str, slice_id: str, parent_record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{slice_id}:{parent_record_id}".encode()).hexdigest()


def _sample(args: argparse.Namespace) -> int:
    inventory_path: Path = args.inventory_manifest
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "admitted" or inventory.get("blocker_codes"):
        raise ContrastiveProviderError("contrastive_inventory_not_admitted")
    output_rows: list[dict[str, Any]] = []
    slice_reports: list[dict[str, Any]] = []
    for item in inventory["slices"]:
        if item.get("status") != "materialized":
            raise ContrastiveProviderError(f"contrastive_slice_not_materialized:{item['slice_id']}")
        artifact = Path(item["artifact_path"])
        if _sha256_file(artifact) != item["artifact_sha256"]:
            raise ContrastiveProviderError(f"contrastive_slice_hash_mismatch:{item['slice_id']}")
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with artifact.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_parent[str(row["parent_record_id"])].append(row)
        selected_parents = tuple(
            sorted(
                by_parent,
                key=lambda parent: _sample_key(args.seed, item["slice_id"], parent),
            )[: args.parents_per_slice]
        )
        if len(selected_parents) != args.parents_per_slice:
            raise ContrastiveProviderError(f"contrastive_slice_parent_shortfall:{item['slice_id']}")
        selected_rows = [row for parent in selected_parents for row in by_parent[parent]]
        route = ROUTE_BY_DOMAIN[item["domain"]]
        for row in selected_rows:
            row["contrastive_domain"] = item["domain"]
            row["contrastive_route"] = route
            row["contrastive_scenario"] = item["scenario"]
            row["contrastive_source_id"] = item["base_source_id"]
            output_rows.append(row)
        slice_reports.append(
            {
                "slice_id": item["slice_id"],
                "domain": item["domain"],
                "scenario": item["scenario"],
                "source_id": item["base_source_id"],
                "selected_parent_count": len(selected_parents),
                "selected_record_count": len(selected_rows),
                "selected_parent_ids_sha256": hashlib.sha256(
                    json.dumps(selected_parents, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    output_rows.sort(key=lambda row: (row["contrastive_domain"], row["contrastive_scenario"], row["fixture_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "contrastive-quality-development-sample-v1",
        "status": "frozen_development_sample",
        "inventory_manifest_path": str(inventory_path),
        "inventory_manifest_file_sha256": _sha256_file(inventory_path),
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "selection_seed": args.seed,
        "parents_per_slice": args.parents_per_slice,
        "slices": slice_reports,
        "output_artifact_path": str(args.output),
        "output_artifact_sha256": _sha256_file(args.output),
        "selected_record_count": len(output_rows),
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "selector_membership_mutated": False,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[contrastive-sample] records={len(output_rows)} output={args.output}")
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    manifest = build_model_snapshot_manifest(args.model_id, args.revision, args.snapshot_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(item.size_bytes for item in manifest.files)
    print(
        f"[contrastive-snapshot] files={len(manifest.files)} bytes={total_bytes} "
        f"artifact_sha256={manifest.artifact_sha256} output={args.output}"
    )
    return 0


def _audit(args: argparse.Namespace) -> int:
    provider = load_contrastive_provider(args.provider)
    evidence = ContrastiveEvidenceBundle.model_validate_json(args.evidence.read_text(encoding="utf-8"))
    sample_rows = tuple(
        json.loads(line)
        for line in args.sample.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    required_routes = tuple(args.required_route)
    effect_bins = {route: 0 for route in required_routes}
    compatibility_manifest = None
    if args.tokenizer_compatibility_manifest is not None:
        compatibility_manifest = FrozenTokenizerCompatibilityManifest.model_validate_json(
            args.tokenizer_compatibility_manifest.read_text(encoding="utf-8")
        )
    report = build_contrastive_quality_audit(
        ContrastiveAuditInputs(
            provider=provider,
            evidence=evidence,
            sample_rows=sample_rows,
            sample_artifact_sha256=_sha256_file(args.sample),
            required_routes=required_routes,
            minimum_source_groups_per_route=args.minimum_source_groups_per_route,
            empirical_effect_bins_by_route=effect_bins,
            common_baseline_artifact_sha256=args.common_baseline_artifact_sha256,
            provider_training_disjointness_artifact_sha256=(
                args.provider_training_disjointness_artifact_sha256
            ),
            tokenizer_compatibility_manifest=compatibility_manifest,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"[contrastive-audit] status={report.status} records={report.scored_record_count} "
        f"blockers={len(report.blocker_codes)} output={args.output}"
    )
    return 0


def _tokenizer_compatibility(args: argparse.Namespace) -> int:
    request = TokenizerCompatibilityRequest(
        target=NativeTokenizerSnapshot(
            args.target_model_id,
            args.target_revision,
            args.target_snapshot,
        ),
        reference=NativeTokenizerSnapshot(
            args.reference_model_id,
            args.reference_revision,
            args.reference_snapshot,
        ),
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
    )
    manifest = build_tokenizer_compatibility_manifest(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"[contrastive-tokenizer-compatibility] files={len(manifest.files)} "
        f"artifact_sha256={manifest.artifact_sha256} output={args.output}"
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or combine one replaceable contrastive Quality provider pair.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--provider", type=Path, required=True)
    score.add_argument("--provider-registry", type=Path, required=True)
    score.add_argument("--role", choices=[item.value for item in ModelRole], required=True)
    score.add_argument("--model-path", type=Path, required=True)
    score.add_argument("--model-snapshot-manifest", type=Path)
    score.add_argument("--tokenizer-path", type=Path, required=True)
    score.add_argument("--tokenizer-snapshot-manifest", type=Path)
    score.add_argument("--device", required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--route")
    score.add_argument("--route-field")
    score.add_argument("--id-field", default="fixture_id")
    score.add_argument("--text-field", default="text")
    score.add_argument("--progress-every", type=_positive_int, default=25)
    score.add_argument("--max-records", type=_positive_int)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(run=_score)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--target", type=Path, required=True)
    combine.add_argument("--reference", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(run=_combine)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--inventory-manifest", type=Path, required=True)
    sample.add_argument("--parents-per-slice", type=_positive_int, default=50)
    sample.add_argument("--seed", default="contrastive-quality-development-v1")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--manifest", type=Path, required=True)
    sample.set_defaults(run=_sample)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--model-id", required=True)
    snapshot.add_argument("--revision", required=True)
    snapshot.add_argument("--snapshot-path", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.set_defaults(run=_snapshot)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--provider", type=Path, required=True)
    audit.add_argument("--evidence", type=Path, required=True)
    audit.add_argument("--sample", type=Path, required=True)
    audit.add_argument("--required-route", action="append", required=True)
    audit.add_argument("--minimum-source-groups-per-route", type=_positive_int, default=3)
    audit.add_argument("--common-baseline-artifact-sha256")
    audit.add_argument("--provider-training-disjointness-artifact-sha256")
    audit.add_argument("--tokenizer-compatibility-manifest", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(run=_audit)
    compatibility = subparsers.add_parser("tokenizer-compatibility")
    compatibility.add_argument("--target-model-id", required=True)
    compatibility.add_argument("--target-revision", required=True)
    compatibility.add_argument("--target-snapshot", type=Path, required=True)
    compatibility.add_argument("--reference-model-id", required=True)
    compatibility.add_argument("--reference-revision", required=True)
    compatibility.add_argument("--reference-snapshot", type=Path, required=True)
    compatibility.add_argument("--tokenizer-id", required=True)
    compatibility.add_argument("--tokenizer-revision", required=True)
    compatibility.add_argument("--output", type=Path, required=True)
    compatibility.set_defaults(run=_tokenizer_compatibility)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
