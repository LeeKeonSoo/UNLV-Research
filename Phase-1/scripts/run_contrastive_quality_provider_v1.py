#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from contrastive_quality_provider import (
    ContrastiveProviderError,
    ModelRole,
    ModelScoreBundle,
    ModelScoreObservation,
    Precision,
    combine_model_score_bundles,
    load_contrastive_provider,
    score_token_ids,
)
from model_provider_contract import ProviderRole, load_provider_registry


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
    if args.route not in provider.supported_routes:
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
                    route=args.route,
                    token_ids_sha256=_token_hash(token_ids),
                    scored_token_count=score.scored_token_count,
                    mean_nll=score.mean_nll,
                    mean_entropy=score.mean_entropy,
                    truncated=truncated,
                )
            )
            if args.progress_every and len(records) % args.progress_every == 0:
                print(f"[contrastive-score] role={role.value} records={len(records)} line={line_number}", flush=True)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or combine one replaceable contrastive Quality provider pair.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--provider", type=Path, required=True)
    score.add_argument("--provider-registry", type=Path, required=True)
    score.add_argument("--role", choices=[item.value for item in ModelRole], required=True)
    score.add_argument("--model-path", type=Path, required=True)
    score.add_argument("--tokenizer-path", type=Path, required=True)
    score.add_argument("--device", required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--route", required=True)
    score.add_argument("--id-field", default="fixture_id")
    score.add_argument("--text-field", default="text")
    score.add_argument("--progress-every", type=int, default=25)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(run=_score)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--target", type=Path, required=True)
    combine.add_argument("--reference", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(run=_combine)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
