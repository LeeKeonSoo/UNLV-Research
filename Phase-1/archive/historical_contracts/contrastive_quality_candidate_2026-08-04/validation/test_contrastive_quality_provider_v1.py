from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from contrastive_quality_provider import (
    ContrastiveProviderError,
    ModelRole,
    ModelScoreBundle,
    ModelScoreObservation,
    NativeTokenizerSnapshot,
    TokenizerCompatibilityRequest,
    build_model_snapshot_manifest,
    build_tokenizer_compatibility_manifest,
    combine_model_score_bundles,
    load_contrastive_provider,
    score_token_ids,
)
from model_provider_contract import load_provider_registry


CONFIG = ROOT / "configs" / "contrastive_quality_provider_qwen3_4b_8b_v1.json"


def _score_bundle(role: ModelRole, nll: tuple[float, float], entropy: tuple[float, float]) -> ModelScoreBundle:
    records = tuple(
        ModelScoreObservation(
            record_uid=f"record-{index}",
            route="code_artifact",
            token_ids_sha256=f"{index + 1}" * 64,
            scored_token_count=127,
            mean_nll=value,
            mean_entropy=entropy[index],
            truncated=False,
        )
        for index, value in enumerate(nll)
    )
    return ModelScoreBundle.create(
        provider_identity_sha256="a" * 64,
        scoring_contract_identity_sha256="9" * 64,
        role=role,
        model_identity_sha256=("b" if role is ModelRole.TARGET else "c") * 64,
        tokenizer_identity_sha256="d" * 64,
        input_artifact_sha256="e" * 64,
        records=records,
        quantization_validation_artifact_sha256=None,
    )


def test_qwen_pair_is_replaceable_but_frozen_by_identity() -> None:
    provider = load_contrastive_provider(CONFIG)
    registry = load_provider_registry(ROOT / "configs" / "model_provider_registry_v1.json")
    registry_provider = next(item for item in registry.providers if item.provider_id == provider.provider_id)

    assert provider.lifecycle == "audit_only"
    assert provider.target.model_id == "Qwen/Qwen3-4B-Base"
    assert provider.reference.model_id == "Qwen/Qwen3-8B-Base"
    assert provider.runtime_authority is False
    assert provider.direct_deletion_authority is False
    assert provider.weighted_quality_formula_used is False
    assert provider.benchmark_outcomes_available is False
    assert provider.utility_available is False
    original_identity = provider.identity_sha256()
    assert registry_provider.implementation_contract_path == CONFIG.relative_to(ROOT).as_posix()
    assert registry_provider.implementation_contract_identity_sha256 == original_identity

    payload = provider.model_dump(mode="json")
    payload["reference"]["model_id"] = "organization/replacement-reference"
    replacement = type(provider).model_validate(payload)
    assert replacement.identity_sha256() != original_identity
    assert replacement.lifecycle == "audit_only"


def test_contrastive_join_requires_identical_records_tokens_and_input() -> None:
    target = _score_bundle(ModelRole.TARGET, (2.0, 1.0), (3.0, 2.0))
    reference = _score_bundle(ModelRole.REFERENCE, (1.5, 1.2), (1.0, 2.5))

    combined = combine_model_score_bundles(target, reference)

    assert combined.provider_identity_sha256 == "a" * 64
    assert combined.scoring_contract_identity_sha256 == "9" * 64
    assert combined.records[0].excess_nll == 0.5
    assert math.isclose(combined.records[1].excess_nll, -0.2)
    assert combined.records[0].reference_entropy == 1.0
    assert combined.records[0].target_nll == 2.0
    assert combined.runtime_authority is False
    assert combined.direct_deletion_authority is False
    assert combined.benchmark_outcomes_read is False
    assert combined.utility_read is False

    changed = ModelScoreBundle.create(
        provider_identity_sha256=reference.provider_identity_sha256,
        scoring_contract_identity_sha256=reference.scoring_contract_identity_sha256,
        role=reference.role,
        model_identity_sha256=reference.model_identity_sha256,
        tokenizer_identity_sha256=reference.tokenizer_identity_sha256,
        input_artifact_sha256=reference.input_artifact_sha256,
        records=(
            reference.records[0].model_copy(update={"token_ids_sha256": "f" * 64}),
            reference.records[1],
        ),
        quantization_validation_artifact_sha256=None,
    )
    try:
        combine_model_score_bundles(target, changed)
    except ContrastiveProviderError as error:
        assert error.reason_code == "contrastive_token_identity_mismatch:record-0"
    else:
        raise AssertionError("Mismatched token identities entered contrastive evidence")


def test_benchmark_feedback_and_weighted_formula_are_not_parseable_inputs() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["benchmark_results"] = {"humaneval_plus": 0.9}
    try:
        load_type = type(load_contrastive_provider(CONFIG))
        load_type.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("Benchmark feedback entered the provider contract")


def test_chunked_scoring_counts_every_target_token_once() -> None:
    import torch

    class UniformToyModel:
        def __call__(self, *, input_ids, past_key_values, use_cache, logits_to_keep):
            assert use_cache is True
            assert logits_to_keep == input_ids.shape[1]
            logits = torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device)
            return SimpleNamespace(logits=logits, past_key_values=object())

    score = score_token_ids(UniformToyModel(), (0, 1, 2, 3, 0), chunk_tokens=2, device="cpu")

    assert score.scored_token_count == 4
    assert math.isclose(score.mean_nll, math.log(4), rel_tol=1e-6)
    assert math.isclose(score.mean_entropy, math.log(4), rel_tol=1e-6)


def test_snapshot_manifest_hashes_every_frozen_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory) / ("1" * 40)
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "weights.bin").write_bytes(b"frozen-weights")

        manifest = build_model_snapshot_manifest("organization/model", snapshot.name, snapshot)

        assert manifest.model_id == "organization/model"
        assert {item.relative_path for item in manifest.files} == {"config.json", "weights.bin"}
        assert all(len(item.sha256) == 64 for item in manifest.files)


def test_shared_tokenizer_manifest_requires_identical_native_tokenizer_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "target"
        reference = Path(directory) / "reference"
        target.mkdir()
        reference.mkdir()
        for snapshot in (target, reference):
            (snapshot / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
            (snapshot / "tokenizer_config.json").write_text('{"eos_token":"<eos>"}', encoding="utf-8")

        request = TokenizerCompatibilityRequest(
            target=NativeTokenizerSnapshot("organization/target", "target-revision", target),
            reference=NativeTokenizerSnapshot("organization/reference", "reference-revision", reference),
            tokenizer_id="organization/target",
            tokenizer_revision="target-revision",
        )
        manifest = build_tokenizer_compatibility_manifest(request)

        assert len(manifest.files) == 2
        assert all(item.target_sha256 == item.reference_sha256 for item in manifest.files)

        (reference / "tokenizer.json").write_text('{"version":"2"}', encoding="utf-8")
        try:
            build_tokenizer_compatibility_manifest(request)
        except ContrastiveProviderError as error:
            assert error.reason_code == "contrastive_native_tokenizer_hash_mismatch:tokenizer.json"
        else:
            raise AssertionError("Incompatible native tokenizers entered one contrastive provider")


if __name__ == "__main__":
    test_qwen_pair_is_replaceable_but_frozen_by_identity()
    test_contrastive_join_requires_identical_records_tokens_and_input()
    test_benchmark_feedback_and_weighted_formula_are_not_parseable_inputs()
    test_chunked_scoring_counts_every_target_token_once()
    test_snapshot_manifest_hashes_every_frozen_file()
    test_shared_tokenizer_manifest_requires_identical_native_tokenizer_files()
    print("[contrastive-quality-provider-v1] replaceable pair and strict evidence join: pass")
