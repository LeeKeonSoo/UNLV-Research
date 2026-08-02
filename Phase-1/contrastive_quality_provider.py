from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
type JsonValue = str | int | float | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ContrastiveProviderError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class ModelRole(str, Enum):
    TARGET = "target"
    REFERENCE = "reference"


class Precision(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT4 = "int4"


class ContrastiveModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: ModelRole
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_sha256: Sha256 | None
    precision: Precision
    quantization_validation_artifact_sha256: Sha256 | None

    @model_validator(mode="after")
    def validate_precision_evidence(self) -> "ContrastiveModelSpec":
        if self.precision in {Precision.INT8, Precision.INT4} and self.quantization_validation_artifact_sha256 is None:
            return self
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class ContrastiveTokenizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tokenizer_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_sha256: Sha256 | None
    add_special_tokens: Literal[False]
    append_eos_per_record: Literal[True]

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class ContrastiveScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    maximum_context_tokens: int = Field(ge=128)
    inference_chunk_tokens: int = Field(ge=1)
    minimum_scored_tokens: int = Field(ge=1)
    loss_unit: Literal["nats_per_nonpadding_target_token"]
    entropy_unit: Literal["nats_per_next_token_distribution"]
    output_evidence: tuple[str, ...]
    scalar_quality_score_emitted: Literal[False]
    threshold_decision_emitted: Literal[False]

    @model_validator(mode="after")
    def validate_evidence_tuple(self) -> "ContrastiveScoringSpec":
        expected = {
            "target_nll",
            "reference_nll",
            "excess_nll",
            "target_entropy",
            "reference_entropy",
        }
        if set(self.output_evidence) != expected or len(self.output_evidence) != len(expected):
            raise ContrastiveProviderError("contrastive_output_evidence_incomplete")
        if self.inference_chunk_tokens > self.maximum_context_tokens:
            raise ContrastiveProviderError("contrastive_inference_chunk_exceeds_context")
        return self


class ContrastiveQualityProvider(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["contrastive-quality-provider-v1"]
    provider_id: str = Field(min_length=1)
    lifecycle: Literal["audit_only"]
    target: ContrastiveModelSpec
    reference: ContrastiveModelSpec
    tokenizer: ContrastiveTokenizerSpec
    scoring: ContrastiveScoringSpec
    supported_routes: tuple[str, ...]
    unknown_mixed_ood_action: Literal["abstain_retain"]
    weighted_quality_formula_used: Literal[False]
    benchmark_outcomes_available: Literal[False]
    utility_available: Literal[False]
    runtime_authority: Literal[False]
    direct_deletion_authority: Literal[False]
    replacement_invalidates_calibration: Literal[True]

    @model_validator(mode="after")
    def validate_pair(self) -> "ContrastiveQualityProvider":
        if self.target.role is not ModelRole.TARGET or self.reference.role is not ModelRole.REFERENCE:
            raise ContrastiveProviderError("contrastive_model_roles_invalid")
        if not self.supported_routes or len(set(self.supported_routes)) != len(self.supported_routes):
            raise ContrastiveProviderError("contrastive_supported_routes_invalid")
        return self

    def identity_sha256(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "lifecycle",
                "runtime_authority",
                "direct_deletion_authority",
            },
        )
        return hash_json(payload)


class ModelScoreObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_uid: str = Field(min_length=1)
    route: str = Field(min_length=1)
    token_ids_sha256: Sha256
    scored_token_count: int = Field(ge=1)
    mean_nll: float = Field(ge=0.0)
    mean_entropy: float = Field(ge=0.0)
    truncated: bool

    @model_validator(mode="after")
    def validate_finite_metrics(self) -> "ModelScoreObservation":
        if not math.isfinite(self.mean_nll) or not math.isfinite(self.mean_entropy):
            raise ContrastiveProviderError("contrastive_nonfinite_model_score")
        return self


class ModelScoreBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["contrastive-model-score-bundle-v1"]
    provider_identity_sha256: Sha256
    scoring_contract_identity_sha256: Sha256
    role: ModelRole
    model_identity_sha256: Sha256
    tokenizer_identity_sha256: Sha256
    input_artifact_sha256: Sha256
    records: tuple[ModelScoreObservation, ...]
    quantization_validation_artifact_sha256: Sha256 | None
    bundle_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    utility_read: Literal[False] = False

    @model_validator(mode="after")
    def validate_bundle(self) -> "ModelScoreBundle":
        if not self.records or len({item.record_uid for item in self.records}) != len(self.records):
            raise ContrastiveProviderError("contrastive_model_score_records_invalid")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if self.bundle_sha256 != hash_json(payload):
            raise ContrastiveProviderError("contrastive_model_score_bundle_hash_mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        provider_identity_sha256: str,
        scoring_contract_identity_sha256: str,
        role: ModelRole,
        model_identity_sha256: str,
        tokenizer_identity_sha256: str,
        input_artifact_sha256: str,
        records: tuple[ModelScoreObservation, ...],
        quantization_validation_artifact_sha256: str | None,
    ) -> "ModelScoreBundle":
        payload = {
            "schema_version": "contrastive-model-score-bundle-v1",
            "provider_identity_sha256": provider_identity_sha256,
            "scoring_contract_identity_sha256": scoring_contract_identity_sha256,
            "role": role.value,
            "model_identity_sha256": model_identity_sha256,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
            "input_artifact_sha256": input_artifact_sha256,
            "records": [item.model_dump(mode="json") for item in records],
            "quantization_validation_artifact_sha256": quantization_validation_artifact_sha256,
            "benchmark_outcomes_read": False,
            "utility_read": False,
        }
        return cls(bundle_sha256=hash_json(payload), **payload)


class ContrastiveQualityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_uid: str
    route: str
    token_ids_sha256: Sha256
    scored_token_count: int = Field(ge=1)
    target_nll: float
    reference_nll: float
    excess_nll: float
    target_entropy: float
    reference_entropy: float
    truncated: bool


class ContrastiveEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["contrastive-quality-evidence-bundle-v1"]
    provider_identity_sha256: Sha256
    scoring_contract_identity_sha256: Sha256
    target_bundle_sha256: Sha256
    reference_bundle_sha256: Sha256
    tokenizer_identity_sha256: Sha256
    input_artifact_sha256: Sha256
    records: tuple[ContrastiveQualityObservation, ...]
    evidence_bundle_sha256: Sha256
    scalar_quality_score_emitted: Literal[False] = False
    threshold_decision_emitted: Literal[False] = False
    runtime_authority: Literal[False] = False
    direct_deletion_authority: Literal[False] = False
    benchmark_outcomes_read: Literal[False] = False
    utility_read: Literal[False] = False

    @model_validator(mode="after")
    def validate_bundle(self) -> "ContrastiveEvidenceBundle":
        payload = self.model_dump(mode="json", exclude={"evidence_bundle_sha256"})
        if self.evidence_bundle_sha256 != hash_json(payload):
            raise ContrastiveProviderError("contrastive_evidence_bundle_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class TokenScore:
    scored_token_count: int
    mean_nll: float
    mean_entropy: float


def score_token_ids(
    model: object,
    token_ids: tuple[int, ...],
    *,
    chunk_tokens: int,
    device: str,
) -> TokenScore:
    import torch
    import torch.nn.functional as functional

    if len(token_ids) < 2:
        raise ContrastiveProviderError("contrastive_too_few_token_ids")
    if chunk_tokens < 1:
        raise ContrastiveProviderError("contrastive_chunk_tokens_invalid")
    past_key_values = None
    previous_next_logits = None
    nll_sum = 0.0
    entropy_sum = 0.0
    scored = 0
    with torch.inference_mode():
        for start in range(0, len(token_ids), chunk_tokens):
            chunk = token_ids[start : start + chunk_tokens]
            input_ids = torch.tensor((chunk,), dtype=torch.long, device=device)
            outputs: SimpleNamespace = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=len(chunk),
            )
            logits = outputs.logits[0].float()
            if logits.shape[0] != len(chunk):
                raise ContrastiveProviderError("contrastive_model_logit_length_mismatch")
            if previous_next_logits is None:
                prediction_logits = logits[:-1]
                targets = input_ids[0, 1:]
            else:
                prediction_logits = torch.cat((previous_next_logits.unsqueeze(0), logits[:-1]), dim=0)
                targets = input_ids[0]
            if targets.numel():
                log_probabilities = functional.log_softmax(prediction_logits, dim=-1)
                token_nll = functional.nll_loss(log_probabilities, targets, reduction="sum")
                probabilities = log_probabilities.exp()
                token_entropy = -(probabilities * log_probabilities).sum(dim=-1).sum()
                nll_sum += float(token_nll.item())
                entropy_sum += float(token_entropy.item())
                scored += int(targets.numel())
            previous_next_logits = logits[-1]
            past_key_values = outputs.past_key_values
    if scored != len(token_ids) - 1:
        raise ContrastiveProviderError("contrastive_scored_token_count_mismatch")
    return TokenScore(scored, nll_sum / scored, entropy_sum / scored)


def combine_model_score_bundles(
    target: ModelScoreBundle,
    reference: ModelScoreBundle,
) -> ContrastiveEvidenceBundle:
    if target.role is not ModelRole.TARGET or reference.role is not ModelRole.REFERENCE:
        raise ContrastiveProviderError("contrastive_score_bundle_roles_invalid")
    shared_fields = (
        (target.provider_identity_sha256, reference.provider_identity_sha256, "provider_identity"),
        (
            target.scoring_contract_identity_sha256,
            reference.scoring_contract_identity_sha256,
            "scoring_contract_identity",
        ),
        (target.tokenizer_identity_sha256, reference.tokenizer_identity_sha256, "tokenizer_identity"),
        (target.input_artifact_sha256, reference.input_artifact_sha256, "input_artifact"),
    )
    for left, right, name in shared_fields:
        if left != right:
            raise ContrastiveProviderError(f"contrastive_{name}_mismatch")
    target_by_uid = {item.record_uid: item for item in target.records}
    reference_by_uid = {item.record_uid: item for item in reference.records}
    if set(target_by_uid) != set(reference_by_uid):
        raise ContrastiveProviderError("contrastive_record_set_mismatch")
    observations: list[ContrastiveQualityObservation] = []
    for uid in sorted(target_by_uid):
        target_item = target_by_uid[uid]
        reference_item = reference_by_uid[uid]
        if target_item.token_ids_sha256 != reference_item.token_ids_sha256:
            raise ContrastiveProviderError(f"contrastive_token_identity_mismatch:{uid}")
        if target_item.scored_token_count != reference_item.scored_token_count:
            raise ContrastiveProviderError(f"contrastive_token_count_mismatch:{uid}")
        if target_item.route != reference_item.route:
            raise ContrastiveProviderError(f"contrastive_route_mismatch:{uid}")
        observations.append(
            ContrastiveQualityObservation(
                record_uid=uid,
                route=target_item.route,
                token_ids_sha256=target_item.token_ids_sha256,
                scored_token_count=target_item.scored_token_count,
                target_nll=target_item.mean_nll,
                reference_nll=reference_item.mean_nll,
                excess_nll=target_item.mean_nll - reference_item.mean_nll,
                target_entropy=target_item.mean_entropy,
                reference_entropy=reference_item.mean_entropy,
                truncated=target_item.truncated or reference_item.truncated,
            )
        )
    payload = {
        "schema_version": "contrastive-quality-evidence-bundle-v1",
        "provider_identity_sha256": target.provider_identity_sha256,
        "scoring_contract_identity_sha256": target.scoring_contract_identity_sha256,
        "target_bundle_sha256": target.bundle_sha256,
        "reference_bundle_sha256": reference.bundle_sha256,
        "tokenizer_identity_sha256": target.tokenizer_identity_sha256,
        "input_artifact_sha256": target.input_artifact_sha256,
        "records": [item.model_dump(mode="json") for item in observations],
        "scalar_quality_score_emitted": False,
        "threshold_decision_emitted": False,
        "runtime_authority": False,
        "direct_deletion_authority": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    return ContrastiveEvidenceBundle(evidence_bundle_sha256=hash_json(payload), **payload)


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contrastive_provider(path: Path) -> ContrastiveQualityProvider:
    return ContrastiveQualityProvider.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "ContrastiveEvidenceBundle",
    "ContrastiveProviderError",
    "ContrastiveQualityProvider",
    "ModelRole",
    "ModelScoreBundle",
    "ModelScoreObservation",
    "TokenScore",
    "combine_model_score_bundles",
    "load_contrastive_provider",
    "score_token_ids",
]
