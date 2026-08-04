from __future__ import annotations

from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PanelContractError(RuntimeError):
    """Raised when teacher-panel evidence violates the frozen contract."""


class TeacherLocation(str, Enum):
    NVIDIA_BUILD = "nvidia_build"
    LOCAL = "local"


class PolicyDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class PanelDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class TeacherSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_id: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    location: TeacherLocation
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    model_card_url: str = Field(min_length=1)
    endpoint_base_url: str | None
    api_key_environment_variable: str | None
    reasoning_mode: Literal["disabled", "bounded"]
    inference_precision: Literal["endpoint_managed", "bitsandbytes_int8"]

    @model_validator(mode="after")
    def validate_location_contract(self) -> "TeacherSpec":
        match self.location:
            case TeacherLocation.NVIDIA_BUILD:
                if self.endpoint_base_url is None or self.api_key_environment_variable is None:
                    raise PanelContractError("NVIDIA Build teachers require endpoint and API-key variable")
                if self.inference_precision != "endpoint_managed":
                    raise PanelContractError("NVIDIA Build teachers must use endpoint-managed precision")
            case TeacherLocation.LOCAL:
                if self.endpoint_base_url is not None or self.api_key_environment_variable is not None:
                    raise PanelContractError("Local teachers cannot declare a hosted endpoint or API key")
                if self.inference_precision != "bitsandbytes_int8":
                    raise PanelContractError("The frozen local teacher requires bitsandbytes int8 inference")
            case unreachable:
                assert_never(unreachable)
        return self


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    fail_boundary: str = Field(min_length=1)
    abstain_boundary: str = Field(min_length=1)


class FixtureMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policies: int = Field(gt=0)
    routes: int = Field(gt=0)
    fixture_classes: int = Field(gt=0)
    samples_per_cell: int = Field(gt=0)
    total: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_total(self) -> "FixtureMatrix":
        calculated = self.policies * self.routes * self.fixture_classes * self.samples_per_cell
        if calculated != self.total:
            raise PanelContractError("Fixture-matrix dimensions must multiply to the declared total")
        return self


class PromotionGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protected_fixture_count: int = Field(ge=800)
    confidence: Literal[0.95]
    normal_false_removal_upper_bound: Literal[0.005]
    hard_false_removal_upper_bound: Literal[0.02]
    smoke_suite_may_activate_runtime: Literal[False]


class ResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_enum: tuple[Literal["pass"], Literal["fail"], Literal["abstain"]]
    reason_codes_required: Literal[True]
    reason_code_pattern: Literal["^[a-z][a-z0-9_]{0,63}$"]
    maximum_schema_retries: Literal[1]
    invalid_response_action: Literal["abstain"]


class TeacherPanel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["quality-teacher-panel-v1"]
    lifecycle: Literal["candidate_qualification"]
    runtime_activation: Literal[False]
    teacher_output_alone_may_delete: Literal[False]
    training_objective: Literal["continued_pretraining"]
    initial_language_scope: tuple[Literal["english"], ...]
    allowed_external_data: tuple[Literal["public_license_compatible_calibration_samples"], ...]
    forbidden_inputs: tuple[str, ...]
    teachers: tuple[TeacherSpec, TeacherSpec, TeacherSpec]
    policies: tuple[QualityPolicy, QualityPolicy, QualityPolicy, QualityPolicy]
    response_contract: ResponseContract
    fixture_matrix: FixtureMatrix
    promotion_gate: PromotionGate

    @model_validator(mode="after")
    def validate_panel(self) -> "TeacherPanel":
        if len({teacher.teacher_id for teacher in self.teachers}) != 3:
            raise PanelContractError("Teacher IDs must be unique")
        if len({teacher.organization for teacher in self.teachers}) != 3:
            raise PanelContractError("Teacher organizations must be independent")
        locations = Counter(teacher.location for teacher in self.teachers)
        if locations != Counter({TeacherLocation.NVIDIA_BUILD: 2, TeacherLocation.LOCAL: 1}):
            raise PanelContractError("Panel requires exactly two NVIDIA Build and one local teacher")
        expected_policies = {
            "q1_correctness_evidence",
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        }
        if {policy.policy_id for policy in self.policies} != expected_policies:
            raise PanelContractError("Panel must declare Q1-Q4 exactly once")
        return self


class TeacherVote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    decision: PolicyDecision
    reason_codes: tuple[str, ...] = Field(min_length=1)


def load_teacher_panel(path: Path) -> TeacherPanel:
    return TeacherPanel.model_validate_json(path.read_text(encoding="utf-8"))


def _majority(votes: tuple[TeacherVote, ...]) -> tuple[PolicyDecision, frozenset[str]] | None:
    if len(votes) != 3 or len({vote.teacher_id for vote in votes}) != 3:
        raise PanelContractError("A panel pass requires exactly three unique teachers")
    if len({vote.policy_id for vote in votes}) != 1:
        raise PanelContractError("All votes in a panel pass must address the same policy")
    counts = Counter(vote.decision for vote in votes)
    decision, count = counts.most_common(1)[0]
    if count < 2 or decision is PolicyDecision.ABSTAIN:
        return None
    supporters = frozenset(vote.teacher_id for vote in votes if vote.decision is decision)
    return decision, supporters


def decide_panel(
    first_pass: tuple[TeacherVote, ...],
    second_pass: tuple[TeacherVote, ...] | None,
) -> PanelDecision:
    first = _majority(first_pass)
    if first is None:
        return PanelDecision.ABSTAIN
    first_decision, first_supporters = first
    if len(first_supporters) == 3:
        return PanelDecision(first_decision.value)
    if second_pass is None:
        return PanelDecision.ABSTAIN
    if {vote.teacher_id for vote in first_pass} != {vote.teacher_id for vote in second_pass}:
        raise PanelContractError("Blinded second pass must use the same three teachers")
    second = _majority(second_pass)
    if second is None:
        return PanelDecision.ABSTAIN
    second_decision, second_supporters = second
    stable_supporters = first_supporters & second_supporters
    if second_decision is first_decision and len(stable_supporters) >= 2:
        return PanelDecision(first_decision.value)
    return PanelDecision.ABSTAIN
