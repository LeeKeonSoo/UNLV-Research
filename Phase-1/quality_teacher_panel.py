from __future__ import annotations

from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


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


PolicyReasonCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"),
]
ReasoningControl = Literal[
    "none",
    "enable_thinking_false",
    "thinking_false",
    "reasoning_effort_none",
]


class PolicyReasonCodes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    pass_: tuple[PolicyReasonCode, ...] = Field(alias="pass", min_length=1)
    fail: tuple[PolicyReasonCode, ...] = Field(min_length=1)
    abstain: tuple[PolicyReasonCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disjoint_codes(self) -> "PolicyReasonCodes":
        groups = (set(self.pass_), set(self.fail), set(self.abstain))
        if sum(len(group) for group in groups) != len(set().union(*groups)):
            raise PanelContractError("Policy reason-code decision groups must be disjoint")
        return self

    def for_decision(self, decision: PolicyDecision) -> tuple[str, ...]:
        match decision:
            case PolicyDecision.PASS:
                return self.pass_
            case PolicyDecision.FAIL:
                return self.fail
            case PolicyDecision.ABSTAIN:
                return self.abstain
            case unreachable:
                assert_never(unreachable)

    def all(self) -> tuple[str, ...]:
        return self.pass_ + self.fail + self.abstain


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
    maximum_new_tokens: int = Field(gt=0, le=4096)
    request_timeout_seconds: int | None = Field(default=None, gt=0, le=900)
    maximum_transport_retries: int | None = Field(default=None, ge=0, le=2)
    structured_output_mode: Literal["json_object"] | None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    reasoning_control: ReasoningControl = "none"

    @model_validator(mode="after")
    def validate_location_contract(self) -> "TeacherSpec":
        match self.location:
            case TeacherLocation.NVIDIA_BUILD:
                if self.endpoint_base_url is None or self.api_key_environment_variable is None:
                    raise PanelContractError("NVIDIA Build teachers require endpoint and API-key variable")
                if self.inference_precision != "endpoint_managed":
                    raise PanelContractError("NVIDIA Build teachers must use endpoint-managed precision")
                if (
                    self.request_timeout_seconds is None
                    or self.maximum_transport_retries is None
                    or self.structured_output_mode is None
                ):
                    raise PanelContractError("NVIDIA Build teachers require frozen transport controls")
            case TeacherLocation.LOCAL:
                if self.endpoint_base_url is not None or self.api_key_environment_variable is not None:
                    raise PanelContractError("Local teachers cannot declare a hosted endpoint or API key")
                if self.inference_precision != "bitsandbytes_int8":
                    raise PanelContractError("The frozen local teacher requires bitsandbytes int8 inference")
                if (
                    self.request_timeout_seconds is not None
                    or self.maximum_transport_retries is not None
                    or self.structured_output_mode is not None
                ):
                    raise PanelContractError("Local teachers cannot declare hosted transport controls")
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
    reason_codes: PolicyReasonCodes


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

    schema_version: Literal["quality-teacher-panel-v1", "quality-teacher-panel-v2"]
    lifecycle: Literal["candidate_qualification", "runtime_experiment_quality_deletion"]
    runtime_activation: bool
    transport_mode: Literal[
        "single_policy_request",
        "all_policies_per_unit_request",
    ] = "single_policy_request"
    unit_batch_size: int = Field(default=1, ge=1, le=8)
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
        expected_activation = self.lifecycle == "runtime_experiment_quality_deletion"
        if self.runtime_activation is not expected_activation:
            raise PanelContractError(
                "Panel lifecycle and runtime_activation must change together"
            )
        if self.lifecycle == "runtime_experiment_quality_deletion" and self.schema_version != "quality-teacher-panel-v2":
            raise PanelContractError("Only the frozen v2 hosted panel may be runtime-active")
        if self.runtime_activation and self.transport_mode != "all_policies_per_unit_request":
            raise PanelContractError("Runtime Quality deletion requires combined Q1-Q4 transport")
        if self.runtime_activation and self.unit_batch_size != 4:
            raise PanelContractError("Runtime Quality deletion requires the frozen four-unit batch")
        if len({teacher.teacher_id for teacher in self.teachers}) != 3:
            raise PanelContractError("Teacher IDs must be unique")
        if len({teacher.organization for teacher in self.teachers}) != 3:
            raise PanelContractError("Teacher organizations must be independent")
        locations = Counter(teacher.location for teacher in self.teachers)
        match self.schema_version:
            case "quality-teacher-panel-v1":
                expected_locations = Counter(
                    {TeacherLocation.NVIDIA_BUILD: 2, TeacherLocation.LOCAL: 1}
                )
            case "quality-teacher-panel-v2":
                expected_locations = Counter({TeacherLocation.NVIDIA_BUILD: 3})
            case unreachable:
                assert_never(unreachable)
        if locations != expected_locations:
            raise PanelContractError(
                f"{self.schema_version} has an invalid teacher-location topology"
            )
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
