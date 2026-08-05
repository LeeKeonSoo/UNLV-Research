from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import sleep
from typing import Any, Mapping

from dotenv import load_dotenv

from quality_operating_points import CurationMode, QualityAction
from quality_stage_bridge import apply_coverage_veto, propose_quality_removals
from quality_teacher_adapters import NvidiaBuildBackend
from quality_teacher_panel import (
    PanelDecision,
    PolicyDecision,
    TeacherLocation,
    TeacherVote,
    load_teacher_panel,
)
from quality_teacher_runtime import EvaluationUnit, PanelPolicyResult
from quality_teacher_batch_runtime import (
    HostedPolicySetBatchAdapter,
    PolicySetBatchAdapter,
    UnitBatchResult,
    evaluate_quality_units_batched,
)
from quality_teacher_unit_runtime import InsufficientTeacherAvailability


JsonMap = dict[str, Any]
OBSERVATION_SCHEMA = "quality-teacher-corpus-observation-v2"
REPORT_SCHEMA = "quality-teacher-materialization-report-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[JsonMap]:
    rows: list[JsonMap] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _vote_to_mapping(vote: TeacherVote) -> JsonMap:
    return {
        "teacher_id": vote.teacher_id,
        "policy_id": vote.policy_id,
        "decision": vote.decision.value,
        "reason_codes": list(vote.reason_codes),
    }


def _vote_from_mapping(payload: Mapping[str, Any]) -> TeacherVote:
    return TeacherVote(
        teacher_id=str(payload["teacher_id"]),
        policy_id=str(payload["policy_id"]),
        decision=PolicyDecision(str(payload["decision"])),
        reason_codes=tuple(str(code) for code in payload.get("reason_codes") or ()),
    )


def _result_to_mapping(result: PanelPolicyResult) -> JsonMap:
    return {
        "policy_id": result.policy_id,
        "panel_decision": result.decision.value,
        "decision_source": result.decision_source,
        "decision_reason_codes": list(result.reason_codes),
        "first_pass": [_vote_to_mapping(vote) for vote in result.first_pass],
        "second_pass": (
            None
            if result.second_pass is None
            else [_vote_to_mapping(vote) for vote in result.second_pass]
        ),
    }


def panel_policy_result_to_mapping(result: PanelPolicyResult) -> JsonMap:
    """Serialize one frozen panel result for the Stage-B audit trail."""
    return _result_to_mapping(result)


def _result_from_mapping(payload: Mapping[str, Any]) -> PanelPolicyResult:
    second = payload.get("second_pass")
    return PanelPolicyResult(
        policy_id=str(payload["policy_id"]),
        decision=PanelDecision(str(payload["panel_decision"])),
        first_pass=tuple(_vote_from_mapping(vote) for vote in payload.get("first_pass") or ()),
        second_pass=(
            None
            if second is None
            else tuple(_vote_from_mapping(vote) for vote in second)
        ),
        decision_source=str(payload.get("decision_source") or "teacher_panel"),
        reason_codes=tuple(str(code) for code in payload.get("decision_reason_codes") or ()),
    )


def _task_id(panel_sha256: str, chunk_uid: str, text_sha256: str) -> str:
    payload = f"{panel_sha256}\0{chunk_uid}\0{text_sha256}\0combined_q1_q4"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path, panel_sha256: str) -> tuple[dict[str, JsonMap], int]:
    if not path.exists():
        return {}, 0
    cached: dict[str, JsonMap] = {}
    ignored_unavailable = 0
    for row in _read_jsonl(path):
        if row.get("schema_version") != OBSERVATION_SCHEMA:
            raise RuntimeError(f"Incompatible Quality observation schema in {path}")
        if row.get("teacher_panel_sha256") != panel_sha256:
            raise RuntimeError(f"Quality observation panel identity mismatch in {path}")
        if len(tuple(row.get("available_teacher_ids") or ())) < 2:
            ignored_unavailable += 1
            continue
        if len(tuple(row.get("policy_results") or ())) != 4:
            ignored_unavailable += 1
            continue
        cached[str(row["task_id"])] = row
    return cached, ignored_unavailable


def _evaluate_reliably(
    panel: Any,
    adapters: Mapping[str, PolicySetBatchAdapter],
    units: tuple[EvaluationUnit, ...],
    *,
    maximum_attempts: int = 4,
) -> tuple[UnitBatchResult, ...]:
    for attempt in range(1, maximum_attempts + 1):
        try:
            return evaluate_quality_units_batched(panel, adapters, units)
        except InsufficientTeacherAvailability:
            pass
        if attempt < maximum_attempts:
            sleep(2 ** (attempt - 1))
    raise RuntimeError(
        f"Quality provider unavailable after {maximum_attempts} attempts: "
        f"{','.join(unit.unit_id for unit in units)}"
    )


def _build_policy_set_adapters(panel: Any) -> Mapping[str, PolicySetBatchAdapter]:
    adapters: dict[str, PolicySetBatchAdapter] = {}
    for teacher in panel.teachers:
        if teacher.location is not TeacherLocation.NVIDIA_BUILD:
            raise RuntimeError("Combined Quality materialization requires hosted teachers")
        environment_variable = teacher.api_key_environment_variable
        endpoint = teacher.endpoint_base_url
        timeout = teacher.request_timeout_seconds
        retries = teacher.maximum_transport_retries
        if environment_variable is None or endpoint is None or timeout is None or retries is None:
            raise RuntimeError(f"Incomplete hosted teacher contract: {teacher.teacher_id}")
        adapters[teacher.teacher_id] = HostedPolicySetBatchAdapter(
            teacher=teacher,
            backend=NvidiaBuildBackend(
                api_key=os.environ.get(environment_variable, ""),
                base_url=endpoint,
                timeout_seconds=timeout,
                maximum_transport_retries=retries,
            ),
        )
    return adapters


def _evaluation_unit(row: Mapping[str, Any]) -> EvaluationUnit:
    text = str(row.get("text") or "")
    if not text:
        raise RuntimeError(f"Quality input has no text: {row.get('chunk_uid')}")
    declared_context = row.get("quality_declared_context")
    attached = row.get("quality_attached_evidence") or ()
    return EvaluationUnit(
        unit_id=str(row["chunk_uid"]),
        text=text,
        declared_context=None if declared_context is None else str(declared_context),
        attached_evidence=tuple(str(item) for item in attached),
        declared_verifier=None,
    )


def score_quality_rows(
    rows: list[JsonMap],
    *,
    panel_path: Path,
    dotenv_path: Path,
    cache_path: Path,
    task_workers: int,
) -> tuple[dict[str, tuple[PanelPolicyResult, ...]], JsonMap]:
    if task_workers < 1:
        raise ValueError("task_workers must be positive")
    if not load_dotenv(dotenv_path):
        raise RuntimeError(f"Could not load dotenv: {dotenv_path}")
    panel = load_teacher_panel(panel_path)
    if not panel.runtime_activation:
        raise RuntimeError("Quality teacher panel is not authorized for runtime materialization")
    panel_sha256 = _sha256(panel_path)
    cache, ignored_unavailable = _load_cache(cache_path, panel_sha256)
    expected_policy_ids = tuple(policy.policy_id for policy in panel.policies)
    adapters = _build_policy_set_adapters(panel)
    pending: list[tuple[str, str, EvaluationUnit]] = []
    required_task_ids: list[str] = []
    for row in rows:
        unit = _evaluation_unit(row)
        text_digest = _text_sha256(unit.text)
        task_id = _task_id(panel_sha256, unit.unit_id, text_digest)
        required_task_ids.append(task_id)
        if task_id not in cache:
            pending.append((task_id, text_digest, unit))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_observations = 0
    if pending:
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(max_workers=task_workers) as executor:
                unit_batches = [
                    pending[offset : offset + panel.unit_batch_size]
                    for offset in range(0, len(pending), panel.unit_batch_size)
                ]
                for offset in range(0, len(unit_batches), task_workers):
                    batch_group = unit_batches[offset : offset + task_workers]
                    futures = {
                        executor.submit(
                            _evaluate_reliably,
                            panel,
                            adapters,
                            tuple(item[2] for item in unit_batch),
                        ): unit_batch
                        for unit_batch in batch_group
                    }
                    for future in as_completed(futures):
                        result_by_id = {item.unit_id: item.evidence for item in future.result()}
                        for task_id, text_digest, unit in futures[future]:
                            result = result_by_id[unit.unit_id]
                            observation = {
                                "schema_version": OBSERVATION_SCHEMA,
                                "task_id": task_id,
                                "teacher_panel_sha256": panel_sha256,
                                "chunk_uid": unit.unit_id,
                                "text_sha256": text_digest,
                                "available_teacher_ids": list(result.available_teacher_ids),
                                "unavailable_teacher_ids": list(result.unavailable_teacher_ids),
                                "policy_results": [
                                    _result_to_mapping(policy_result)
                                    for policy_result in result.policy_results
                                ],
                            }
                            handle.write(json.dumps(observation, ensure_ascii=True, sort_keys=True) + "\n")
                            cache[task_id] = observation
                            new_observations += 1
                        handle.flush()
                    print(
                        json.dumps(
                            {
                                "quality_progress": sum(
                                    task_id in cache for task_id in required_task_ids
                                ),
                                "required": len(rows),
                            }
                        ),
                        flush=True,
                    )

    results_by_chunk: dict[str, tuple[PanelPolicyResult, ...]] = {}
    for row in rows:
        chunk_uid = str(row["chunk_uid"])
        text_digest = _text_sha256(str(row["text"]))
        observation = cache[_task_id(panel_sha256, chunk_uid, text_digest)]
        results = tuple(
            _result_from_mapping(result) for result in observation["policy_results"]
        )
        if tuple(result.policy_id for result in results) != expected_policy_ids:
            raise RuntimeError(f"Quality policy order mismatch in cache: {chunk_uid}")
        results_by_chunk[chunk_uid] = results
    used_observations = tuple(cache[task_id] for task_id in required_task_ids)
    availability_counts = {
        teacher.teacher_id: {
            "available_units": sum(
                teacher.teacher_id in tuple(observation.get("available_teacher_ids") or ())
                for observation in used_observations
            ),
            "unavailable_units": sum(
                teacher.teacher_id in tuple(observation.get("unavailable_teacher_ids") or ())
                for observation in used_observations
            ),
        }
        for teacher in panel.teachers
    }
    return results_by_chunk, {
        "teacher_panel_path": str(panel_path),
        "teacher_panel_sha256": panel_sha256,
        "quality_policy_ids": list(expected_policy_ids),
        "input_chunks": len(rows),
        "transport_mode": "all_q1_q4_policies_for_four_units_per_teacher_request",
        "unit_batch_size": panel.unit_batch_size,
        "required_observations": len(rows),
        "new_observations": new_observations,
        "reused_observations": len(rows) - new_observations,
        "unavailable_cache_observations_ignored": ignored_unavailable,
        "teacher_availability": availability_counts,
        "units_with_three_available_teachers": sum(
            len(tuple(observation.get("available_teacher_ids") or ())) == 3
            for observation in used_observations
        ),
        "units_with_two_available_teachers": sum(
            len(tuple(observation.get("available_teacher_ids") or ())) == 2
            for observation in used_observations
        ),
        "observation_cache_path": str(cache_path),
        "observation_cache_sha256": _sha256(cache_path),
    }


def _annotate_result(row: JsonMap, results: tuple[PanelPolicyResult, ...]) -> JsonMap:
    annotated = dict(row)
    annotated["quality_teacher_evidence"] = [_result_to_mapping(result) for result in results]
    return annotated


def materialize_modes(
    rows: list[JsonMap],
    results_by_chunk: Mapping[str, tuple[PanelPolicyResult, ...]],
    *,
    output_dir: Path,
    scoring_audit: Mapping[str, Any],
) -> JsonMap:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_modes: JsonMap = {}
    normal_retained_ids: set[str] | None = None
    for mode in (CurationMode.NORMAL, CurationMode.HARD):
        proposals = propose_quality_removals(results_by_chunk, mode)
        # Coverage receives the full typed proposal set. No metadata quota or target mix
        # can create protected IDs; malformed proposals fail the invariant below.
        final = apply_coverage_veto(proposals, protected_uids=set())
        retained: list[JsonMap] = []
        removed: list[JsonMap] = []
        reason_counts: dict[str, int] = {}
        for source_row in rows:
            chunk_uid = str(source_row["chunk_uid"])
            result = results_by_chunk[chunk_uid]
            decision = final[chunk_uid]
            row = _annotate_result(source_row, result)
            row["quality_stage_decision"] = asdict(decision)
            if decision.final_action == QualityAction.REMOVE.value:
                if not decision.failed_policy_ids:
                    raise RuntimeError(f"Unreasoned Quality removal: {chunk_uid}")
                removed.append(row)
                reason_counts[decision.stage_b_reason_code] = (
                    reason_counts.get(decision.stage_b_reason_code, 0) + 1
                )
            else:
                retained.append(row)
        retained_ids = {str(row["chunk_uid"]) for row in retained}
        removed_ids = {str(row["chunk_uid"]) for row in removed}
        input_ids = {str(row["chunk_uid"]) for row in rows}
        if retained_ids & removed_ids or retained_ids | removed_ids != input_ids:
            raise RuntimeError(f"Quality materialization partition invariant failed for {mode.value}")
        if normal_retained_ids is None:
            normal_retained_ids = retained_ids
        elif not retained_ids.issubset(normal_retained_ids):
            raise RuntimeError("Hard retained set must be a subset of Normal retained set")
        retained_path = output_dir / f"{mode.value}_curated_chunks.jsonl"
        removed_path = output_dir / f"{mode.value}_quality_removed_chunks.jsonl"
        _write_jsonl(retained_path, retained)
        _write_jsonl(removed_path, removed)
        report_modes[mode.value] = {
            "input_chunks": len(rows),
            "retained_chunks": len(retained),
            "removed_chunks": len(removed),
            "input_whitespace_token_proxy": sum(int(row.get("token_proxy") or 0) for row in rows),
            "retained_whitespace_token_proxy": sum(int(row.get("token_proxy") or 0) for row in retained),
            "removed_whitespace_token_proxy": sum(int(row.get("token_proxy") or 0) for row in removed),
            "reason_code_counts": reason_counts,
            "retained_path": str(retained_path),
            "retained_sha256": _sha256(retained_path),
            "removed_path": str(removed_path),
            "removed_sha256": _sha256(removed_path),
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "quality_teacher_materialization_complete",
        "stage_contract": {
            "stage_b": "Q1-Q4 typed Quality fail proposals",
            "stage_c": "Coverage materialization invariant and veto boundary",
        },
        "scoring": dict(scoring_audit),
        "modes": report_modes,
        "forbidden_runtime_inputs_read": [],
        "abstain_action": "retain",
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "token_budget_read": False,
    }
    report_path = output_dir / "quality_materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Q1-Q4 Quality panel and materialize Normal/Hard outputs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-workers", type=int, default=8)
    args = parser.parse_args()
    rows = _read_jsonl(args.input)
    results, scoring = score_quality_rows(
        rows,
        panel_path=args.panel,
        dotenv_path=args.dotenv,
        cache_path=args.cache,
        task_workers=args.task_workers,
    )
    report = materialize_modes(rows, results, output_dir=args.output_dir, scoring_audit=scoring)
    print(json.dumps({"status": report["status"], "modes": report["modes"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
