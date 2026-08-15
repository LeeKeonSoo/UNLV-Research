#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from quality_teacher_adapters import CompletionRequest, StructuredResponseFormat
from quality_teacher_batch_runtime import (
    PolicySetBatchGenerationRequest,
    _messages,
    build_policy_set_response_schema,
    parse_policy_set_batch_response,
)
from quality_teacher_openai_batch import (
    build_batch_request,
    extract_batch_output_text,
    summarize_luna_batch_usage,
)
from quality_teacher_openai import openai_text_format
from quality_teacher_materialization import (
    OBSERVATION_SCHEMA,
    _quality_runtime_sha256,
    _result_to_mapping,
    _task_id,
)
from quality_teacher_panel import PolicyDecision, TeacherVote, decide_single_teacher, load_teacher_panel
from quality_teacher_runtime import DeclaredVerifierEvidence, EvaluationUnit, PanelPolicyResult


MODEL_ID = "gpt-5.6-luna"
TEACHER_ID = "openai-gpt-5.6-luna-single-v1"
ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(items: list[EvaluationUnit], size: int) -> Iterable[tuple[EvaluationUnit, ...]]:
    for offset in range(0, len(items), size):
        yield tuple(items[offset : offset + size])


def alias_units(
    units: tuple[EvaluationUnit, ...],
) -> tuple[tuple[EvaluationUnit, ...], list[dict[str, str]]]:
    aliased: list[EvaluationUnit] = []
    linkage: list[dict[str, str]] = []
    for index, unit in enumerate(units):
        alias = f"u{index:02d}"
        aliased.append(unit.model_copy(update={"unit_id": alias}))
        linkage.append({"alias": alias, "unit_id": unit.unit_id})
    return tuple(aliased), linkage


def _unit(row: Mapping[str, Any]) -> EvaluationUnit:
    text = str(row.get("text") or "")
    if not text:
        raise RuntimeError(f"Quality input has no text: {row.get('chunk_uid')}")
    context = row.get("quality_declared_context")
    evidence = row.get("quality_attached_evidence") or ()
    raw_verifier = row.get("quality_declared_verifier")
    verifier = (
        None
        if raw_verifier is None
        else DeclaredVerifierEvidence.model_validate(raw_verifier)
    )
    return EvaluationUnit(
        unit_id=str(row["chunk_uid"]),
        text=text,
        declared_context=None if context is None else str(context),
        attached_evidence=tuple(str(item) for item in evidence),
        declared_verifier=verifier,
    )


def prepare(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    panel_path = Path(args.panel).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_teacher_panel(panel_path)
    rows = _read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    units = [_unit(row) for row in rows]
    if not units:
        raise RuntimeError("No Quality units were selected")

    request_path = output_dir / "batch_input.jsonl"
    manifest_groups: list[dict[str, Any]] = []
    reason_codes = tuple(code for policy in panel.policies for code in policy.reason_codes.all())
    with request_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, group in enumerate(_chunks(units, args.unit_batch_size)):
            custom_id = f"{args.split}-p1-{index:06d}"
            blind_run_id = uuid4().hex
            aliased_group, unit_linkage = alias_units(group)
            generation = PolicySetBatchGenerationRequest(
                teacher_id=TEACHER_ID,
                model_id=MODEL_ID,
                policies=panel.policies,
                units=aliased_group,
                pass_index=1,
                blind_run_id=blind_run_id,
                schema_retry=False,
            )
            request = CompletionRequest(
                model_id=MODEL_ID,
                messages=_messages(generation),
                maximum_new_tokens=args.maximum_output_tokens,
                response_format=StructuredResponseFormat(
                    "json_object",
                    reason_codes,
                    json_schema_name="quality_policy_votes",
                    json_schema=build_policy_set_response_schema(panel.policies, aliased_group),
                ),
            )
            batch_row = build_batch_request(
                custom_id=custom_id,
                request=request,
                reasoning_effort=args.reasoning_effort,
            )
            handle.write(json.dumps(batch_row, ensure_ascii=True, sort_keys=True) + "\n")
            manifest_groups.append(
                {
                    "custom_id": custom_id,
                    "blind_run_id": blind_run_id,
                    "pass_index": 1,
                    "unit_linkage": unit_linkage,
                }
            )

    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "quality-teacher-luna-batch-manifest-v2",
            "model_id": MODEL_ID,
            "teacher_id": TEACHER_ID,
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
            "split": args.split,
            "reasoning_effort": args.reasoning_effort,
            "maximum_output_tokens": args.maximum_output_tokens,
            "unit_batch_size": args.unit_batch_size,
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "panel_path": str(panel_path),
            "panel_sha256": _sha256(panel_path),
            "unit_count": len(units),
            "request_count": len(manifest_groups),
            "groups": manifest_groups,
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "units": len(units),
                "requests": len(manifest_groups),
                "batch_input": str(request_path),
                "bytes": request_path.stat().st_size,
            }
        )
    )


def prepare_retry(args: argparse.Namespace) -> None:
    source_dir = Path(args.output_dir).resolve()
    retry_dir = Path(args.retry_output_dir).resolve()
    retry_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    invalid_ids = set(str(item) for item in report.get("invalid_request_ids") or ())
    if not invalid_ids:
        raise RuntimeError("No invalid request IDs require retry")

    error_path = source_dir / "batch_errors.jsonl"
    transport_failure_ids = (
        {str(row["custom_id"]) for row in _read_jsonl(error_path)}
        if error_path.is_file()
        else set()
    )
    source_requests = {
        str(row["custom_id"]): row
        for row in _read_jsonl(source_dir / "batch_input.jsonl")
    }
    groups_by_id = {
        str(group["custom_id"]): group
        for group in manifest["groups"]
    }
    panel = load_teacher_panel(Path(manifest["panel_path"]))
    reason_codes = tuple(code for policy in panel.policies for code in policy.reason_codes.all())
    retry_groups: list[dict[str, Any]] = []
    request_path = retry_dir / "batch_input.jsonl"
    with request_path.open("w", encoding="utf-8", newline="\n") as handle:
        for original_id in sorted(invalid_ids):
            source_row = source_requests.get(original_id)
            source_group = groups_by_id.get(original_id)
            if source_row is None or source_group is None:
                raise RuntimeError(f"Retry linkage is missing: {original_id}")
            retry_row = copy.deepcopy(source_row)
            retry_id = f"{original_id}-r{args.attempt}"
            retry_row["custom_id"] = retry_id
            body = retry_row["body"]
            raw_input = str(body["input"])
            prefix = "Return a json object only.\n"
            if raw_input.startswith(prefix):
                raw_input = raw_input[len(prefix) :]
            payload = json.loads(raw_input)
            if original_id not in transport_failure_ids:
                payload["execution"]["schema_retry"] = True
                raw_input = json.dumps(payload, ensure_ascii=True, sort_keys=True)
                body["instructions"] = (
                    str(body["instructions"])
                    + " The previous response violated the response schema; re-evaluate independently."
                )
            units = tuple(EvaluationUnit.model_validate(item) for item in payload["units"])
            body["text"]["format"] = openai_text_format(
                StructuredResponseFormat(
                    "json_object",
                    reason_codes,
                    json_schema_name="quality_policy_votes",
                    json_schema=build_policy_set_response_schema(panel.policies, units),
                )
            )
            body["input"] = prefix + raw_input
            handle.write(json.dumps(retry_row, ensure_ascii=True, sort_keys=True) + "\n")
            retry_group = copy.deepcopy(source_group)
            retry_group["custom_id"] = retry_id
            retry_group["original_custom_id"] = original_id
            retry_group["transport_retry"] = original_id in transport_failure_ids
            retry_group["schema_retry"] = original_id not in transport_failure_ids
            retry_groups.append(retry_group)

    retry_manifest = {
        **{key: value for key, value in manifest.items() if key != "groups"},
        "schema_version": "quality-teacher-luna-batch-manifest-v3",
        "split": f"{manifest['split']}-retry-{args.attempt}",
        "unit_count": sum(len(group["unit_linkage"]) for group in retry_groups),
        "request_count": len(retry_groups),
        "retry_attempt": args.attempt,
        "source_output_dir": str(source_dir),
        "transport_retry_count": sum(
            bool(group["transport_retry"]) for group in retry_groups
        ),
        "schema_retry_count": sum(bool(group["schema_retry"]) for group in retry_groups),
        "groups": retry_groups,
    }
    _write_json(retry_dir / "manifest.json", retry_manifest)
    print(
        json.dumps(
            {
                "status": "retry_prepared",
                "units": retry_manifest["unit_count"],
                "requests": retry_manifest["request_count"],
                "transport_retries": retry_manifest["transport_retry_count"],
                "schema_retries": retry_manifest["schema_retry_count"],
                "batch_input": str(request_path),
            }
        )
    )


def _merge_parsed_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            unit_id = str(row["unit_id"])
            existing = merged.get(unit_id)
            if existing is not None and existing != row:
                raise RuntimeError(f"Conflicting Batch evidence for unit: {unit_id}")
            merged[unit_id] = row
    return merged


def prepare_second_pass(args: argparse.Namespace) -> None:
    source_dirs = [Path(item).resolve() for item in args.first_pass_dir]
    if not source_dirs:
        raise RuntimeError("At least one first-pass directory is required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_paths = [source / "parsed_pass_1.jsonl" for source in source_dirs]
    if not all(path.is_file() for path in parsed_paths):
        missing = [str(path) for path in parsed_paths if not path.is_file()]
        raise RuntimeError(f"First-pass parsed evidence is missing: {missing}")
    merged = _merge_parsed_rows(parsed_paths)

    base_manifest = json.loads((source_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    expected_units = int(base_manifest["unit_count"])
    if len(merged) != expected_units:
        raise RuntimeError(
            f"First-pass evidence is incomplete: {len(merged)}/{expected_units}"
        )
    panel_path = Path(args.panel).resolve()
    panel = load_teacher_panel(panel_path)
    input_path = Path(base_manifest["input_path"])
    input_units = [_unit(row) for row in _read_jsonl(input_path)]
    units_by_id = {unit.unit_id: unit for unit in input_units}
    fail_unit_ids = {
        unit_id
        for unit_id, row in merged.items()
        if any(str(vote["decision"]) == "fail" for vote in row["votes"])
    }
    fail_units = [unit for unit in input_units if unit.unit_id in fail_unit_ids]
    if len(fail_units) != len(fail_unit_ids):
        raise RuntimeError("First-pass fail evidence references unknown input units")

    merged_path = output_dir / "first_pass_merged.jsonl"
    with merged_path.open("w", encoding="utf-8", newline="\n") as handle:
        for unit in input_units:
            handle.write(json.dumps(merged[unit.unit_id], ensure_ascii=True, sort_keys=True) + "\n")

    request_path = output_dir / "batch_input.jsonl"
    groups: list[dict[str, Any]] = []
    reason_codes = tuple(code for policy in panel.policies for code in policy.reason_codes.all())
    with request_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, group in enumerate(_chunks(fail_units, args.unit_batch_size)):
            custom_id = f"{args.split}-p2-{index:06d}"
            blind_run_id = uuid4().hex
            aliased_group, unit_linkage = alias_units(group)
            generation = PolicySetBatchGenerationRequest(
                teacher_id=TEACHER_ID,
                model_id=MODEL_ID,
                policies=panel.policies,
                units=aliased_group,
                pass_index=2,
                blind_run_id=blind_run_id,
                schema_retry=False,
            )
            request = CompletionRequest(
                model_id=MODEL_ID,
                messages=_messages(generation),
                maximum_new_tokens=args.maximum_output_tokens,
                response_format=StructuredResponseFormat(
                    "json_object",
                    reason_codes,
                    json_schema_name="quality_policy_votes",
                    json_schema=build_policy_set_response_schema(panel.policies, aliased_group),
                ),
            )
            batch_row = build_batch_request(
                custom_id=custom_id,
                request=request,
                reasoning_effort=args.reasoning_effort,
            )
            handle.write(json.dumps(batch_row, ensure_ascii=True, sort_keys=True) + "\n")
            groups.append(
                {
                    "custom_id": custom_id,
                    "blind_run_id": blind_run_id,
                    "pass_index": 2,
                    "unit_linkage": unit_linkage,
                }
            )

    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "quality-teacher-luna-batch-manifest-v4",
            "model_id": MODEL_ID,
            "teacher_id": TEACHER_ID,
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
            "split": args.split,
            "reasoning_effort": args.reasoning_effort,
            "maximum_output_tokens": args.maximum_output_tokens,
            "unit_batch_size": args.unit_batch_size,
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "panel_path": str(panel_path),
            "panel_sha256": _sha256(panel_path),
            "unit_count": len(fail_units),
            "request_count": len(groups),
            "pass_index": 2,
            "first_pass_source_dirs": [str(path) for path in source_dirs],
            "first_pass_complete_units": len(merged),
            "groups": groups,
        },
    )
    print(
        json.dumps(
            {
                "status": "second_pass_prepared",
                "first_pass_units": len(merged),
                "fail_candidate_units": len(fail_units),
                "requests": len(groups),
                "batch_input": str(request_path),
            }
        )
    )


def materialize_observations(args: argparse.Namespace) -> None:
    first_pass_dir = Path(args.first_pass_dir).resolve()
    second_pass_dirs = [Path(item).resolve() for item in args.second_pass_dir]
    panel_path = Path(args.panel).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    panel = load_teacher_panel(panel_path)
    teacher = panel.teachers[0]
    first_rows = _merge_parsed_rows([first_pass_dir / "first_pass_merged.jsonl"])
    second_rows = _merge_parsed_rows(
        [directory / "parsed_pass_2.jsonl" for directory in second_pass_dirs]
    )
    input_rows = _read_jsonl(input_path)
    expected_uids = {str(row["chunk_uid"]) for row in input_rows}
    if set(first_rows) != expected_uids:
        raise RuntimeError(
            f"First-pass observation universe mismatch: {len(first_rows)}/{len(expected_uids)}"
        )
    expected_second_uids = {
        unit_id
        for unit_id, row in first_rows.items()
        if any(str(vote["decision"]) == "fail" for vote in row["votes"])
    }
    if set(second_rows) != expected_second_uids:
        raise RuntimeError(
            f"Second-pass observation universe mismatch: {len(second_rows)}/"
            f"{len(expected_second_uids)}"
        )

    panel_sha256 = _sha256(panel_path)
    runtime_sha256 = _quality_runtime_sha256()
    decision_counts = {"pass": 0, "fail": 0, "abstain": 0}
    first_fail_votes = 0
    confirmed_fail_votes = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for input_row in input_rows:
            chunk_uid = str(input_row["chunk_uid"])
            text = str(input_row["text"])
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            first_by_policy = {
                str(vote["policy_id"]): vote
                for vote in first_rows[chunk_uid]["votes"]
            }
            second_by_policy = (
                {
                    str(vote["policy_id"]): vote
                    for vote in second_rows[chunk_uid]["votes"]
                }
                if chunk_uid in second_rows
                else {}
            )
            policy_results: list[dict[str, Any]] = []
            for policy in panel.policies:
                first_payload = first_by_policy[policy.policy_id]
                first_vote = TeacherVote(
                    teacher_id=teacher.teacher_id,
                    policy_id=policy.policy_id,
                    decision=PolicyDecision(str(first_payload["decision"])),
                    reason_codes=tuple(str(code) for code in first_payload["reason_codes"]),
                )
                second_votes: tuple[TeacherVote, ...] | None = None
                if first_vote.decision is PolicyDecision.FAIL:
                    first_fail_votes += 1
                    second_payload = second_by_policy[policy.policy_id]
                    second_vote = TeacherVote(
                        teacher_id=teacher.teacher_id,
                        policy_id=policy.policy_id,
                        decision=PolicyDecision(str(second_payload["decision"])),
                        reason_codes=tuple(
                            str(code) for code in second_payload["reason_codes"]
                        ),
                    )
                    second_votes = (second_vote,)
                decision = decide_single_teacher((first_vote,), second_votes)
                if decision.value == "fail":
                    confirmed_fail_votes += 1
                decision_counts[decision.value] += 1
                policy_results.append(
                    _result_to_mapping(
                        PanelPolicyResult(
                            policy_id=policy.policy_id,
                            decision=decision,
                            first_pass=(first_vote,),
                            second_pass=second_votes,
                        )
                    )
                )
            observation = {
                "schema_version": OBSERVATION_SCHEMA,
                "task_id": _task_id(
                    panel_sha256,
                    runtime_sha256,
                    chunk_uid,
                    text_sha256,
                ),
                "teacher_panel_sha256": panel_sha256,
                "quality_runtime_sha256": runtime_sha256,
                "aggregation_strategy": panel.aggregation_strategy,
                "chunk_uid": chunk_uid,
                "text_sha256": text_sha256,
                "available_teacher_ids": [teacher.teacher_id],
                "unavailable_teacher_ids": [],
                "policy_results": policy_results,
            }
            handle.write(json.dumps(observation, ensure_ascii=True, sort_keys=True) + "\n")

    report = {
        "schema_version": "quality-teacher-luna-observation-materialization-v1",
        "status": "complete",
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "teacher_panel_path": str(panel_path),
        "teacher_panel_sha256": panel_sha256,
        "quality_runtime_sha256": runtime_sha256,
        "aggregation_strategy": panel.aggregation_strategy,
        "observation_count": len(input_rows),
        "policy_vote_count": len(input_rows) * len(panel.policies),
        "first_fail_votes": first_fail_votes,
        "confirmed_fail_votes": confirmed_fail_votes,
        "unconfirmed_fail_votes_retained_as_abstain": (
            first_fail_votes - confirmed_fail_votes
        ),
        "decision_counts": decision_counts,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=True))


def split_batch(args: argparse.Namespace) -> None:
    source_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_output_dir).resolve()
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    panel = load_teacher_panel(Path(manifest["panel_path"]))
    reason_codes = tuple(code for policy in panel.policies for code in policy.reason_codes.all())
    requests = {
        str(row["custom_id"]): row
        for row in _read_jsonl(source_dir / "batch_input.jsonl")
    }
    prefix = "Return a json object only.\n"
    output_groups: list[dict[str, Any]] = []
    request_path = split_dir / "batch_input.jsonl"
    with request_path.open("w", encoding="utf-8", newline="\n") as handle:
        for group in manifest["groups"]:
            source_id = str(group["custom_id"])
            source_row = requests[source_id]
            body = source_row["body"]
            raw_input = str(body["input"])
            payload = json.loads(raw_input[len(prefix) :] if raw_input.startswith(prefix) else raw_input)
            units = list(payload["units"])
            linkage = list(group["unit_linkage"])
            if len(units) != len(linkage):
                raise RuntimeError(f"Split linkage mismatch: {source_id}")
            for index in range(0, len(units), args.unit_batch_size):
                subset_units = units[index : index + args.unit_batch_size]
                subset_linkage = linkage[index : index + args.unit_batch_size]
                split_index = index // args.unit_batch_size
                custom_id = f"{source_id}-s{split_index:02d}"
                blind_run_id = uuid4().hex
                subset_payload = copy.deepcopy(payload)
                subset_payload["units"] = subset_units
                subset_payload["execution"]["blind_run_id"] = blind_run_id
                split_row = copy.deepcopy(source_row)
                split_row["custom_id"] = custom_id
                split_row["body"]["input"] = prefix + json.dumps(
                    subset_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                subset_evaluation_units = tuple(
                    EvaluationUnit.model_validate(item) for item in subset_units
                )
                split_row["body"]["text"]["format"] = openai_text_format(
                    StructuredResponseFormat(
                        "json_object",
                        reason_codes,
                        json_schema_name="quality_policy_votes",
                        json_schema=build_policy_set_response_schema(
                            panel.policies,
                            subset_evaluation_units,
                        ),
                    )
                )
                handle.write(json.dumps(split_row, ensure_ascii=True, sort_keys=True) + "\n")
                split_group = copy.deepcopy(group)
                split_group["custom_id"] = custom_id
                split_group["source_custom_id"] = source_id
                split_group["blind_run_id"] = blind_run_id
                split_group["unit_linkage"] = subset_linkage
                output_groups.append(split_group)

    split_manifest = {
        **{key: value for key, value in manifest.items() if key != "groups"},
        "schema_version": "quality-teacher-luna-batch-manifest-v5",
        "split": f"{manifest['split']}-split-{args.unit_batch_size}",
        "unit_batch_size": args.unit_batch_size,
        "request_count": len(output_groups),
        "source_output_dir": str(source_dir),
        "groups": output_groups,
    }
    _write_json(split_dir / "manifest.json", split_manifest)
    print(
        json.dumps(
            {
                "status": "split_prepared",
                "units": split_manifest["unit_count"],
                "requests": len(output_groups),
                "unit_batch_size": args.unit_batch_size,
                "batch_input": str(request_path),
            }
        )
    )


def _client() -> OpenAI:
    repository_root = Path(__file__).resolve().parents[1]
    load_dotenv(repository_root / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key, max_retries=2, timeout=120.0)


def _dump_sdk_model(value: object) -> dict[str, Any]:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise RuntimeError(f"Expected an OpenAI SDK model, got {type(value).__name__}")
    payload = model_dump(mode="json")
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI SDK model did not serialize to an object")
    return payload


def submit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    request_path = output_dir / "batch_input.jsonl"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    client = _client()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={
            "workflow": "quality_teacher_luna",
            "split": str(manifest["split"]),
            "schema": "quality-teacher-luna-batch-v1",
        },
    )
    state = {
        "schema_version": "quality-teacher-luna-batch-state-v1",
        "input_file": _dump_sdk_model(uploaded),
        "batch": _dump_sdk_model(batch),
    }
    _write_json(output_dir / "state.json", state)
    print(json.dumps({"status": batch.status, "batch_id": batch.id, "input_file_id": uploaded.id}))


def status(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_id = str(state["batch"]["id"])
    batch = _client().batches.retrieve(batch_id)
    state["batch"] = _dump_sdk_model(batch)
    _write_json(state_path, state)
    print(
        json.dumps(
            {
                "batch_id": batch.id,
                "status": batch.status,
                "request_counts": _dump_sdk_model(batch).get("request_counts"),
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
        )
    )


def cancel(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = _client().batches.cancel(str(state["batch"]["id"]))
    state["batch"] = _dump_sdk_model(batch)
    _write_json(state_path, state)
    print(json.dumps({"batch_id": batch.id, "status": batch.status}))


def collect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = _client().batches.retrieve(str(state["batch"]["id"]))
    state["batch"] = _dump_sdk_model(batch)
    _write_json(state_path, state)
    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"Batch is not collectable: {batch.status}")

    content = _client().files.content(batch.output_file_id).content
    result_path = output_dir / "batch_output.jsonl"
    result_path.write_bytes(content)
    result_rows = _read_jsonl(result_path)
    by_id = {str(row.get("custom_id")): row for row in result_rows}

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    panel = load_teacher_panel(Path(manifest["panel_path"]))
    required_unit_ids = {
        str(item["unit_id"])
        for group in manifest["groups"]
        for item in group["unit_linkage"]
    }
    units_by_id = {
        unit.unit_id: unit
        for row in _read_jsonl(Path(manifest["input_path"]))
        for unit in (_unit(row),)
        if unit.unit_id in required_unit_ids
    }
    if set(units_by_id) != required_unit_ids:
        raise RuntimeError("Batch manifest references missing input units")
    decisions = {"pass": 0, "fail": 0, "abstain": 0}
    parsed_units = 0
    invalid: list[str] = []
    pass_indexes = {int(group["pass_index"]) for group in manifest["groups"]}
    if len(pass_indexes) != 1:
        raise RuntimeError("One Batch artifact cannot mix teacher pass indexes")
    pass_index = next(iter(pass_indexes))
    parsed_path = output_dir / f"parsed_pass_{pass_index}.jsonl"
    with parsed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for group in manifest["groups"]:
            custom_id = str(group["custom_id"])
            row = by_id.get(custom_id)
            if row is None:
                invalid.append(custom_id)
                continue
            try:
                raw = extract_batch_output_text(row)
            except Exception:
                invalid.append(custom_id)
                continue
            linkage = tuple(group["unit_linkage"])
            group_units = tuple(
                units_by_id[str(item["unit_id"])].model_copy(
                    update={"unit_id": str(item["alias"])}
                )
                for item in linkage
            )
            parsed = parse_policy_set_batch_response(raw, group_units, panel.policies)
            if parsed is None:
                invalid.append(custom_id)
                continue
            original_by_alias = {
                str(item["alias"]): str(item["unit_id"])
                for item in linkage
            }
            for unit in group_units:
                votes = parsed[unit.unit_id]
                for vote in votes:
                    decisions[vote.decision.value] += 1
                handle.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "unit_id": original_by_alias[unit.unit_id],
                            "provider_unit_alias": unit.unit_id,
                            "pass_index": pass_index,
                            "votes": [vote.model_dump(mode="json") for vote in votes],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
                parsed_units += 1

    report = {
        "schema_version": "quality-teacher-luna-batch-smoke-report-v1",
        "batch_id": batch.id,
        "model_id": MODEL_ID,
        "pass_index": pass_index,
        "batch_discount_applied_by_api": True,
        "expected_units": manifest["unit_count"],
        "parsed_units": parsed_units,
        "expected_policy_votes": int(manifest["unit_count"]) * 4,
        "decision_counts": decisions,
        "invalid_request_ids": invalid,
        "passed": parsed_units == int(manifest["unit_count"]) and not invalid,
        "request_counts": _dump_sdk_model(batch).get("request_counts"),
        "token_usage_and_estimated_cost": summarize_luna_batch_usage(result_rows),
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=True))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="GPT-5.6 Luna Batch Quality-teacher runner")
    subparsers = command.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--panel", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--split", required=True)
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--unit-batch-size", type=int, default=16, choices=range(1, 17))
    prepare_parser.add_argument("--maximum-output-tokens", type=int, default=4096)
    prepare_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    prepare_parser.set_defaults(handler=prepare)

    retry_parser = subparsers.add_parser("prepare-retry")
    retry_parser.add_argument("--output-dir", required=True)
    retry_parser.add_argument("--retry-output-dir", required=True)
    retry_parser.add_argument("--attempt", type=int, default=1)
    retry_parser.set_defaults(handler=prepare_retry)

    second_parser = subparsers.add_parser("prepare-second-pass")
    second_parser.add_argument("--first-pass-dir", action="append", required=True)
    second_parser.add_argument("--panel", required=True)
    second_parser.add_argument("--output-dir", required=True)
    second_parser.add_argument("--split", required=True)
    second_parser.add_argument("--unit-batch-size", type=int, default=16, choices=range(1, 17))
    second_parser.add_argument("--maximum-output-tokens", type=int, default=4096)
    second_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    second_parser.set_defaults(handler=prepare_second_pass)

    observations_parser = subparsers.add_parser("materialize-observations")
    observations_parser.add_argument("--first-pass-dir", required=True)
    observations_parser.add_argument("--second-pass-dir", action="append", required=True)
    observations_parser.add_argument("--panel", required=True)
    observations_parser.add_argument("--input", required=True)
    observations_parser.add_argument("--output", required=True)
    observations_parser.set_defaults(handler=materialize_observations)

    split_parser = subparsers.add_parser("split-batch")
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--split-output-dir", required=True)
    split_parser.add_argument("--unit-batch-size", type=int, choices=range(1, 17), required=True)
    split_parser.set_defaults(handler=split_batch)

    for name, handler in (
        ("submit", submit),
        ("status", status),
        ("cancel", cancel),
        ("collect", collect),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--output-dir", required=True)
        subparser.set_defaults(handler=handler)
    return command


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
