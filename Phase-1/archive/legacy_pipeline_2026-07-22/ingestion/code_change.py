"""Contracts for temporal software change bundles before generic Stage 0."""

from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Dict, List


CHANGE_BUNDLE_SCHEMA_VERSION = "temporal-code-change-bundle-v1"
ALLOWED_CHANGE_TYPES = {"added", "modified", "deleted", "renamed"}
ALLOWED_CONTENT_TYPES = {"code", "test", "documentation", "configuration", "other"}
ALLOWED_RIGHTS_STATUSES = {"allowed", "restricted", "unknown"}
TRAINING_CONTENT_TYPES = {"code", "test", "documentation"}
EXCLUDED_PATH_PARTS = {
    "vendor",
    "vendors",
    "vendored",
    "third_party",
    "third-party",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}
EXCLUDED_FILE_NAMES = {
    "poetry.lock",
    "pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}
EXCLUDED_SUFFIXES = {
    ".bin",
    ".dll",
    ".exe",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".min.js",
    ".min.css",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".svg",
    ".webp",
    ".zip",
}
SHA_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
GENERATED_FILE_NAME_RE = re.compile(
    r"(?:^generated[_-].*|.*[_-]generated|.*_pb2(?:_grpc)?|.*\.generated)\.(?:py|txt|rst|md)$",
    re.IGNORECASE,
)
GENERATED_CONTENT_RE = re.compile(
    r"\b(?:auto[- ]?generated|automatically generated|generated (?:code|file)|do not edit)\b",
    re.IGNORECASE,
)


def normalize_repository_identity(value: str) -> str:
    identity = str(value or "").strip().lower()
    identity = re.sub(r"^https?://(?:www\.)?github\.com/", "", identity)
    identity = re.sub(r"^git@github\.com:", "", identity)
    identity = re.sub(r"\.git$", "", identity)
    return identity.strip("/")


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _valid_file_record(record: Any, index: int) -> List[str]:
    prefix = f"files[{index}]"
    if not isinstance(record, dict):
        return [prefix]
    errors: List[str] = []
    path = record.get("path")
    if not isinstance(path, str) or not path.strip():
        errors.append(f"{prefix}.path")
    if record.get("change_type") not in ALLOWED_CHANGE_TYPES:
        errors.append(f"{prefix}.change_type")
    if record.get("content_type") not in ALLOWED_CONTENT_TYPES:
        errors.append(f"{prefix}.content_type")
    rights = record.get("rights")
    if not isinstance(rights, dict) or rights.get("status") not in ALLOWED_RIGHTS_STATUSES:
        errors.append(f"{prefix}.rights.status")
    for field in ("generated", "vendored", "binary", "secret_detected", "pii_detected"):
        if not isinstance(record.get(field), bool):
            errors.append(f"{prefix}.{field}")
    before = record.get("before_text")
    after = record.get("after_text")
    if before is not None and not isinstance(before, str):
        errors.append(f"{prefix}.before_text")
    if after is not None and not isinstance(after, str):
        errors.append(f"{prefix}.after_text")
    if record.get("change_type") == "added" and not isinstance(after, str):
        errors.append(f"{prefix}.after_text_required")
    if record.get("change_type") == "deleted" and not isinstance(before, str):
        errors.append(f"{prefix}.before_text_required")
    return errors


def validate_change_bundle(bundle: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if bundle.get("schema_version") != CHANGE_BUNDLE_SCHEMA_VERSION:
        errors.append("schema_version")
    for field in ("bundle_id", "repository_identity", "repository_url", "merge_timestamp"):
        if not isinstance(bundle.get(field), str) or not str(bundle.get(field)).strip():
            errors.append(field)
    if isinstance(bundle.get("repository_identity"), str) and "/" not in normalize_repository_identity(
        bundle["repository_identity"]
    ):
        errors.append("repository_identity")
    if not _valid_timestamp(bundle.get("merge_timestamp")):
        errors.append("merge_timestamp")
    for field in ("parent_commit", "merge_commit"):
        if not isinstance(bundle.get(field), str) or not SHA_RE.match(str(bundle.get(field))):
            errors.append(field)
    repository_rights = bundle.get("repository_rights")
    if not isinstance(repository_rights, dict) or repository_rights.get("status") not in ALLOWED_RIGHTS_STATUSES:
        errors.append("repository_rights.status")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance")
    else:
        for field in ("collector", "collector_version", "collected_at", "source_urls"):
            value = provenance.get(field)
            if field == "source_urls":
                if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                    errors.append(f"provenance.{field}")
            elif not isinstance(value, str) or not value.strip():
                errors.append(f"provenance.{field}")
    files = bundle.get("files")
    if not isinstance(files, list) or not files:
        errors.append("files")
    else:
        for index, record in enumerate(files):
            errors.extend(_valid_file_record(record, index))
    prose = bundle.get("prose")
    if not isinstance(prose, dict):
        errors.append("prose")
    else:
        if prose.get("training_authorized") is not False:
            errors.append("prose.training_authorized")
        for field in ("title", "body"):
            if field in prose and prose[field] is not None and not isinstance(prose[field], str):
                errors.append(f"prose.{field}")
    execution = bundle.get("execution_validation")
    if not isinstance(execution, dict):
        errors.append("execution_validation")
    else:
        for field in ("test_suite_present", "parent_checkout_reproducible", "merge_checkout_reproducible"):
            if not isinstance(execution.get(field), bool):
                errors.append(f"execution_validation.{field}")
        if not isinstance(execution.get("test_command_verified"), bool):
            errors.append("execution_validation.test_command_verified")
        if execution.get("test_suite_present") is True and (
            not isinstance(execution.get("test_command"), str) or not execution.get("test_command", "").strip()
        ):
            errors.append("execution_validation.test_command")
    return sorted(set(errors))


def _path_exclusion(path_value: str) -> str | None:
    path = PurePosixPath(str(path_value).replace("\\", "/").lower())
    if path.name in EXCLUDED_FILE_NAMES:
        return "lock_file"
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return "vendored_or_build_path"
    value = str(path)
    if any(value.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return "binary_generated_or_minified_suffix"
    return None


def path_exclusion_reason(path_value: str) -> str | None:
    return _path_exclusion(path_value)


def generated_file_detection(path_value: str, *texts: str | None) -> Dict[str, Any]:
    evidence: List[str] = []
    path = PurePosixPath(str(path_value).replace("\\", "/"))
    if GENERATED_FILE_NAME_RE.fullmatch(path.name):
        evidence.append("generated_filename_pattern")
    inspected_text = False
    for text in texts:
        if not isinstance(text, str):
            continue
        inspected_text = True
        if GENERATED_CONTENT_RE.search(text[:4096]):
            evidence.append("generated_content_marker")
    return {
        "generated": bool(evidence),
        "evidence": sorted(set(evidence)),
        "status": "completed_heuristic_v1" if inspected_text else "incomplete_no_text",
    }


def substantive_change_decision(record: Dict[str, Any]) -> Dict[str, Any]:
    before = record.get("before_text")
    after = record.get("after_text")
    change_type = record.get("change_type")
    if change_type == "added":
        return {"substantive": isinstance(after, str) and bool(after.strip()), "method": "added_content"}
    if change_type == "deleted":
        return {"substantive": isinstance(before, str) and bool(before.strip()), "method": "deleted_content"}
    if not isinstance(before, str) or not isinstance(after, str):
        return {"substantive": None, "method": "content_unavailable"}
    if str(record.get("path") or "").lower().endswith(".py"):
        try:
            before_ast = ast.dump(ast.parse(before), annotate_fields=True, include_attributes=False)
            after_ast = ast.dump(ast.parse(after), annotate_fields=True, include_attributes=False)
            return {"substantive": before_ast != after_ast, "method": "python_ast_without_locations"}
        except SyntaxError:
            pass
    normalize = lambda value: re.sub(r"\s+", " ", value).strip()
    return {"substantive": normalize(before) != normalize(after), "method": "whitespace_normalized_text"}


def file_training_eligibility(record: Dict[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    rights = record.get("rights") if isinstance(record.get("rights"), dict) else {}
    if rights.get("status") != "allowed":
        blockers.append("rights_not_allowed")
    if record.get("content_type") not in TRAINING_CONTENT_TYPES:
        blockers.append("content_type_not_training_authorized")
    for field in ("generated", "vendored", "binary", "secret_detected", "pii_detected"):
        if record.get(field) is True:
            blockers.append(field)
    path_reason = _path_exclusion(str(record.get("path") or ""))
    if path_reason:
        blockers.append(path_reason)
    if record.get("change_type") == "deleted":
        blockers.append("deleted_file_has_no_post_change_payload")
    if not isinstance(record.get("after_text"), str) or not str(record.get("after_text")).strip():
        blockers.append("missing_post_change_text")
    return {"eligible": not blockers, "blockers": sorted(set(blockers))}


def bundle_training_payload(bundle: Dict[str, Any]) -> Dict[str, Any]:
    validation_errors = validate_change_bundle(bundle)
    repository_rights = bundle.get("repository_rights") if isinstance(bundle.get("repository_rights"), dict) else {}
    file_rows = []
    payloads = []
    for record in bundle.get("files") if isinstance(bundle.get("files"), list) else []:
        eligibility = file_training_eligibility(record)
        file_rows.append({"path": record.get("path"), **eligibility})
        if eligibility["eligible"]:
            payloads.append(
                {
                    "path": record["path"],
                    "change_type": record["change_type"],
                    "content_type": record["content_type"],
                    "text": record["after_text"],
                    "license": (record.get("rights") or {}).get("license"),
                }
            )
    blockers: List[str] = []
    if validation_errors:
        blockers.append("invalid_change_bundle_contract")
    if repository_rights.get("status") != "allowed":
        blockers.append("repository_rights_not_allowed")
    if not payloads:
        blockers.append("no_training_eligible_files")
    return {
        "eligible": not blockers,
        "blockers": sorted(set(blockers)),
        "validation_errors": validation_errors,
        "training_payloads": payloads,
        "file_eligibility": file_rows,
        "excluded_prose": bundle.get("prose"),
    }


def bundle_protocol_eligibility(bundle: Dict[str, Any], protocol: Dict[str, Any]) -> Dict[str, Any]:
    payload = bundle_training_payload(bundle)
    blockers = list(payload["blockers"])
    allowed_licenses = set(protocol["collection_contract"]["allowed_licenses"])
    repository_rights = bundle.get("repository_rights") if isinstance(bundle.get("repository_rights"), dict) else {}
    if repository_rights.get("license") not in allowed_licenses:
        blockers.append("repository_license_not_allowlisted")
    for item in payload["training_payloads"]:
        if item.get("license") not in allowed_licenses:
            blockers.append("file_license_not_allowlisted")
    execution = bundle.get("execution_validation") if isinstance(bundle.get("execution_validation"), dict) else {}
    if execution.get("test_suite_present") is not True:
        blockers.append("test_suite_not_present")
    if execution.get("parent_checkout_reproducible") is not True:
        blockers.append("parent_checkout_not_reproducible")
    if execution.get("merge_checkout_reproducible") is not True:
        blockers.append("merge_checkout_not_reproducible")
    substantive_rows = [
        {"path": record.get("path"), **substantive_change_decision(record)}
        for record in (bundle.get("files") if isinstance(bundle.get("files"), list) else [])
    ]
    if not any(row["substantive"] is True for row in substantive_rows):
        if any(row["substantive"] is None for row in substantive_rows):
            blockers.append("substantive_change_not_verified")
        else:
            blockers.append("pure_formatting_or_metadata_only_change")
    return {
        **payload,
        "eligible": not blockers,
        "blockers": sorted(set(blockers)),
        "substantive_change_evidence": substantive_rows,
    }


def bundle_executable_evaluation_eligibility(bundle: Dict[str, Any]) -> Dict[str, Any]:
    execution = bundle.get("execution_validation") if isinstance(bundle.get("execution_validation"), dict) else {}
    blockers = []
    if execution.get("test_suite_present") is not True:
        blockers.append("test_suite_not_present")
    if execution.get("parent_checkout_reproducible") is not True:
        blockers.append("parent_checkout_not_reproducible")
    if execution.get("merge_checkout_reproducible") is not True:
        blockers.append("merge_checkout_not_reproducible")
    if execution.get("test_command_verified") is not True:
        blockers.append("test_command_not_verified")
    return {
        "eligible": not blockers,
        "blockers": sorted(set(blockers)),
        "test_command": execution.get("test_command"),
    }
