"""Stage A text normalization and conservative quarantine heuristics."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List

from ingestion.candidate_contract import CANDIDATE_RECORD_SCHEMA_VERSION, release_eligibility


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
PHONE_CONTEXT_RE = re.compile(r"\b(?:cell|contact|fax|mobile|phone|telephone|tel|whatsapp)\b", re.IGNORECASE)
FORMATTED_PHONE_RE = re.compile(
    r"^(?:\(?\d{3}\)?\s*\d{3}[-. ]\d{4}|\d{3}[-. ]\d{3}[-. ]\d{4})$"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[A-Za-z0-9_\-/.+]{8,}",
    re.IGNORECASE,
)
SECRET_TOKEN_RE = re.compile(
    r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9_\-/.+]{20,}", re.IGNORECASE)
SESSION_HEADER_RE = re.compile(
    r"\b(?:cookie|set-cookie|authorization)\b['\"]?\s*[:=]\s*['\"]?[^'\"\n]{40,}",
    re.IGNORECASE,
)
ACQUISITION_FAILURE_RE = re.compile(
    r"\b(?:access denied|captcha required|rate limit exceeded|request timed out|"
    r"page not found|service unavailable|internal server error)\b",
    re.IGNORECASE,
)
ACQUISITION_FAILURE_STATUS_CODES = frozenset({401, 403, 404, 408, 429, 500, 502, 503, 504})
ACQUISITION_FAILURE_STATUSES = frozenset({"blocked", "error", "failed", "timeout"})
HTML_TAG_RE = re.compile(r"<[^>]+>")
BENCHMARK_RE = re.compile(
    r"\b(?:MMLU|GSM8K|HumanEval|MBPP|EvalPlus|LiveCodeBench|SWE[- ]bench|HellaSwag|ARC[- ]Challenge|TruthfulQA)\b",
    re.IGNORECASE,
)
POISONING_RE = re.compile(
    r"\b(?:ignore (?:all |the )?previous instructions|system prompt|training data poisoning|backdoor trigger|when you see .{0,40} output .{0,40}|hidden trigger)\b",
    re.IGNORECASE,
)
MOJIBAKE_MARKERS = ("�", "Ã", "Â", "â€™", "â€œ", "??")


STAGE_A_PAYLOAD_ABSENCE_REASON = "payload_absence"
STAGE_A_TEXT_CONTRACT_REASON = "text_contract_violation"
STAGE_A_CORRUPTION_REASON = "unrecoverable_corruption"
STAGE_A_ACQUISITION_FAILURE_REASON = "acquisition_failure"
STAGE_A_NORMALIZED_TEXT_REASON_CODES = frozenset({
    STAGE_A_PAYLOAD_ABSENCE_REASON,
    STAGE_A_TEXT_CONTRACT_REASON,
    STAGE_A_CORRUPTION_REASON,
    STAGE_A_ACQUISITION_FAILURE_REASON,
})
STAGE_A_POLICY_REASON_CODES = {
    "stage_a_normalized_text_integrity": STAGE_A_NORMALIZED_TEXT_REASON_CODES,
}


def normalize_text(text: str, *, context: str = "preserve") -> Dict[str, Any]:
    original = str(text or "")
    transformations: List[str] = []
    if context == "preserve":
        digest = hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest()
        return {
            "text": original,
            "transformations": transformations,
            "original_sha256": digest,
            "normalized_sha256": digest,
        }
    if context == "repository_code":
        normalized = original.replace("\r\n", "\n").replace("\r", "\n")
        if normalized != original:
            transformations.append("line_ending_normalization")
        if normalized.startswith("\ufeff"):
            normalized = normalized[1:]
            transformations.append("leading_utf8_bom_removal")
        return {
            "text": normalized,
            "transformations": transformations,
            "original_sha256": hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest(),
            "normalized_sha256": hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest(),
        }
    normalized = unicodedata.normalize("NFKC", original)
    if normalized != original:
        transformations.append("unicode_nfkc")
    collapsed = re.sub(r"[ \t\f\v]+", " ", normalized)
    collapsed = re.sub(r"\r\n?", "\n", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()
    if collapsed != normalized.strip():
        transformations.append("whitespace_normalization")
    return {
        "text": collapsed,
        "transformations": transformations,
        "original_sha256": hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest(),
        "normalized_sha256": hashlib.sha256(collapsed.encode("utf-8", errors="replace")).hexdigest(),
    }


def _phone_diagnostics(value: str, pii_context: str) -> Dict[str, int]:
    candidates = list(PHONE_RE.finditer(value))
    if pii_context not in {"repository_code", "technical_math"}:
        return {
            "phone_candidate_count": len(candidates),
            "phone_high_confidence_count": len(candidates),
            "phone_suppressed_count": 0,
        }
    high_confidence = 0
    for match in candidates:
        candidate = match.group(0).strip()
        digit_count = sum(character.isdigit() for character in candidate)
        nearby = value[max(0, match.start() - 40) : min(len(value), match.end() + 40)]
        if not 10 <= digit_count <= 15:
            continue
        if candidate.startswith("+") or FORMATTED_PHONE_RE.fullmatch(candidate) or PHONE_CONTEXT_RE.search(nearby):
            high_confidence += 1
    return {
        "phone_candidate_count": len(candidates),
        "phone_high_confidence_count": high_confidence,
        "phone_suppressed_count": len(candidates) - high_confidence,
    }


def detect_hazards(text: str, *, pii_context: str = "general") -> Dict[str, Any]:
    value = str(text or "")
    email_hits = len(EMAIL_RE.findall(value))
    phone = _phone_diagnostics(value, pii_context)
    pii_hits = email_hits + phone["phone_high_confidence_count"]
    secret_hits = (
        len(SECRET_ASSIGNMENT_RE.findall(value))
        + len(SECRET_TOKEN_RE.findall(value))
        + len(BEARER_TOKEN_RE.findall(value))
        + len(SESSION_HEADER_RE.findall(value))
    )
    benchmark_hits = len(BENCHMARK_RE.findall(value))
    poisoning_hits = len(POISONING_RE.findall(value))
    mojibake_hits = sum(value.count(marker) for marker in MOJIBAKE_MARKERS)
    return {
        "pii_detected": bool(pii_hits),
        "secret_detected": bool(secret_hits),
        "benchmark_contamination": bool(benchmark_hits),
        "poisoning_suspected": bool(poisoning_hits),
        "diagnostics": {
            "pii_hit_count": pii_hits,
            "email_hit_count": email_hits,
            **phone,
            "pii_context": pii_context,
            "secret_hit_count": secret_hits,
            "benchmark_marker_count": benchmark_hits,
            "poisoning_marker_count": poisoning_hits,
            "mojibake_marker_count": mojibake_hits,
        },
    }


def _is_acquisition_failure(raw: Dict[str, Any], normalized_text: str) -> bool:
    declared_status = str(raw.get("acquisition_status") or "").strip().casefold()
    if declared_status in ACQUISITION_FAILURE_STATUSES:
        return True
    raw_http_status = raw.get("http_status")
    try:
        http_status = int(raw_http_status) if raw_http_status is not None else None
    except (TypeError, ValueError):
        http_status = None
    if http_status in ACQUISITION_FAILURE_STATUS_CODES:
        return ACQUISITION_FAILURE_RE.search(normalized_text) is not None
    text_only_body = " ".join(HTML_TAG_RE.sub(" ", normalized_text).split())
    return (
        len(text_only_body.split()) <= 8
        and ACQUISITION_FAILURE_RE.fullmatch(text_only_body) is not None
    )


def _validity_reasons(raw_text: Any, normalized_text: str, raw: Dict[str, Any]) -> List[str]:
    """Return only closed text-validity failures for the source-agnostic profile."""
    reasons: List[str] = []
    visible = [character for character in normalized_text if not character.isspace()]
    if not isinstance(raw_text, str) or not visible:
        reasons.append(STAGE_A_PAYLOAD_ABSENCE_REASON)
    if raw.get("text_contract") == "violated":
        reasons.append(STAGE_A_TEXT_CONTRACT_REASON)
    corruption = sum(character == "\ufffd" or unicodedata.category(character) == "Cc" for character in visible)
    if visible and corruption / len(visible) >= 0.2:
        reasons.append(STAGE_A_CORRUPTION_REASON)
    if _is_acquisition_failure(raw, normalized_text):
        reasons.append(STAGE_A_ACQUISITION_FAILURE_REASON)
    return reasons


def process_candidate(
    raw: Dict[str, Any], *, index: int = 0, stage_a_policy: str = "text_only_v2"
) -> Dict[str, Any]:
    if stage_a_policy != "text_only_v2":
        raise ValueError(f"Unsupported Stage A policy: {stage_a_policy}")
    pii_context = str(raw.get("pii_context") or "general")
    normalization_context = str(raw.get("normalization_context") or "preserve")
    raw_text = raw.get("text")
    normalized = normalize_text(
        raw_text if isinstance(raw_text, str) else "",
        context=normalization_context,
    )
    hazards = detect_hazards(normalized["text"], pii_context=pii_context)
    hazards["diagnostics"]["audit_only"] = True
    rights = raw.get("rights") if isinstance(raw.get("rights"), dict) else {}
    raw_provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    provenance = {
        "source_name": str(raw_provenance.get("source_name") or raw.get("source_name") or "unknown"),
        "source_uri": str(raw_provenance.get("source_uri") or raw.get("source_uri") or "unknown"),
        "collected_at": str(raw_provenance.get("collected_at") or raw.get("collected_at") or "unknown"),
    }
    rights_status = str(rights.get("status") or "unknown")
    reasons = _validity_reasons(raw_text, normalized["text"], raw)

    record = {
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "stage_a_policy": stage_a_policy,
        "record_id": str(raw.get("record_id") or raw.get("id") or f"candidate-{index:06d}"),
        "text": normalized["text"],
        "provenance": {
            **provenance,
            "original_sha256": normalized["original_sha256"],
            "normalized_sha256": normalized["normalized_sha256"],
        },
        "language": raw.get("language") if isinstance(raw.get("language"), dict) else {"code": "und", "confidence": None},
        "artifact_context": raw.get("artifact_context") if isinstance(raw.get("artifact_context"), dict) else None,
        "rights": {"status": rights_status, "license": rights.get("license")},
        "hazards": hazards,
        "quarantine": {
            "status": "quarantined" if reasons else "release_candidate",
            "reasons": sorted(set(reasons)),
        },
        "stage_a_decision": {
            "accepted": not reasons,
            "trigger": "declared_text_integrity_or_audit_quarantine_reason" if reasons else "no_active_stage_a_quarantine_reason",
            "non_trigger_boundary": "normalized_text_meets_declared_integrity_conditions" if not reasons else "quarantine_reason_present",
            "reason_codes": sorted(set(reasons)),
            "token_delta_proxy": len(normalized["text"].split()) - len(str(raw.get("text") or "").split()),
            "utility_read": False,
            "benchmark_outcomes_read": False,
        },
        "transformations": normalized["transformations"],
        "normalization_context": normalization_context,
        "partition": raw.get("partition") if isinstance(raw.get("partition"), dict) else None,
    }
    record["release_eligibility"] = release_eligibility(record)
    return record
