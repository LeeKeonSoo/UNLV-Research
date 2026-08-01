from __future__ import annotations

import hashlib
import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from route_conditioned_quality import (
    HEAD_NAMES,
    KNOWN_ROUTES,
    EvidenceHead,
    HeadName,
    HeadOutcome,
    QualityUnit,
    RouteEvidenceBundle,
    evaluate_route_conditioned_quality,
)


ROOT: Final = Path(__file__).resolve().parent
EvidenceState = Literal["evidence_ready", "blocked", "unsupported"]
STATE_OUTCOMES: Final = {
    "blocked": "indeterminate",
    "unsupported": "missing",
}
FORBIDDEN_INPUTS: Final = frozenset(
    {
        "source_identity",
        "source_reputation",
        "dataset_identity",
        "path",
        "human_quality_label",
        "Utility",
        "NLL",
        "benchmark_outcomes",
        "target_retention_fraction",
        "domain_quota",
        "weighted_quality_score",
    }
)


class CandidateEvidenceContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateHeadSpec:
    route: str
    head: HeadName
    evidence_state: EvidenceState
    outcome: HeadOutcome
    provider_id: str
    provider_version: str
    artifact: str
    artifact_sha256: str
    blocking_reason: str | None
    negative_boundary_id: str | None

    @classmethod
    def from_mapping(
        cls, route: str, head: HeadName, raw: dict[str, Any]
    ) -> "CandidateHeadSpec":
        state = str(raw.get("evidence_state"))
        outcome = str(raw.get("outcome"))
        if state not in {"evidence_ready", "blocked", "unsupported"}:
            raise CandidateEvidenceContractError(f"Unsupported evidence state: {state}")
        if state != "evidence_ready" and outcome in {"pass", "negative"}:
            raise CandidateEvidenceContractError(
                "blocked or unsupported evidence cannot declare pass or negative"
            )
        if state in STATE_OUTCOMES and outcome != STATE_OUTCOMES[state]:
            raise CandidateEvidenceContractError(
                f"{state} evidence must emit {STATE_OUTCOMES[state]}"
            )
        if state == "evidence_ready" and outcome not in {"pass", "negative"}:
            raise CandidateEvidenceContractError(
                "evidence_ready requires pass or a named negative outcome"
            )
        boundary = raw.get("negative_boundary_id")
        if outcome == "negative" and not boundary:
            raise CandidateEvidenceContractError("negative evidence requires a named boundary")
        if outcome != "negative" and boundary is not None:
            raise CandidateEvidenceContractError("only negative evidence may name a boundary")
        required = ("provider_id", "provider_version", "artifact", "artifact_sha256")
        if not all(isinstance(raw.get(field), str) and raw[field] for field in required):
            raise CandidateEvidenceContractError("candidate head identity must be complete")
        blocking = raw.get("blocking_reason")
        if state != "evidence_ready" and not isinstance(blocking, str):
            raise CandidateEvidenceContractError("blocked evidence requires a reason")
        return cls(
            route,
            head,
            state,  # type: ignore[arg-type]
            outcome,  # type: ignore[arg-type]
            raw["provider_id"],
            raw["provider_version"],
            raw["artifact"],
            raw["artifact_sha256"],
            blocking,
            boundary,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _evidence_head(spec: CandidateHeadSpec) -> EvidenceHead:
    artifact = _artifact_path(spec.artifact)
    if not artifact.is_file():
        raise CandidateEvidenceContractError(f"evidence artifact is missing: {spec.artifact}")
    actual = _sha256(artifact)
    if actual != spec.artifact_sha256:
        raise CandidateEvidenceContractError(
            f"artifact SHA-256 mismatch for {spec.route}.{spec.head}"
        )
    return EvidenceHead(
        name=spec.head,
        outcome=spec.outcome,
        evidence_id=f"{spec.provider_id}:{spec.route}:{spec.head}",
        provider_version=spec.provider_version,
        artifact_sha256=actual,
        negative_boundary_id=spec.negative_boundary_id,
    )


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "route-quality-evidence-candidates-v1":
        raise CandidateEvidenceContractError("unexpected candidate manifest schema")
    if raw.get("runtime_activation") is not False:
        raise CandidateEvidenceContractError("candidate evidence cannot be runtime-active")
    if raw.get("selector_inputs") != ["chunk_text", "frozen_candidate_registry"]:
        raise CandidateEvidenceContractError("candidate selector input contract changed")
    if set(raw.get("forbidden_inputs") or ()) != FORBIDDEN_INPUTS:
        raise CandidateEvidenceContractError("candidate forbidden-input boundary changed")
    if set(raw.get("routes") or {}) != set(KNOWN_ROUTES):
        raise CandidateEvidenceContractError("candidate manifest must declare every route")
    return raw, _sha256(path)


def _bundle(route: str, raw: dict[str, Any], profile_id: str, profile_sha: str) -> RouteEvidenceBundle:
    if set(raw) != set(HEAD_NAMES):
        raise CandidateEvidenceContractError(f"{route} must declare exactly two Quality heads")
    specs = {
        head: CandidateHeadSpec.from_mapping(route, head, raw[head])
        for head in HEAD_NAMES
    }
    return RouteEvidenceBundle(
        route=route,  # type: ignore[arg-type]
        substantive_payload=_evidence_head(specs["substantive_payload"]),
        route_specific_evidence=_evidence_head(specs["route_specific_evidence"]),
        profile_id=f"{profile_id}:{route}",
        profile_sha256=profile_sha,
    )


def evaluate_registered_candidate(text: str, manifest_path: Path) -> dict[str, Any]:
    """Evaluate frozen candidate evidence without granting runtime authority."""
    manifest, manifest_sha = _load_manifest(manifest_path)
    bundles = tuple(
        _bundle(route, manifest["routes"][route], manifest["profile_id"], manifest_sha)
        for route in KNOWN_ROUTES
    )
    decision = evaluate_route_conditioned_quality(QualityUnit(text, bundles))
    evaluated = set(decision.evaluated_routes)
    outcomes = {
        bundle.route: {
            "substantive_payload": bundle.substantive_payload.outcome,
            "route_specific_evidence": bundle.route_specific_evidence.outcome,
        }
        for bundle in bundles
        if bundle.route in evaluated
    }
    decision_map = asdict(decision)
    for field in (
        "routed_routes",
        "evaluated_routes",
        "qualifying_routes",
        "negative_routes",
        "evidence_artifact_hashes",
    ):
        decision_map[field] = list(decision_map[field])
    return {
        "schema_version": "route-quality-candidate-evaluation-v1",
        "candidate_registry_sha256": manifest_sha,
        "routed_routes": list(decision.routed_routes),
        "head_outcomes": outcomes,
        "decision": decision_map,
        "runtime_authorized": False,
        "selector_inputs": ["chunk_text", "frozen_candidate_registry"],
        "utility_read": False,
        "benchmark_outcomes_read": False,
    }


def audit_candidate_registry(manifest_path: Path) -> dict[str, Any]:
    """Validate every registered artifact and summarize candidate availability."""
    manifest, manifest_sha = _load_manifest(manifest_path)
    routes: dict[str, Any] = {}
    state_counts = {"evidence_ready": 0, "blocked": 0, "unsupported": 0}
    ready_routes: list[str] = []
    for route in KNOWN_ROUTES:
        bundle = _bundle(
            route, manifest["routes"][route], manifest["profile_id"], manifest_sha
        )
        route_states: dict[str, str] = {}
        for head in HEAD_NAMES:
            spec = CandidateHeadSpec.from_mapping(
                route, head, manifest["routes"][route][head]
            )
            state_counts[spec.evidence_state] += 1
            route_states[head] = spec.evidence_state
        if all(
            evidence.outcome == "pass"
            for evidence in (bundle.substantive_payload, bundle.route_specific_evidence)
        ):
            ready_routes.append(route)
        routes[route] = {
            "states": route_states,
            "outcomes": {
                "substantive_payload": bundle.substantive_payload.outcome,
                "route_specific_evidence": bundle.route_specific_evidence.outcome,
            },
        }
    return {
        "schema_version": "route-quality-evidence-candidate-audit-v1",
        "candidate_registry_sha256": manifest_sha,
        "summary": {
            "routes": len(routes),
            "evidence_ready_heads": state_counts["evidence_ready"],
            "blocked_heads": state_counts["blocked"],
            "unsupported_heads": state_counts["unsupported"],
        },
        "ready_routes": ready_routes,
        "routes": routes,
        "runtime_authorized": False,
        "claim_boundary": "Candidate artifact availability only; no runtime or downstream-effectiveness claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen route Quality candidates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_candidate_registry(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
