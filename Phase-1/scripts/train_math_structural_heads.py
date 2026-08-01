#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_structural_evidence import FeatureSchema, coherence_corruptions, extract_structural_features, payload_corruptions


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class TextRow:
    record_id: str
    source_group: str
    text: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.record_id or not self.source_group or not self.text or self.token_count <= 0:
            raise StructuralHeadContractError("Structural-head rows require identity, source, text, and positive token count")


@dataclass(frozen=True, slots=True)
class StructuralHeadContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def split_source_roles(
    rows: tuple[TextRow, ...], training_sources: frozenset[str], calibration_sources: frozenset[str]
) -> tuple[tuple[TextRow, ...], tuple[TextRow, ...]]:
    if training_sources & calibration_sources:
        raise StructuralHeadContractError("Training and calibration source groups must be disjoint")
    declared = training_sources | calibration_sources
    observed = {row.source_group for row in rows}
    undeclared = sorted(observed - declared)
    if undeclared:
        raise StructuralHeadContractError(f"Clean controls contain undeclared source groups: {', '.join(undeclared)}")
    training = tuple(row for row in rows if row.source_group in training_sources)
    calibration = tuple(row for row in rows if row.source_group in calibration_sources)
    if not training or not calibration:
        raise StructuralHeadContractError("Both structural-head source roles require records")
    return training, calibration


def load_text_rows(path: Path, default_source: str) -> tuple[TextRow, ...]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id, text = raw.get("record_id"), raw.get("text")
            token_count = raw.get("token_count", raw.get("token_proxy"))
            if not isinstance(record_id, str) or not isinstance(text, str) or not isinstance(token_count, int):
                raise StructuralHeadContractError("Input rows require record_id, text, and token count")
            rows.append(TextRow(record_id, str(raw.get("source_group") or default_source), text, token_count))
    return tuple(rows)


def feature_vector(text: str, schema: FeatureSchema = "v1") -> tuple[float, ...]:
    return extract_structural_features(text).vector(schema)


def _examples(
    rows: tuple[TextRow, ...], head: str, feature_schema: FeatureSchema = "v1"
) -> tuple[list[tuple[float, ...]], list[int]]:
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    for row in rows:
        features.append(feature_vector(row.text, feature_schema))
        labels.append(1)
        corruptions = payload_corruptions(row.text, row.record_id) if head == "substantive_payload" else coherence_corruptions(row.text, row.record_id)
        for corrupted in corruptions:
            features.append(feature_vector(corrupted.text, feature_schema))
            labels.append(0)
    return features, labels


def _train_model(rows: tuple[TextRow, ...], head: str, feature_schema: FeatureSchema = "v1"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = _examples(rows, head, feature_schema)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0)),
        ]
    )
    model.fit(features, labels)
    return model


def positive_scores(
    model, rows: tuple[TextRow, ...], feature_schema: FeatureSchema = "v1"
) -> tuple[float, ...]:
    probabilities = model.predict_proba([feature_vector(row.text, feature_schema) for row in rows])
    return tuple(float(row[1]) for row in probabilities)


def _heldout_metrics(
    model, rows: tuple[TextRow, ...], head: str, feature_schema: FeatureSchema = "v1"
) -> dict[str, JsonValue]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    features, labels = _examples(rows, head, feature_schema)
    scores = tuple(float(row[1]) for row in model.predict_proba(features))
    positive_scores = tuple(score for score, label in zip(scores, labels, strict=True) if label == 1)
    negative_scores = tuple(score for score, label in zip(scores, labels, strict=True) if label == 0)
    return {
        "records": len(rows),
        "examples": len(labels),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "positive_score_min": min(positive_scores),
        "positive_score_median": sorted(positive_scores)[len(positive_scores) // 2],
        "negative_score_max": max(negative_scores),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_scores(
    path: Path,
    rows: tuple[TextRow, ...],
    payload_scores: tuple[float, ...],
    coherence_scores: tuple[float, ...],
    artifact_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row, payload, coherence in zip(rows, payload_scores, coherence_scores, strict=True):
            target.write(
                json.dumps(
                    {
                        "schema_version": "math-structural-head-score-v1",
                        "record_id": row.record_id,
                        "source_group": row.source_group,
                        "token_count": row.token_count,
                        "substantive_payload": payload,
                        "coherence_completeness": coherence,
                        "provider_artifact_sha256": artifact_hashes,
                        "status": "development_candidate_no_runtime_authority",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    import joblib
    import sklearn

    parser = argparse.ArgumentParser(description="Train source-disjoint Math structural evidence candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_feature_schema = config.get("feature_schema_version", "v1")
    if raw_feature_schema not in {"v1", "v2"}:
        raise StructuralHeadContractError("Unsupported structural feature schema")
    feature_schema: FeatureSchema = raw_feature_schema
    clean = load_text_rows(Path(config["clean_control_text"]), "math-clean-control")
    training, calibration = split_source_roles(
        clean, frozenset(config["training_source_groups"]), frozenset(config["calibration_source_groups"])
    )
    candidate = load_text_rows(Path(config["candidate_text"]), "openwebmath-candidate")
    payload_model = _train_model(training, "substantive_payload", feature_schema)
    coherence_model = _train_model(training, "coherence_completeness", feature_schema)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    payload_path, coherence_path = args.model_dir / "substantive_payload.joblib", args.model_dir / "coherence_completeness.joblib"
    joblib.dump(payload_model, payload_path)
    joblib.dump(coherence_model, coherence_path)
    hashes = {"substantive_payload": _sha256(payload_path), "coherence_completeness": _sha256(coherence_path)}
    write_scores(args.calibration_output, calibration, positive_scores(payload_model, calibration, feature_schema), positive_scores(coherence_model, calibration, feature_schema), hashes)
    write_scores(args.candidate_output, candidate, positive_scores(payload_model, candidate, feature_schema), positive_scores(coherence_model, candidate, feature_schema), hashes)
    report: dict[str, JsonValue] = {
        "schema_version": "math-structural-head-development-report-v1",
        "status": "development_candidate_pending_four_head_calibration",
        "training_records": len(training),
        "calibration_records": len(calibration),
        "candidate_records": len(candidate),
        "training_source_groups": sorted({row.source_group for row in training}),
        "calibration_source_groups": sorted({row.source_group for row in calibration}),
        "heldout_metrics": {
            "substantive_payload": _heldout_metrics(payload_model, calibration, "substantive_payload", feature_schema),
            "coherence_completeness": _heldout_metrics(coherence_model, calibration, "coherence_completeness", feature_schema),
        },
        "model_artifact_sha256": hashes,
        "feature_schema_version": feature_schema,
        "sklearn_version": sklearn.__version__,
        "claim_boundary": config["claim_boundary"],
        "target_retention_fraction_used": False,
        "external_results_visible": False,
        "runtime_activation": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["heldout_metrics"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
