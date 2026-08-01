#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_structural_evidence import FeatureSchema
from scripts.train_math_structural_heads import load_text_rows, positive_scores, write_scores


@dataclass(frozen=True, slots=True)
class FrozenScorerError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_hashes(model_paths: dict[str, Path], expected_hashes: dict[str, str]) -> dict[str, str]:
    observed = {head: _sha256(path) for head, path in model_paths.items()}
    if observed != expected_hashes:
        raise FrozenScorerError("Frozen structural model hash mismatch")
    return observed


def main() -> int:
    import joblib

    parser = argparse.ArgumentParser(description="Score text with frozen Math structural-head models.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_paths = {
        "substantive_payload": args.model_dir / "substantive_payload.joblib",
        "coherence_completeness": args.model_dir / "coherence_completeness.joblib",
    }
    expected = {head: str(value) for head, value in manifest["structural_model_artifact_sha256"].items()}
    hashes = verify_model_hashes(model_paths, expected)
    raw_feature_schema = manifest.get("structural_feature_schema_version", "v1")
    if raw_feature_schema not in {"v1", "v2"}:
        raise FrozenScorerError("Unsupported structural feature schema")
    feature_schema: FeatureSchema = raw_feature_schema
    rows = load_text_rows(args.input, "math-clean-control")
    payload_model = joblib.load(model_paths["substantive_payload"])
    coherence_model = joblib.load(model_paths["coherence_completeness"])
    write_scores(
        args.output,
        rows,
        positive_scores(payload_model, rows, feature_schema),
        positive_scores(coherence_model, rows, feature_schema),
        hashes,
    )
    print(json.dumps({"status": "frozen_structural_heads_scored", "records": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
