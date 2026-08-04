#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_experiment_preflight import EXPECTED_STAGES, build_final_experiment_preflight


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(path: Path, ids: tuple[str, ...]) -> dict:
    output = path.with_suffix(".jsonl")
    output.write_text("".join(json.dumps({"chunk_uid": uid}) + "\n" for uid in ids), encoding="utf-8")
    return {
        "status": "curation_materialization_complete",
        "stage_contract": EXPECTED_STAGES,
        "stage_c_coverage": {
            "semantic_graph_consumed": True,
            "final_status": "pass",
            "complete_recheck_passed": True,
            "may_create_new_removal": False,
            "scientific_promotion_claimed": False,
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "source_pool_role_read": False,
        },
        "outputs": {"stage_c_curated": {"path": str(output), "sha256": _sha(output)}},
    }


def test_preflight_separates_experiment_readiness_from_scientific_release() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        normal_path, hard_path = root / "normal.json", root / "hard.json"
        release_path, training_path = root / "release.json", root / "training.json"
        normal = _arm(normal_path, ("a", "b"))
        hard = _arm(hard_path, ("a",))
        _write(normal_path, normal)
        _write(hard_path, hard)
        _write(release_path, {"framework_release": "blocked", "release_blockers": ["candidate"]})
        raw_source = root / "raw.jsonl"
        raw_source.write_text('{"text":"raw"}\n', encoding="utf-8")
        sources = {
            "raw_audited_natural": raw_source,
            "normal_natural": Path(normal["outputs"]["stage_c_curated"]["path"]),
            "hard_natural": Path(hard["outputs"]["stage_c_curated"]["path"]),
        }
        arms = {}
        for index, (name, count) in enumerate(zip(sources, (100, 80, 60), strict=True)):
            arm_path, blocks_path = root / f"{name}.jsonl", root / f"{name}.pt"
            arm_path.write_text(name, encoding="utf-8")
            blocks_path.write_bytes(bytes([index]))
            arms[name] = {
                "stream_tokens": count,
                "source_path": str(sources[name]), "source_sha256": _sha(sources[name]),
                "arm_path": str(arm_path), "arm_sha256": _sha(arm_path),
                "blocks_path": str(blocks_path), "blocks_sha256": _sha(blocks_path),
            }
        _write(training_path, {
            "status": "tokenizer_materialization_complete",
            "arms": arms,
            "selector_boundary": {
                "utility_read": False,
                "benchmark_outcomes_read": False,
                "target_token_fraction_read": False,
            },
        })

        report = build_final_experiment_preflight(
            normal_path, hard_path, release_path, training_path
        )

    assert report["framework_materialization_ready"] is True
    assert report["external_confirmatory_ready"] is True
    assert report["paper_claim_ready"] is False
    assert report["production_release_ready"] is False
    assert report["hard_subset_or_equal_normal"] is True


if __name__ == "__main__":
    test_preflight_separates_experiment_readiness_from_scientific_release()
    print("[final-experiment-preflight-v1] readiness boundaries: pass")
