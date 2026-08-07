#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import materialize, resolve_curation_mode
from quality_teacher_panel import PanelDecision, PolicyDecision, TeacherVote
from quality_teacher_runtime import PanelPolicyResult


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _all_pass_quality_scorer(rows, **_kwargs):
    results = {}
    for row in rows:
        uid = str(row["chunk_uid"])
        policy_results = []
        for policy_id in QUALITY_POLICY_IDS:
            votes = tuple(
                TeacherVote(
                    teacher_id=f"teacher-{index}",
                    policy_id=policy_id,
                    decision=PolicyDecision.PASS,
                    reason_codes=("fixture_pass",),
                )
                for index in range(3)
            )
            policy_results.append(
                PanelPolicyResult(
                    policy_id=policy_id,
                    decision=PanelDecision.PASS,
                    first_pass=votes,
                    second_pass=None,
                )
            )
        results[uid] = tuple(policy_results)
    return results, {"fixture": True, "input_chunks": len(rows)}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hard_mode_is_development_only_until_ablation_promotion() -> None:
    try:
        resolve_curation_mode("hard")
    except RuntimeError as error:
        assert "development" in str(error)
    else:
        raise AssertionError("Hard mode must remain fail-closed for a production run.")

    mode = resolve_curation_mode("hard", execution_scope="development")
    assert mode["mode"] == "hard"
    assert mode["profile_id"] == "hard_structural_v1"
    assert mode["authorization"] == "development_candidate_release_blocked"
    assert mode["effective_policy_sha256"]


def test_final_hard_runtime_does_not_execute_unpromoted_span_candidates() -> None:
    repeated = "This repeated transport template documents retries, timeouts, authentication, and stable generated client behavior."
    rows = [
        {
            "id": "a",
            "text": "# SPDX-License-Identifier: Apache-2.0\n# Licensed under the Apache License\n\n"
            f"{repeated}\n\nThe implementation preserves an independently useful explanation of recovery behavior.",
        },
        {
            "id": "b",
            "text": f"{repeated}\n\nA separate payload explains token refresh and a distinct authentication failure path.",
        },
        {
            "id": "c",
            "text": "def add(left, right):\n    return left + right\n\n# SPDX-License-Identifier: Apache-2.0\n"
            "# Licensed under the Apache License\n\nprint(add(2, 3))",
        },
        {"id": "short-snippet", "text": "x = 1"},
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.jsonl"
        output_dir = root / "out"
        config_path = root / "config.json"
        _write_jsonl(input_path, rows)
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "hard",
                    "execution_scope": "development",
                    "input": {"candidate_files": [str(input_path)], "text_fields": ["text"], "defaults": {}},
                    "output_dir": str(output_dir),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {
                        "minimum_residual_chars": 40,
                        "no_binding_budget_action": "selection_without_binding_budget",
                    },
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8",
        )

        report = materialize(config_path, quality_scorer=_all_pass_quality_scorer)
        curated = [
            json.loads(line)
            for line in (output_dir / "stage_c_curated_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    assert report["curation_mode"]["authorization"] == "development_candidate_release_blocked"
    assert "stage_b_hard_span_transformations" not in report["summary"]
    assert "hard_runtime_audit" not in report
    assert report["summary"]["stage_b_total_span_transformations"] == 0
    assert report["coverage_impact_audit"]["residual_payload"]["span_rewrite_active"] is False
    assert report["coverage_impact_audit"]["residual_payload"]["passed"] is True
    assert "SPDX-License-Identifier" in next(row["text"] for row in curated if row["chunk_uid"].startswith("a::"))
    assert repeated in next(row["text"] for row in curated if row["chunk_uid"].startswith("b::"))
    assert next(row["text"] for row in curated if row["chunk_uid"].startswith("short-snippet::")) == "x = 1"


if __name__ == "__main__":
    test_hard_mode_is_development_only_until_ablation_promotion()
    test_final_hard_runtime_does_not_execute_unpromoted_span_candidates()
    print("[hard-structural-runtime] final Hard excludes unpromoted span candidates: pass")
