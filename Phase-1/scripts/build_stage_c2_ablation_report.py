from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage_c2_model_relative_selector import select_model_relative_candidates


JsonMap = dict[str, Any]


def _rows(path: Path) -> list[JsonMap]:
    evidence = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        {
            "chunk_uid": row["chunk_uid"],
            "text": "",
            "stage_c2_proxy_evidence": {key: value for key, value in row.items() if key != "chunk_uid"},
        }
        for row in evidence
    ]


def build_report(root: Path) -> JsonMap:
    config: JsonMap = {
        "semantic_index": {"cosine_threshold": 0.98},
        "evidence_thresholds": {"minimum_familiarity": 0.8, "maximum_novelty": 0.2, "maximum_gradient_alignment": 0.05},
    }
    corpus_files = {"code_raw_like": "code", "math_raw_like": "math", "general_text_raw_like": "general"}
    report: JsonMap = {
        "schema_version": "stage-c2-ablation-report-v1",
        "status": "development_only_not_a_promotion_decision",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "arms": {},
    }
    for corpus_id, label in corpus_files.items():
        rows = _rows(root / "calibrated" / f"{label}_proxy_evidence.jsonl")
        arms: JsonMap = {"off": {"candidate_removed_chunks": 0, "reason_codes": []}}
        for mode in ("semantic_only", "proxy_only", "joint"):
            _, rejected, audit = select_model_relative_candidates(rows, {**config, "ablation_mode": mode})
            arms[mode] = {
                "candidate_removed_chunks": audit["candidate_removed_chunks"],
                "reason_codes": sorted({row["stage_c2_selection"].get("removed_reason") for row in rejected if row["stage_c2_selection"].get("removed_reason")}),
                "rejected_chunk_uids": [row["chunk_uid"] for row in rejected],
            }
        report["arms"][corpus_id] = arms
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a frozen Stage C-2 ablation report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
