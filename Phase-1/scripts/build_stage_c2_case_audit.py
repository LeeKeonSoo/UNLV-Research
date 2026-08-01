from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage_c2_proxy_lm_scoring import read_jsonl_records


JsonMap = dict[str, Any]
SOURCES = {
    "code_raw_like": ("code", Path(r"D:\UNLV-Research\code_5m_corpus_v2\current_framework_7m_v1\baseline\stage_b_pass_chunks.jsonl")),
    "math_raw_like": ("math", Path(r"D:\UNLV-Research\cross_domain_stress\abc_curation_openwebmath_5m_technical_math_v2\stage_b_pass_chunks.jsonl")),
    "general_text_raw_like": ("general", Path(r"D:\UNLV-Research\cross_domain_stress\abc_curation_fineweb_edu_v1\stage_b_pass_chunks.jsonl")),
}


def _text_records(path: Path) -> dict[str, JsonMap]:
    rows, _ = read_jsonl_records(path)
    return {str(row["chunk_uid"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-label-based Stage C-2 ablation case packets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    ablation = json.loads((arguments.root / "reports" / "stage_c2_ablation_report.json").read_text(encoding="utf-8"))
    packets: list[JsonMap] = []
    for corpus_id, arms in ablation["arms"].items():
        label, source_path = SOURCES[corpus_id]
        text_by_uid = _text_records(source_path)
        evidence_by_uid = {
            row["chunk_uid"]: row
            for row in [json.loads(line) for line in (arguments.root / "calibrated" / f"{label}_proxy_evidence.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        }
        for arm, result in arms.items():
            for chunk_uid in result.get("rejected_chunk_uids", []):
                row = text_by_uid[str(chunk_uid)]
                evidence = evidence_by_uid[str(chunk_uid)]
                text = str(row["text"])
                packets.append({
                    "corpus": corpus_id,
                    "arm": arm,
                    "chunk_uid": chunk_uid,
                    "reason_codes": result["reason_codes"],
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text_preview": text[:480],
                    "token_proxy": row.get("token_proxy"),
                    "semantic_bucket": evidence["semantic_bucket"],
                    "familiarity": evidence["familiarity"],
                    "novelty": evidence["novelty"],
                    "gradient_alignment": evidence["gradient_alignment"],
                    "promotion_risk": "no_semantic_family_evidence" if arm == "proxy_only" else "requires_structural_case_review",
                })
    report = {
        "schema_version": "stage-c2-ablation-case-audit-v1",
        "status": "non_label_based_case_audit_not_a_promotion_decision",
        "forbidden": ["human_quality_label", "source_identity", "benchmark_outcomes"],
        "case_packets": packets,
        "summary": {"packets": len(packets), "proxy_only_packets": sum(packet["arm"] == "proxy_only" for packet in packets), "joint_packets": sum(packet["arm"] == "joint" for packet in packets)},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
