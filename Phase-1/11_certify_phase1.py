#!/usr/bin/env python3
"""Strict closeout certification for Phase-1 metric identity."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation"

IDENTITY_CONFIG_PATH = PROJECT_DIR / "configs" / "metric_identity_v1.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
MAIN_SCORE_PATH = VALIDATION_DIR / "gate_scores_main.json"
TRANSFER_SCORE_PATH = VALIDATION_DIR / "gate_scores_transfer.json"
FULL_REPORT_PATH = VALIDATION_DIR / "full_validation_report.json"

LABEL_DIR = PROJECT_DIR / "validation" / "labels"
DOMAIN_LABELS = LABEL_DIR / "domain_labels_v1.csv"
QUALITY_LABELS = LABEL_DIR / "quality_labels_v1.csv"
DIFF_LABELS = LABEL_DIR / "difficulty_sanity_v1.csv"
OOD_LABELS = LABEL_DIR / "ood_labels_v1.csv"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_float(v: Any):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _required_gate_paths(config: Dict[str, Any]) -> List[str]:
    paths = []
    for metric, gates in (config.get("gates") or {}).items():
        if not isinstance(gates, dict):
            continue
        for gate_name in gates:
            paths.append(f"{metric}.{gate_name}")
    return sorted(paths)


def _gate_entry(split_score: Dict[str, Any], gate_path: str):
    metric, gate = gate_path.split(".", 1)
    metric_map = (split_score.get("gate_scores") or {}).get(metric, {})
    if not isinstance(metric_map, dict):
        return None
    entry = metric_map.get(gate)
    if not isinstance(entry, dict):
        return None
    return entry


def _eval_gate(config: Dict[str, Any], entry: Dict[str, Any], gate_path: str) -> Tuple[bool, str]:
    metric, gate = gate_path.split(".", 1)
    gate_cfg = ((config.get("gates") or {}).get(metric) or {}).get(gate) or {}
    direction = str(gate_cfg.get("direction") or "gte")
    threshold = gate_cfg.get("threshold")

    if direction == "eq":
        actual = entry.get("pass")
        ok = actual == threshold
        return bool(ok), f"pass={actual} expected={threshold}"

    value = _safe_float(entry.get("value"))
    if value is None:
        return False, "value is null"

    thr = _safe_float(threshold)
    if thr is None:
        return False, "invalid threshold"

    if direction == "gte":
        ok = value >= thr
        return bool(ok), f"value={value} >= {thr}"
    if direction == "lte":
        ok = value <= thr
        return bool(ok), f"value={value} <= {thr}"

    return False, f"unsupported direction={direction}"


def _count_split_rows(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"calibration": 0, "main_eval": 0, "transfer_eval": 0}
    out = {"calibration": 0, "main_eval": 0, "transfer_eval": 0}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split") or "").strip()
            if split in out:
                out[split] += 1
    return out


def _extract_ci(summary: Dict[str, Any]) -> Dict[str, Any]:
    def ci(split_obj: Dict[str, Any], gate_path: str):
        e = _gate_entry(split_obj, gate_path)
        if not e:
            return None
        return e.get("ci95")

    return {
        "main_eval": {
            "domain.top1_accuracy": ci(summary["main"], "domain.top1_accuracy"),
            "domain.top3_recall": ci(summary["main"], "domain.top3_recall"),
            "quality.macro_precision": ci(summary["main"], "quality.macro_precision"),
            "difficulty.out_of_range_rate": ci(summary["main"], "difficulty.out_of_range_rate"),
        },
        "transfer_eval": {
            "domain.top1_accuracy": ci(summary["transfer"], "domain.top1_accuracy"),
            "domain.top3_recall": ci(summary["transfer"], "domain.top3_recall"),
            "quality.macro_precision": ci(summary["transfer"], "quality.macro_precision"),
            "difficulty.out_of_range_rate": ci(summary["transfer"], "difficulty.out_of_range_rate"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify Phase-1 strict closeout.")
    parser.add_argument("--identity-config", type=Path, default=IDENTITY_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--main-score", type=Path, default=MAIN_SCORE_PATH)
    parser.add_argument("--transfer-score", type=Path, default=TRANSFER_SCORE_PATH)
    parser.add_argument("--full-report", type=Path, default=FULL_REPORT_PATH)
    args = parser.parse_args()

    config = _load_json(args.identity_config)
    manifest = _load_json(args.manifest)
    main_score = _load_json(args.main_score)
    transfer_score = _load_json(args.transfer_score)

    required_paths = _required_gate_paths(config)
    transfer_policy = config.get("transfer_policy") or {}
    transfer_required = bool(transfer_policy.get("required", True))
    min_datasets = int(transfer_policy.get("min_datasets", 1))
    anchor_datasets = set(transfer_policy.get("anchor_datasets") or ["khan_academy", "tiny_textbooks"])

    failures: List[Dict[str, Any]] = []

    split_checks = [("main_eval", main_score)]
    if transfer_required:
        split_checks.append(("transfer_eval", transfer_score))

    for split_name, split_obj in split_checks:
        for gate_path in required_paths:
            entry = _gate_entry(split_obj, gate_path)
            if entry is None:
                failures.append(
                    {
                        "split": split_name,
                        "gate_path": gate_path,
                        "reason": "missing gate entry",
                    }
                )
                continue
            ok, detail = _eval_gate(config, entry, gate_path)
            if not ok:
                failures.append(
                    {
                        "split": split_name,
                        "gate_path": gate_path,
                        "reason": detail,
                        "entry": entry,
                    }
                )

    transfer_datasets = set(transfer_score.get("datasets_in_split") or [])
    observed_transfer = sorted(d for d in transfer_datasets if d not in anchor_datasets)
    transfer_policy_pass = (not transfer_required) or (len(observed_transfer) >= min_datasets)

    if not transfer_policy_pass:
        failures.append(
            {
                "split": "transfer_eval",
                "gate_path": "transfer_policy.min_datasets",
                "reason": f"observed_transfer_datasets={len(observed_transfer)} < required={min_datasets}",
                "observed_transfer_datasets": observed_transfer,
            }
        )

    status = "pass" if not failures else "fail"

    label_counts = {
        "domain_labels_v1": _count_split_rows(DOMAIN_LABELS),
        "quality_labels_v1": _count_split_rows(QUALITY_LABELS),
        "difficulty_sanity_v1": _count_split_rows(DIFF_LABELS),
        "ood_labels_v1": _count_split_rows(OOD_LABELS),
    }

    manifest["metric_identity_version"] = config.get("identity_version", "v1")
    manifest["gate_evidence"] = {
        "identity_config": str(args.identity_config),
        "main_gate_scores": str(args.main_score),
        "transfer_gate_scores": str(args.transfer_score),
    }
    manifest["manual_validation_counts"] = label_counts
    manifest["transfer_validation"] = {
        "required": transfer_required,
        "min_datasets": min_datasets,
        "anchor_datasets": sorted(anchor_datasets),
        "observed_datasets": sorted(transfer_datasets),
        "observed_transfer_datasets": observed_transfer,
        "pass": transfer_policy_pass,
    }
    manifest["certification_status"] = status
    manifest["certification_failed_gates"] = failures

    _save_json(args.manifest, manifest)

    if args.full_report.exists():
        full_report = _load_json(args.full_report)
    else:
        full_report = {
            "summary": {
                "passed": 0,
                "failed": 0,
                "total": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failed_items": [],
            },
            "results": [],
        }

    bundle = {"main": main_score, "transfer": transfer_score}
    full_report["gate_scores"] = bundle
    full_report["gate_confidence_intervals"] = _extract_ci(bundle)
    full_report["transfer_results"] = manifest["transfer_validation"]
    full_report["closeout_decision"] = {
        "status": status,
        "failed_gates": failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _save_json(args.full_report, full_report)

    print(f"[11] certification_status: {status}")
    print(f"[11] updated manifest: {args.manifest}")
    print(f"[11] updated report: {args.full_report}")

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
