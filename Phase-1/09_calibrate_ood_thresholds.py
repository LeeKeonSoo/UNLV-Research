#!/usr/bin/env python3
"""Calibrate OOD thresholds using calibration split labels."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
IDENTITY_CONFIG_PATH = PROJECT_DIR / "configs" / "metric_identity_v1.json"
OOD_LABELS_PATH = PROJECT_DIR / "validation" / "labels" / "ood_labels_v1.csv"
CALIB_REPORT_PATH = OUTPUT_DIR / "validation" / "ood_calibration_report.json"

REQUIRED_COLUMNS = ["dataset", "source", "doc_id", "chunk_id", "gold_in_domain", "split"]


def _safe_float(v: Any):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _to_bool(v: str) -> bool | None:
    raw = str(v or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_calibration_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        rows = []
        for row in reader:
            if str(row.get("split") or "").strip() != "calibration":
                continue
            gold = _to_bool(row.get("gold_in_domain", ""))
            if gold is None:
                continue
            rows.append(
                {
                    "dataset": str(row["dataset"]).strip(),
                    "source": str(row["source"]).strip(),
                    "doc_id": str(row["doc_id"]).strip(),
                    "chunk_id": int(row["chunk_id"]),
                    "gold_in_domain": gold,
                }
            )
    return rows


def _dataset_outputs(manifest: Dict[str, Any]) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for dataset_name, meta in (manifest.get("dataset_versions_and_counts", {}) or {}).items():
        if not isinstance(meta, dict):
            continue
        output_file = Path(str(meta.get("output_file") or f"{dataset_name}_analysis.jsonl")).name
        outputs[dataset_name] = OUTPUT_DIR / output_file
    return outputs


def _extract_top_scores(record: Dict[str, Any]) -> Tuple[float | None, float | None]:
    ood = record.get("ood") or {}
    top1 = _safe_float(ood.get("top1_similarity"))
    top2 = _safe_float(ood.get("top2_similarity"))
    if top1 is not None:
        if top2 is None:
            top2 = 0.0
        return top1, top2

    labels = record.get("domain_labels") or {}
    vals = [_safe_float(v) for v in labels.values()] if isinstance(labels, dict) else []
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    vals.sort(reverse=True)
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    return top1, top2


def _attach_predictions(rows: List[Dict[str, Any]], outputs: Dict[str, Path]) -> List[Dict[str, Any]]:
    needed = {
        (r["dataset"], r["source"], r["doc_id"], r["chunk_id"]): r
        for r in rows
    }

    for dataset_name, path in outputs.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                key = (
                    dataset_name,
                    str(rec.get("source") or dataset_name),
                    str(rec.get("doc_id") or ""),
                    int(rec.get("chunk_id") or 0),
                )
                row = needed.get(key)
                if row is None:
                    continue
                t1, t2 = _extract_top_scores(rec)
                row["top1_similarity"] = t1
                row["top2_similarity"] = t2
                row["margin"] = (t1 - t2) if t1 is not None and t2 is not None else None

    attached = []
    for row in rows:
        if row.get("top1_similarity") is None or row.get("margin") is None:
            continue
        attached.append(row)
    return attached


def _predict_in_domain(top1: float, margin: float, sim_thr: float, margin_thr: float) -> bool:
    return top1 >= sim_thr and margin >= margin_thr


def _binary_macro_f1(golds: List[bool], preds: List[bool]) -> float:
    def class_f1(cls: bool) -> float:
        tp = sum(1 for g, p in zip(golds, preds) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(golds, preds) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(golds, preds) if g == cls and p != cls)
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        if tp == 0 and fp == 0 and fn == 0:
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    return 0.5 * (class_f1(True) + class_f1(False))


def _false_accept_rate(golds: List[bool], preds: List[bool]) -> float:
    neg = sum(1 for g in golds if g is False)
    if neg == 0:
        return 0.0
    fp = sum(1 for g, p in zip(golds, preds) if g is False and p is True)
    return fp / neg


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate OOD thresholds on calibration split.")
    parser.add_argument("--identity-config", type=Path, default=IDENTITY_CONFIG_PATH)
    parser.add_argument("--labels", type=Path, default=OOD_LABELS_PATH)
    parser.add_argument("--report", type=Path, default=CALIB_REPORT_PATH)
    args = parser.parse_args()

    manifest = _load_json(MANIFEST_PATH)
    config = _load_json(args.identity_config)
    rows = _load_calibration_rows(args.labels)
    if not rows:
        raise SystemExit("No calibration rows with labeled gold_in_domain found.")

    outputs = _dataset_outputs(manifest)
    scored = _attach_predictions(rows, outputs)
    if not scored:
        raise SystemExit("No calibration rows matched analysis outputs.")

    golds = [bool(r["gold_in_domain"]) for r in scored]

    best = None
    best_payload = None
    total_grid = 0

    for sim_step in range(15, 36):
        sim_thr = sim_step / 100.0
        for margin_step in range(0, 11):
            margin_thr = margin_step / 100.0
            for near_step in range(5, 21):
                near_thr = near_step / 100.0
                total_grid += 1

                preds = [
                    _predict_in_domain(r["top1_similarity"], r["margin"], sim_thr, margin_thr)
                    for r in scored
                ]
                far = _false_accept_rate(golds, preds)
                if far > 0.10:
                    continue
                macro_f1 = _binary_macro_f1(golds, preds)
                cand = (macro_f1, -far, sim_thr, margin_thr, near_thr)
                if best is None or cand > best:
                    best = cand
                    best_payload = {
                        "in_domain_sim": sim_thr,
                        "margin_min": margin_thr,
                        "ood_near_sim": near_thr,
                        "macro_f1": round(macro_f1, 6),
                        "false_accept_rate": round(far, 6),
                    }

    if best_payload is None:
        raise SystemExit("No threshold candidate satisfied false-accept-rate constraint <= 0.10")

    config.setdefault("ood_thresholds", {})
    config["ood_thresholds"]["in_domain_sim"] = best_payload["in_domain_sim"]
    config["ood_thresholds"]["margin_min"] = best_payload["margin_min"]
    config["ood_thresholds"]["ood_near_sim"] = best_payload["ood_near_sim"]
    _save_json(args.identity_config, config)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "identity_version": config.get("identity_version", "v1"),
        "calibration_rows_total": len(rows),
        "calibration_rows_matched": len(scored),
        "grid_search_candidates": total_grid,
        "constraint": {
            "false_accept_rate_max": 0.10,
        },
        "best_thresholds": best_payload,
        "updated_config_path": str(args.identity_config),
    }
    _save_json(args.report, report)

    print("[09] OOD thresholds calibrated:")
    print(json.dumps(best_payload, indent=2))
    print(f"[09] Updated config: {args.identity_config}")
    print(f"[09] Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
