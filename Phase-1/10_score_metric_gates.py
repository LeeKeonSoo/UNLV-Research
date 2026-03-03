#!/usr/bin/env python3
"""Score Phase-1 metric gates for main_eval and transfer_eval splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
IDENTITY_CONFIG_PATH = PROJECT_DIR / "configs" / "metric_identity_v1.json"
LABEL_DIR = PROJECT_DIR / "validation" / "labels"

DOMAIN_LABELS = LABEL_DIR / "domain_labels_v1.csv"
QUALITY_LABELS = LABEL_DIR / "quality_labels_v1.csv"
DIFF_LABELS = LABEL_DIR / "difficulty_sanity_v1.csv"

MAIN_SCORE_PATH = OUTPUT_DIR / "validation" / "gate_scores_main.json"
TRANSFER_SCORE_PATH = OUTPUT_DIR / "validation" / "gate_scores_transfer.json"

SPLITS = ("main_eval", "transfer_eval")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _to_bool(v: str) -> bool | None:
    raw = str(v or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return None


def _safe_float(v: Any):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _bootstrap_ci(samples: List[Any], scorer, iterations: int, ci: float, seed: int):
    if not samples:
        return [None, None]
    n = len(samples)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        boot = [samples[rng.randrange(n)] for _ in range(n)]
        score = scorer(boot)
        if score is None:
            continue
        values.append(float(score))
    if not values:
        return [None, None]
    values.sort()
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    lo_idx = max(0, min(len(values) - 1, int(lo_p * (len(values) - 1))))
    hi_idx = max(0, min(len(values) - 1, int(hi_p * (len(values) - 1))))
    return [round(values[lo_idx], 6), round(values[hi_idx], 6)]


def _load_csv(path: Path, required_columns: List[str]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        return list(reader)


def _key(dataset: str, source: str, doc_id: str, chunk_id: int) -> Tuple[str, str, str, int]:
    return dataset, source, doc_id, int(chunk_id)


def _required_keys_by_dataset(
    domain_rows: List[Dict[str, str]],
    quality_rows: List[Dict[str, str]],
    diff_rows: List[Dict[str, str]],
) -> Dict[str, set]:
    by_dataset: Dict[str, set] = {}
    for rows in (domain_rows, quality_rows, diff_rows):
        for row in rows:
            split = str(row.get("split") or "").strip()
            if split not in SPLITS:
                continue
            dataset = str(row["dataset"]).strip()
            source = str(row["source"]).strip()
            doc_id = str(row["doc_id"]).strip()
            chunk_id = int(row["chunk_id"])
            by_dataset.setdefault(dataset, set()).add(_key(dataset, source, doc_id, chunk_id))
    return by_dataset


def _dataset_outputs(manifest: Dict[str, Any]) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for dataset_name, meta in (manifest.get("dataset_versions_and_counts", {}) or {}).items():
        if not isinstance(meta, dict):
            continue
        out = Path(str(meta.get("output_file") or f"{dataset_name}_analysis.jsonl")).name
        outputs[dataset_name] = OUTPUT_DIR / out
    return outputs


def _load_required_records(keys_by_dataset: Dict[str, set], outputs: Dict[str, Path]) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for dataset, keys in keys_by_dataset.items():
        path = outputs.get(dataset)
        if path is None or not path.exists() or not keys:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                key = _key(
                    dataset,
                    str(rec.get("source") or dataset),
                    str(rec.get("doc_id") or ""),
                    int(rec.get("chunk_id") or 0),
                )
                if key not in keys:
                    continue
                indexed[key] = {
                    "domain_labels": rec.get("domain_labels") or {},
                    "educational_markers": rec.get("educational_markers") or {},
                    "validity_flags": rec.get("validity_flags") or {},
                    "difficulty": rec.get("difficulty") or {},
                }
    return indexed


def _build_required_gate_paths(config: Dict[str, Any]) -> List[str]:
    paths = []
    for metric, gates in (config.get("gates") or {}).items():
        if not isinstance(gates, dict):
            continue
        for gate_name in gates.keys():
            paths.append(f"{metric}.{gate_name}")
    return sorted(paths)


def _score_domain(rows: List[Dict[str, str]], idx: Dict, cfg: Dict[str, Any], boot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    outcomes_top1: List[int] = []
    outcomes_top3: List[int] = []
    missing = 0
    datasets = set()

    for row in rows:
        if str(row.get("split") or "").strip() not in SPLITS:
            continue
        dataset = str(row["dataset"]).strip()
        source = str(row["source"]).strip()
        doc_id = str(row["doc_id"]).strip()
        chunk_id = int(row["chunk_id"])
        key = _key(dataset, source, doc_id, chunk_id)
        rec = idx.get(key)
        if rec is None:
            missing += 1
            continue

        labels = rec.get("domain_labels") or {}
        if not isinstance(labels, dict) or not labels:
            missing += 1
            continue

        ranked = sorted(
            [(k, _safe_float(v)) for k, v in labels.items() if _safe_float(v) is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            missing += 1
            continue

        top1 = ranked[0][0]
        top3 = {x[0] for x in ranked[:3]}

        gold_primary = str(row.get("gold_primary") or "").strip()
        if not gold_primary:
            continue
        gold_alts = {
            x.strip() for x in str(row.get("gold_alternates_pipe") or "").split("|") if x.strip()
        }

        hit1 = 1 if top1 == gold_primary else 0
        hit3 = 1 if (gold_primary in top3 or bool(gold_alts & top3)) else 0

        outcomes_top1.append(hit1)
        outcomes_top3.append(hit3)
        datasets.add(dataset)

    top1 = (sum(outcomes_top1) / len(outcomes_top1)) if outcomes_top1 else None
    top3 = (sum(outcomes_top3) / len(outcomes_top3)) if outcomes_top3 else None

    gate_cfg = cfg.get("domain") or {}
    top1_thr = float((gate_cfg.get("top1_accuracy") or {}).get("threshold", 0.60))
    top3_thr = float((gate_cfg.get("top3_recall") or {}).get("threshold", 0.85))

    iters = int(boot_cfg.get("iterations", 1000))
    seed = int(boot_cfg.get("seed", 42))
    ci = float(boot_cfg.get("ci", 0.95))

    return {
        "datasets_in_split": sorted(datasets),
        "n_scored": len(outcomes_top1),
        "missing_records": missing,
        "top1_accuracy": {
            "value": round(top1, 6) if top1 is not None else None,
            "threshold": top1_thr,
            "pass": (top1 is not None and top1 >= top1_thr),
            "ci95": _bootstrap_ci(outcomes_top1, lambda s: sum(s) / len(s) if s else None, iters, ci, seed),
        },
        "top3_recall": {
            "value": round(top3, 6) if top3 is not None else None,
            "threshold": top3_thr,
            "pass": (top3 is not None and top3 >= top3_thr),
            "ci95": _bootstrap_ci(outcomes_top3, lambda s: sum(s) / len(s) if s else None, iters, ci, seed + 1),
        },
    }


def _marker_metrics(rows: List[Tuple[bool, bool]]) -> Dict[str, float]:
    tp = sum(1 for g, p in rows if g and p)
    fp = sum(1 for g, p in rows if (not g) and p)
    fn = sum(1 for g, p in rows if g and (not p))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _score_quality(rows: List[Dict[str, str]], idx: Dict, cfg: Dict[str, Any], boot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    markers = [
        ("gold_has_examples", "has_examples"),
        ("gold_has_explanation", "has_explanation"),
        ("gold_has_structure", "has_structure"),
    ]
    paired_rows = []
    missing = 0
    datasets = set()

    for row in rows:
        if str(row.get("split") or "").strip() not in SPLITS:
            continue
        dataset = str(row["dataset"]).strip()
        key = _key(dataset, str(row["source"]).strip(), str(row["doc_id"]).strip(), int(row["chunk_id"]))
        rec = idx.get(key)
        if rec is None:
            missing += 1
            continue
        mk = rec.get("educational_markers") or {}

        item = {}
        valid = True
        for gold_col, pred_col in markers:
            gold = _to_bool(row.get(gold_col, ""))
            pred = mk.get(pred_col)
            if gold is None or not isinstance(pred, bool):
                valid = False
                break
            item[pred_col] = (gold, pred)
        if not valid:
            continue
        paired_rows.append(item)
        datasets.add(dataset)

    def macro_precision(rows_sample: List[Dict[str, Tuple[bool, bool]]]) -> float | None:
        if not rows_sample:
            return None
        precisions = []
        for _, pred_col in markers:
            pts = [x[pred_col] for x in rows_sample]
            m = _marker_metrics(pts)
            precisions.append(m["precision"])
        return sum(precisions) / len(precisions)

    metric_rows = {}
    for _, pred_col in markers:
        pts = [x[pred_col] for x in paired_rows]
        mm = _marker_metrics(pts)
        metric_rows[pred_col] = {k: round(v, 6) for k, v in mm.items()}

    macro_p = macro_precision(paired_rows)
    gate_cfg = cfg.get("quality") or {}
    thr = float((gate_cfg.get("macro_precision") or {}).get("threshold", 0.80))

    iters = int(boot_cfg.get("iterations", 1000))
    seed = int(boot_cfg.get("seed", 42))
    ci = float(boot_cfg.get("ci", 0.95))

    return {
        "datasets_in_split": sorted(datasets),
        "n_scored": len(paired_rows),
        "missing_records": missing,
        "markers": metric_rows,
        "macro_precision": {
            "value": round(macro_p, 6) if macro_p is not None else None,
            "threshold": thr,
            "pass": (macro_p is not None and macro_p >= thr),
            "ci95": _bootstrap_ci(paired_rows, macro_precision, iters, ci, seed + 10),
        },
    }


def _score_difficulty(rows: List[Dict[str, str]], idx: Dict, cfg: Dict[str, Any], boot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    invalid_outcomes: List[int] = []
    sanity_pairs: List[Tuple[bool, bool]] = []
    missing = 0
    datasets = set()

    for row in rows:
        if str(row.get("split") or "").strip() not in SPLITS:
            continue
        dataset = str(row["dataset"]).strip()
        key = _key(dataset, str(row["source"]).strip(), str(row["doc_id"]).strip(), int(row["chunk_id"]))
        rec = idx.get(key)
        if rec is None:
            missing += 1
            continue
        vf = rec.get("validity_flags") or {}
        pred_valid = vf.get("difficulty_valid")
        if not isinstance(pred_valid, bool):
            missing += 1
            continue

        invalid_outcomes.append(0 if pred_valid else 1)
        gold_valid = _to_bool(row.get("gold_valid", ""))
        if gold_valid is not None:
            sanity_pairs.append((gold_valid, pred_valid))
        datasets.add(dataset)

    oor = (sum(invalid_outcomes) / len(invalid_outcomes)) if invalid_outcomes else None
    sanity = (sum(1 for g, p in sanity_pairs if g == p) / len(sanity_pairs)) if sanity_pairs else None

    gate_cfg = cfg.get("difficulty") or {}
    thr = float((gate_cfg.get("out_of_range_rate") or {}).get("threshold", 0.01))

    iters = int(boot_cfg.get("iterations", 1000))
    seed = int(boot_cfg.get("seed", 42))
    ci = float(boot_cfg.get("ci", 0.95))

    return {
        "datasets_in_split": sorted(datasets),
        "n_scored": len(invalid_outcomes),
        "missing_records": missing,
        "sanity_agreement": {
            "value": round(sanity, 6) if sanity is not None else None,
            "n": len(sanity_pairs),
        },
        "out_of_range_rate": {
            "value": round(oor, 6) if oor is not None else None,
            "threshold": thr,
            "pass": (oor is not None and oor <= thr),
            "ci95": _bootstrap_ci(invalid_outcomes, lambda s: sum(s) / len(s) if s else None, iters, ci, seed + 20),
        },
    }


def _score_auto_from_manifest(manifest: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    gates = manifest.get("reliability_gate_outcomes", {}) or {}
    red_state = ((gates.get("redundancy") or {}).get("non_degenerate_distribution") or {})
    ppl_state = ((gates.get("perplexity") or {}).get("non_null_coverage") or {})

    red_thr = (cfg.get("redundancy") or {}).get("non_degenerate_distribution", {}).get("threshold", True)
    ppl_thr = float((cfg.get("perplexity") or {}).get("non_null_coverage", {}).get("threshold", 0.9))

    ppl_val = _safe_float(ppl_state.get("value"))
    red_pass = bool(red_state.get("pass")) if "pass" in red_state else None
    ppl_pass = (ppl_val is not None and ppl_val >= ppl_thr)

    return {
        "redundancy": {
            "non_degenerate_distribution": {
                "value": red_state.get("value"),
                "threshold": red_thr,
                "pass": red_pass,
                "ci95": [None, None],
                "scope": "global_run",
            }
        },
        "perplexity": {
            "non_null_coverage": {
                "value": round(ppl_val, 6) if ppl_val is not None else None,
                "threshold": ppl_thr,
                "pass": ppl_pass if ppl_val is not None else None,
                "ci95": [None, None],
                "scope": "global_run",
            }
        },
    }


def _rows_for_split(rows: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    return [r for r in rows if str(r.get("split") or "").strip() == split]


def _build_split_score(
    split: str,
    domain_rows: List[Dict[str, str]],
    quality_rows: List[Dict[str, str]],
    diff_rows: List[Dict[str, str]],
    idx: Dict,
    config: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    gate_cfg = config.get("gates") or {}
    boot_cfg = config.get("bootstrap") or {}

    d_rows = _rows_for_split(domain_rows, split)
    q_rows = _rows_for_split(quality_rows, split)
    f_rows = _rows_for_split(diff_rows, split)

    domain_score = _score_domain(d_rows, idx, gate_cfg, boot_cfg)
    quality_score = _score_quality(q_rows, idx, gate_cfg, boot_cfg)
    diff_score = _score_difficulty(f_rows, idx, gate_cfg, boot_cfg)
    auto_scores = _score_auto_from_manifest(manifest, gate_cfg)

    datasets = sorted(
        set(domain_score.get("datasets_in_split", []))
        | set(quality_score.get("datasets_in_split", []))
        | set(diff_score.get("datasets_in_split", []))
    )

    return {
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "identity_version": config.get("identity_version", "v1"),
        "required_gate_paths": _build_required_gate_paths(config),
        "datasets_in_split": datasets,
        "counts": {
            "domain_label_rows": len(d_rows),
            "quality_label_rows": len(q_rows),
            "difficulty_label_rows": len(f_rows),
        },
        "gate_scores": {
            "domain": {
                "top1_accuracy": domain_score["top1_accuracy"],
                "top3_recall": domain_score["top3_recall"],
            },
            "quality": {
                "macro_precision": quality_score["macro_precision"],
                "markers": quality_score.get("markers", {}),
            },
            "difficulty": {
                "out_of_range_rate": diff_score["out_of_range_rate"],
                "sanity_agreement": diff_score.get("sanity_agreement", {}),
            },
            "redundancy": auto_scores["redundancy"],
            "perplexity": auto_scores["perplexity"],
        },
        "missing_records": {
            "domain": domain_score.get("missing_records", 0),
            "quality": quality_score.get("missing_records", 0),
            "difficulty": diff_score.get("missing_records", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score metric gates for main_eval + transfer_eval.")
    parser.add_argument("--identity-config", type=Path, default=IDENTITY_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--domain-labels", type=Path, default=DOMAIN_LABELS)
    parser.add_argument("--quality-labels", type=Path, default=QUALITY_LABELS)
    parser.add_argument("--difficulty-labels", type=Path, default=DIFF_LABELS)
    parser.add_argument("--out-main", type=Path, default=MAIN_SCORE_PATH)
    parser.add_argument("--out-transfer", type=Path, default=TRANSFER_SCORE_PATH)
    args = parser.parse_args()

    config = _load_json(args.identity_config)
    manifest = _load_json(args.manifest)

    domain_rows = _load_csv(
        args.domain_labels,
        ["dataset", "source", "doc_id", "chunk_id", "gold_primary", "gold_alternates_pipe", "split"],
    )
    quality_rows = _load_csv(
        args.quality_labels,
        [
            "dataset",
            "source",
            "doc_id",
            "chunk_id",
            "gold_has_examples",
            "gold_has_explanation",
            "gold_has_structure",
            "split",
        ],
    )
    diff_rows = _load_csv(
        args.difficulty_labels,
        ["dataset", "source", "doc_id", "chunk_id", "gold_valid", "reason", "split"],
    )

    keys_by_dataset = _required_keys_by_dataset(domain_rows, quality_rows, diff_rows)
    outputs = _dataset_outputs(manifest)
    idx = _load_required_records(keys_by_dataset, outputs)

    main_score = _build_split_score(
        split="main_eval",
        domain_rows=domain_rows,
        quality_rows=quality_rows,
        diff_rows=diff_rows,
        idx=idx,
        config=config,
        manifest=manifest,
    )
    transfer_score = _build_split_score(
        split="transfer_eval",
        domain_rows=domain_rows,
        quality_rows=quality_rows,
        diff_rows=diff_rows,
        idx=idx,
        config=config,
        manifest=manifest,
    )

    _save_json(args.out_main, main_score)
    _save_json(args.out_transfer, transfer_score)

    print(f"[10] main gates: {args.out_main}")
    print(f"[10] transfer gates: {args.out_transfer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
