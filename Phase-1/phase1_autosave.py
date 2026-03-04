#!/usr/bin/env python3
"""Phase-1 autosave utilities.

This module syncs key artifacts to an external checkpoint location
(typically Google Drive in Colab) while keeping local outputs intact.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


COMMON_ENTRIES: List[str] = [
    "outputs/run_manifest.json",
    "outputs/run_summary.json",
    "outputs/validation",
]

STAGE_ENTRIES: Dict[str, List[str]] = {
    "collect_khan_academy": [
        "khan_k12_concepts",
    ],
    "collect_tiny_textbooks": [
        "tiny_textbooks_raw",
    ],
    "extract_khan_taxonomy": [
        "outputs/concept_prototypes_tfidf.pkl",
        "outputs/khan_taxonomy.json",
    ],
    "build_corpus_index": [
        "outputs/corpus_index.pkl",
        "outputs/corpus_texts.sqlite",
        "outputs/metadata.json",
        "outputs/pipeline_state.json",
    ],
    "compute_metrics": [
        "outputs/*analysis.jsonl",
        "outputs/run_manifest.json",
        "outputs/run_summary.json",
        "outputs/validation",
    ],
    "build_dashboard": [
        "outputs/dashboard.html",
    ],
    "generate_label_templates": [
        "validation/labels",
    ],
    "calibrate_ood_thresholds": [
        "configs/metric_identity_v1.json",
        "outputs/validation/ood_calibration_report.json",
    ],
    "score_metric_gates": [
        "outputs/validation/gate_scores_main.json",
        "outputs/validation/gate_scores_transfer.json",
    ],
    "certify_phase1": [
        "outputs/run_manifest.json",
        "outputs/validation/full_validation_report.json",
    ],
    "validate_phase1_outputs": [
        "outputs/validation",
    ],
    "final": [
        "outputs",
        "validation/labels",
    ],
}


def resolve_autosave_target(
    project_dir: Path,
    autosave_root: Optional[str] = None,
) -> Optional[Path]:
    """Resolve autosave destination path.

    Priority:
    1) explicit autosave_root argument
    2) PHASE1_AUTOSAVE_ROOT env
    3) common Colab Drive destination
    4) local-only mode (return project_dir)
    """
    explicit = (autosave_root or "").strip()
    if explicit.lower() in {"off", "false", "0", "none", "disable"}:
        return None

    env_val = os.getenv("PHASE1_AUTOSAVE_ROOT", "").strip()
    if env_val.lower() in {"off", "false", "0", "none", "disable"}:
        return None

    raw = explicit or env_val
    if raw:
        target = Path(raw).expanduser()
    else:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            target = drive_root / "UNLV-Research" / "Phase-1"
        else:
            # Local-only mode: outputs are already saved in project_dir.
            return project_dir.resolve()

    target = target.resolve()

    # Allow users to pass repository root instead of Phase-1 path.
    phase1_child = target / "Phase-1"
    if phase1_child.is_dir() and (phase1_child / "00_run_phase1.py").exists():
        target = phase1_child

    return target


def _copy_file_if_changed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        s = src.stat()
        d = dst.stat()
        if s.st_size == d.st_size and int(s.st_mtime) <= int(d.st_mtime):
            return False
    shutil.copy2(src, dst)
    return True


def _sync_dir(src_dir: Path, dst_dir: Path) -> Tuple[int, int]:
    copied = 0
    scanned = 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        scanned += 1
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if _copy_file_if_changed(src, dst):
            copied += 1
    return copied, scanned


def _expand_entries(project_dir: Path, entries: Iterable[str]) -> List[Path]:
    expanded: List[Path] = []
    for entry in entries:
        if "*" in entry or "?" in entry or "[" in entry:
            expanded.extend(sorted(project_dir.glob(entry)))
        else:
            expanded.append(project_dir / entry)
    return expanded


def _entries_for_stage(stage: str) -> List[str]:
    stage_specific = STAGE_ENTRIES.get(stage, [])
    out: List[str] = []
    seen = set()
    for entry in [*COMMON_ENTRIES, *stage_specific]:
        if entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
    return out


def autosave_stage(
    project_dir: Path,
    stage: str,
    autosave_root: Optional[str] = None,
    resolved_target: Optional[Path] = None,
) -> Optional[Path]:
    """Autosave stage artifacts. Returns destination path or None if disabled."""
    project_dir = project_dir.resolve()
    target = (resolved_target or resolve_autosave_target(project_dir, autosave_root))
    if target is None:
        print(f"[Autosave] disabled (stage={stage})")
        return None

    target = target.resolve()
    if target == project_dir:
        print(f"[Autosave] local mode (already persisted in {project_dir})")
        return target

    target.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    scanned_files = 0
    missing = 0

    entries = _entries_for_stage(stage)
    for src in _expand_entries(project_dir, entries):
        if not src.exists():
            missing += 1
            continue

        rel = src.relative_to(project_dir)
        dst = target / rel
        if src.is_file():
            scanned_files += 1
            if _copy_file_if_changed(src, dst):
                copied_files += 1
            continue

        c, s = _sync_dir(src, dst)
        copied_files += c
        scanned_files += s

    print(
        f"[Autosave] stage={stage} target={target} "
        f"copied={copied_files} scanned={scanned_files} missing_entries={missing}"
    )
    return target

