#!/usr/bin/env python3
"""Run the generic data evaluation pipeline from prepared inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from data_eval_common import (
    DEFAULT_TOKENIZER_NAME,
    QUALITY_REFERENCE_META_PATH,
    QUALITY_REFERENCE_MODEL_PATH,
    dataset_token_budget,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
DEFAULT_OPENWEBTEXT2_LIMIT = 500000
DEFAULT_OPENWEBTEXT2_TARGET_GB = 2.0
DEFAULT_OPENWEBTEXT2_MATCH_TOKENS_TO_INDEX = 0
CORE_STEPS = (
    ("01_validate_inputs.py", "datasets"),
    ("02_build_index.py", "datasets"),
    ("03_score_core_metrics.py", None),
    ("04_generate_subsets.py", "profiles"),
    ("05_build_dashboard.py", None),
    ("06_validate_outputs.py", "report"),
)
EXTENDED_STEPS = (
    ("07_run_property_benchmarks.py", None),
    ("08_build_metric_maturity_snapshot.py", None),
)
ORDERED_STEPS = CORE_STEPS + EXTENDED_STEPS


def _run_command(cmd: list[str], *, log_file: Path) -> None:
    command_text = " ".join(cmd)
    print(f"\n[data-eval] run: {command_text}", flush=True)
    started = time.time()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[run] {command_text}\n")
        log.flush()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        elapsed = time.time() - started
        status = "done" if returncode == 0 else "failed"
        message = f"[{status}] returncode={returncode} elapsed_sec={elapsed:.1f}\n"
        print(message, end="", flush=True)
        log.write(message)
        log.flush()
        if returncode != 0:
            raise SystemExit(returncode)


def _load_dataset_specs(config_path: Path):
    with config_path.open("r", encoding="utf-8", errors="replace") as f:
        payload = json.load(f)
    return payload.get("datasets", []) if isinstance(payload, dict) else payload


def _profiles_require_dual_eval(profiles_path: Path) -> bool:
    with profiles_path.open("r", encoding="utf-8", errors="replace") as f:
        payload = json.load(f)
    profiles = (payload.get("profiles") or {}) if isinstance(payload, dict) else {}
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        stage_c = profile.get("stage_c_validation") or {}
        utility_probe = profile.get("utility_probe") or {}
        evaluation_mode = str(stage_c.get("evaluation_mode") or "").strip().lower()
        if evaluation_mode == "certification":
            return True
        if bool(stage_c.get("enforce_ood_utility_pass", False)):
            return True
        if bool(utility_probe.get("dual_eval_required", False)):
            return True
    return False


def _resolve_source_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    return path


def _normalized_runtime_spec(raw: dict) -> dict:
    return {
        "name": str(raw.get("name") or "").strip(),
        "format": str(raw.get("format") or "").strip(),
        "source": _resolve_source_path(str(raw.get("source") or "")),
        "batch_glob": str(raw.get("batch_glob") or "batch_*.json"),
        "text_field": str(raw.get("text_field") or "text"),
        "id_fields": [str(x) for x in raw.get("id_fields", [])],
        "metadata_fields": [str(x) for x in raw.get("metadata_fields", [])],
        "min_text_chars": int(raw.get("min_text_chars", 50)),
    }


def _resolve_openwebtext2_target_tokens(
    all_specs,
    explicit_target_tokens: int | None,
    match_tokens_to_index: int | None,
    tokenizer_name: str,
) -> int | None:
    if explicit_target_tokens is not None:
        return explicit_target_tokens
    if match_tokens_to_index is None or match_tokens_to_index < 0:
        return None
    if match_tokens_to_index >= len(all_specs):
        raise SystemExit(f"openwebtext2-match-tokens-to-index out of range: {match_tokens_to_index}")
    budget = dataset_token_budget(_normalized_runtime_spec(all_specs[match_tokens_to_index]), tokenizer_name=tokenizer_name)
    return int(budget["token_count"])


def _should_refresh_openwebtext2_subset(
    source: Path,
    limit: int,
    target_gb: float,
    target_tokens: int | None,
    tokenizer_name: str,
) -> tuple[bool, str]:
    if not source.exists():
        return True, "source missing"
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        return True, "manifest missing"
    try:
        with manifest_path.open("r", encoding="utf-8", errors="replace") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True, "manifest unreadable"

    requested_target_bytes = int(target_gb * (1024 ** 3))
    approx_bytes = int(manifest.get("approx_bytes_written") or 0)
    approx_tokens = int(manifest.get("approx_tokens_written") or 0)
    recorded_target_gb = float(manifest.get("target_gb") or 0.0)
    recorded_target_tokens = manifest.get("target_tokens")
    recorded_limit = int(manifest.get("requested_limit") or manifest.get("records_written") or 0)
    recorded_tokenizer_name = str(manifest.get("tokenizer_name") or "")

    if target_tokens is not None:
        if recorded_target_tokens is None:
            return True, "token target added"
        if int(recorded_target_tokens) != target_tokens:
            return True, "token target changed"
        if recorded_tokenizer_name != tokenizer_name:
            return True, "tokenizer changed"
        if approx_tokens < int(target_tokens * 0.95) and recorded_limit < limit:
            return True, "existing subset capped below requested token target"
        return False, "up to date"

    if abs(recorded_target_gb - target_gb) > 1e-9:
        return True, "target size changed"
    # Rebuild only when the current subset is still far from the requested size
    # and the previous run was capped by a smaller limit.
    if approx_bytes < int(requested_target_bytes * 0.95) and recorded_limit < limit:
        return True, "existing subset capped by smaller limit"
    return False, "up to date"


def _maybe_prepare_openwebtext2_subset(
    all_specs,
    selected_specs,
    limit: int,
    target_gb: float,
    target_tokens: int | None,
    match_tokens_to_index: int | None,
    tokenizer_name: str,
) -> None:
    resolved_target_tokens = _resolve_openwebtext2_target_tokens(
        all_specs,
        explicit_target_tokens=target_tokens,
        match_tokens_to_index=match_tokens_to_index,
        tokenizer_name=tokenizer_name,
    )
    for raw in selected_specs:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        source = _resolve_source_path(str(raw.get("source") or ""))
        if name != "openwebtext2_subset":
            continue
        refresh, reason = _should_refresh_openwebtext2_subset(
            source,
            limit,
            target_gb,
            resolved_target_tokens,
            tokenizer_name,
        )
        if refresh:
            helper = PROJECT_DIR / "prepare_openwebtext2_subset.py"
            cmd = [
                sys.executable,
                str(helper.name),
                "--limit",
                str(limit),
                "--target-gb",
                str(target_gb),
                "--output",
                str(source),
            ]
            if resolved_target_tokens is not None:
                cmd.extend(["--target-tokens", str(resolved_target_tokens), "--tokenizer-name", tokenizer_name])
            print("\n[data-eval] auto-prepare OpenWebText2 subset")
            print(f"  reason: {reason}")
            if resolved_target_tokens is not None:
                print(f"  target_tokens: {resolved_target_tokens} ({tokenizer_name})")
            print("  $", " ".join(cmd))
            result = subprocess.run(cmd, cwd=PROJECT_DIR)
            if result.returncode != 0:
                raise SystemExit(result.returncode)


def _maybe_prepare_reference_quality_model() -> None:
    expected_version = "reference-quality-classifier-v1"
    if QUALITY_REFERENCE_MODEL_PATH.exists() and QUALITY_REFERENCE_META_PATH.exists():
        try:
            meta = json.loads(QUALITY_REFERENCE_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if str(meta.get("version") or "") == expected_version:
            return
    helper = PROJECT_DIR / "prepare_reference_quality_model.py"
    cmd = [sys.executable, str(helper.name)]
    print("\n[data-eval] auto-prepare reference quality model")
    reason = "reference quality model missing or outdated"
    print(f"  reason: {reason}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _reset_outputs() -> None:
    targets = [
        OUTPUT_DIR / "index",
        OUTPUT_DIR / "scored",
        OUTPUT_DIR / "subsets",
        OUTPUT_DIR / "validation",
        OUTPUT_DIR / "dashboard.html",
        OUTPUT_DIR / "run_manifest.json",
        OUTPUT_DIR / "run_summary.json",
        OUTPUT_DIR / "utility_probe_results.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def _select_datasets_by_index(
    config_path: Path,
    selected_indexes: list[int] | None,
    *,
    require_dual_eval: bool,
):
    specs = _load_dataset_specs(config_path)
    if not selected_indexes:
        if require_dual_eval:
            if len(specs) < 2:
                raise SystemExit(
                    "Dual-eval mode requires at least two datasets in the config, "
                    f"but found {len(specs)} in {config_path}."
                )
            selected_indexes = [0, 1]
            print("[data-eval] dual-eval enabled -> default dataset indexes: 0 1")
        else:
            selected_indexes = [0]
    selected = []
    for idx in selected_indexes:
        if idx < 0 or idx >= len(specs):
            raise SystemExit(f"dataset-index out of range: {idx}")
        selected.append(specs[idx])
    if require_dual_eval and len(selected) < 2:
        raise SystemExit(
            "Dual-eval mode requires at least two selected datasets. "
            "Pass --dataset-index 0 1 or provide a config with >=2 datasets."
        )
    return specs, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the generic data evaluation pipeline.")
    parser.add_argument("--stop-after", choices=[step for step, _ in ORDERED_STEPS], default=None)
    parser.add_argument(
        "--flow",
        choices=("core", "full"),
        default="core",
        help="core: 01-06 only, full: 01-08",
    )
    parser.add_argument("--datasets-config", default="datasets_config.json")
    parser.add_argument("--dataset-index", nargs="*", type=int, default=None)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--profiles", default="configs/curation_profiles.json")
    parser.add_argument("--validation-report", default="outputs/validation/full_validation_report.json")
    parser.add_argument("--openwebtext2-limit", type=int, default=DEFAULT_OPENWEBTEXT2_LIMIT)
    parser.add_argument("--openwebtext2-target-gb", type=float, default=DEFAULT_OPENWEBTEXT2_TARGET_GB)
    parser.add_argument("--openwebtext2-target-tokens", type=int, default=None)
    parser.add_argument("--openwebtext2-match-tokens-to-index", type=int, default=DEFAULT_OPENWEBTEXT2_MATCH_TOKENS_TO_INDEX)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--reuse-existing-outputs", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write a tee-style execution log while still showing step progress in the terminal.",
    )
    args = parser.parse_args()

    dataset_config = Path(args.datasets_config)
    if not dataset_config.is_absolute():
        dataset_config = (PROJECT_DIR / dataset_config).resolve()
    profiles_path = Path(args.profiles)
    if not profiles_path.is_absolute():
        profiles_path = (PROJECT_DIR / profiles_path).resolve()
    require_dual_eval = _profiles_require_dual_eval(profiles_path)
    all_specs, selected_specs = _select_datasets_by_index(
        dataset_config,
        args.dataset_index,
        require_dual_eval=require_dual_eval,
    )
    if args.list_datasets:
        for i, spec in enumerate(all_specs):
            print(f"[{i}] {spec.get('name')} -> {spec.get('source')}")
        return 0

    _maybe_prepare_openwebtext2_subset(
        all_specs,
        selected_specs,
        args.openwebtext2_limit,
        args.openwebtext2_target_gb,
        args.openwebtext2_target_tokens,
        args.openwebtext2_match_tokens_to_index,
        args.tokenizer_name,
    )
    _maybe_prepare_reference_quality_model()
    if not args.reuse_existing_outputs:
        _reset_outputs()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump({"datasets": selected_specs}, tmp, indent=2, ensure_ascii=False)
        selected_config_path = Path(tmp.name)

    try:
        active_steps = CORE_STEPS if args.flow == "core" else ORDERED_STEPS
        active_step_names = [step for step, _ in active_steps]
        if args.stop_after and args.stop_after not in active_step_names:
            raise SystemExit(
                f"--stop-after {args.stop_after} is not in active flow={args.flow} steps: {', '.join(active_step_names)}"
            )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = args.log_file or LOG_DIR / f"data_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
        if not log_file.is_absolute():
            log_file = (PROJECT_DIR / log_file).resolve()
        (LOG_DIR / "latest_data_eval.log").write_text(str(log_file), encoding="utf-8")
        print(f"[data-eval] flow={args.flow} steps={', '.join(active_step_names)}")
        print(f"[data-eval] log={log_file}")
        for step, mode in active_steps:
            cmd = [sys.executable, step]
            if mode == "datasets":
                cmd.extend(["--datasets-config", str(selected_config_path)])
            elif mode == "profiles":
                cmd.extend(["--profiles", args.profiles])
            elif mode == "report":
                cmd.extend(["--write-report", args.validation_report])
            _run_command(cmd, log_file=log_file)
            if args.stop_after == step:
                break
    finally:
        if selected_config_path.exists():
            selected_config_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
