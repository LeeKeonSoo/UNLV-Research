#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "deployment_surface_v1.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _relative_files(pattern: str) -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob(pattern)
        if path.is_file() and "archive" not in path.relative_to(ROOT).parts
    }


def test_active_python_surface_matches_deployment_manifest() -> None:
    manifest = _manifest()
    expected = set(manifest["runtime_modules"])
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if path.is_file()
        and not {
            "archive",
            "validation",
            "tmp",
            "output",
            "outputs",
            ".omo",
        }.intersection(path.relative_to(ROOT).parts)
    }
    assert actual == expected


def test_active_config_surface_matches_deployment_manifest() -> None:
    manifest = _manifest()
    assert _relative_files("configs/*.json") == set(manifest["runtime_configs"])


def test_active_validation_surface_matches_deployment_manifest() -> None:
    manifest = _manifest()
    assert _relative_files("validation/*.py") == set(manifest["validation_files"])


if __name__ == "__main__":
    test_active_python_surface_matches_deployment_manifest()
    test_active_config_surface_matches_deployment_manifest()
    test_active_validation_surface_matches_deployment_manifest()
    print("[active-surface] deployment runtime only: pass")
