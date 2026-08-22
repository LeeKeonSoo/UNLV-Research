#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MANIFEST = ROOT / "configs" / "deployment_surface_v1.json"


def _module_name(path: Path) -> str:
    return path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")


def _import_closure(entrypoint: str) -> set[str]:
    files = {
        _module_name(path): path
        for path in ROOT.rglob("*.py")
        if "archive" not in path.parts
        and "validation" not in path.parts
        and "external_evaluation" not in path.parts
        and "scripts" not in path.parts
        and "tmp" not in path.parts
    }
    queue = [Path(entrypoint).with_suffix("").as_posix().replace("/", ".")]
    seen: set[str] = set()
    while queue:
        module = queue.pop(0)
        if module in seen or module not in files:
            continue
        seen.add(module)
        tree = ast.parse(files[module].read_text(encoding="utf-8-sig"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        for imported in imports:
            for candidate in (imported, imported.split(".")[0]):
                if candidate in files and candidate not in seen and candidate not in queue:
                    queue.append(candidate)
    return {files[module].relative_to(ROOT).as_posix() for module in seen}


def test_manifest_contains_exactly_the_runtime_import_closure() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = set(manifest["runtime_modules"])
    closure = _import_closure(str(manifest["entrypoint"]))
    closure.add("ingestion/__init__.py")

    assert closure == declared, {
        "undeclared_runtime_imports": sorted(closure - declared),
        "declared_but_unreachable": sorted(declared - closure),
    }
    assert all((ROOT / path).is_file() for path in declared)
    assert not any(
        forbidden in path
        for forbidden in manifest["forbidden_runtime_families"]
        for path in declared
    )


def test_public_profiles_and_registry_match_the_deployment_policy_set() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    profiles = json.loads(
        (ROOT / "configs" / "runtime_policy_profiles_v1.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "configs" / "runtime_policy_registry_v1.json").read_text(encoding="utf-8")
    )
    expected = set(manifest["active_policy_ids"])
    public = {
        str(profile["user_facing_mode"]): set(profile["enabled_policy_ids"])
        for profile in profiles["profiles"]
        if profile.get("user_facing_mode") in manifest["public_modes"]
    }
    policies = {policy["id"]: policy for policy in registry["policies"]}
    runtime_modules = set(manifest["runtime_modules"])

    assert set(public) == set(manifest["public_modes"])
    assert all(policy_ids == expected for policy_ids in public.values())
    assert expected <= set(policies)
    assert all(
        set(policies[policy_id]["runtime_implementation"]) <= runtime_modules
        for policy_id in expected
    )


def test_every_declared_runtime_config_exists() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert all((ROOT / path).is_file() for path in manifest["runtime_configs"])
    assert all((ROOT / path).is_file() for path in manifest["support_files"])


if __name__ == "__main__":
    test_manifest_contains_exactly_the_runtime_import_closure()
    test_public_profiles_and_registry_match_the_deployment_policy_set()
    test_every_declared_runtime_config_exists()
    print("[deployment-surface-v1] exact runtime closure and policy linkage: pass")
