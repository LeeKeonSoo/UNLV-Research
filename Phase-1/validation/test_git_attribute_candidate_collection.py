#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_git_attribute_candidate_pool import collect_rows


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True)


def test_collect_rows_preserves_only_explicit_git_attribute_declarations() -> None:
    with TemporaryDirectory() as temporary:
        repository = Path(temporary)
        _git(repository, "init")
        (repository / ".gitattributes").write_text("generated.py linguist-generated=true\nauthored.py linguist-generated=false\n", encoding="utf-8")
        (repository / "generated.py").write_text("GENERATED = True\n", encoding="utf-8")
        (repository / "authored.py").write_text("def authored():\n    return 1\n", encoding="utf-8")
        (repository / "unknown.py").write_text("def unknown():\n    return 2\n", encoding="utf-8")
        _git(repository, "add", ".")

        rows = collect_rows(repository, repository_identity="fixture/repository", source_uri="https://example.invalid/fixture", collected_at="2026-07-27T00:00:00Z")

    by_path = {row["partition"]["path"]: row for row in rows}
    assert by_path["generated.py"]["artifact_context"] == {"generation": "generated"}
    assert by_path["authored.py"]["artifact_context"] == {"generation": "authored"}
    assert by_path["unknown.py"]["artifact_context"] == {"generation": "unknown"}
    assert by_path["unknown.py"]["artifact_context"]["generation"] != "authored"


if __name__ == "__main__":
    test_collect_rows_preserves_only_explicit_git_attribute_declarations()
    print("[git-attribute-collection] explicit declaration preservation: pass")
