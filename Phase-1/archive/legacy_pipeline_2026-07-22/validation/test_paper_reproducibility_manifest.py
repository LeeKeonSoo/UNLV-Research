#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "198_build_paper_reproducibility_manifest.py"
    spec = importlib.util.spec_from_file_location("paper_reproducibility_manifest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = module.build(
            tmp_path / "paper_reproducibility_manifest.json",
            tmp_path / "paper_reproducibility_manifest.md",
        )
    assert manifest["status"] == "paper_reproducibility_manifest_frozen"
    assert manifest["summary"]["remaining_required_manifest_items"] == []
    assert manifest["environment"]["conda_environment"] == "research"
    assert manifest["environment"]["default_cuda_visible_devices"] == "1"
    assert manifest["environment"]["primary_gpu"] == "NVIDIA GeForce RTX 3070 Ti"
    commands = [item["command"] for item in manifest["commands"]]
    assert "conda run -n research python 196_build_curation_stage_paper_package.py" in commands
    assert "conda run -n research python 197_build_paper_comparison_tables.py" in commands
    assert manifest["source_scripts"]["paper_claim_release_gate"]["exists"] is True
    assert manifest["artifacts"]["curation_stage_paper_package"]["exists"] is True
    assert manifest["configs"]["lm_curation_operational_framework"]["exists"] is True
    assert manifest["missing_inputs"] == []
    print("[paper-reproducibility-manifest] manifest frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
