from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("03_score_core_metrics.py")
    manifest = module.build_scoring_reproducibility_manifest(ROOT / "outputs" / "index" / "index.sqlite")
    assert manifest["complete"] is True
    for path in (
        "03_score_core_metrics.py",
        "signals/core.py",
        "quality/reference_quality.py",
        "data_eval_common.py",
    ):
        row = manifest["source_files"][str(Path(path))]
        assert row["exists"] is True
        assert row["sha256"]
    assert manifest["model_artifacts"]["reference_quality_model"]["sha256"]
    assert manifest["model_artifacts"]["reference_quality_metadata"]["sha256"]
    assert manifest["index_input"]["sha256"]
    print("[scoring-reproducibility-manifest] scorer/model/index hashes present: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
