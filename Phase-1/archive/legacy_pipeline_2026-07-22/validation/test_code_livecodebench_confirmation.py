from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence import livecodebench_confirmation


def test_remaining_seed_schedule_uses_both_approved_gpus() -> None:
    # Given: the completed pilot seed and four unused frozen training seeds.
    training_seeds = (101, 131, 163, 197, 239)
    pilot_seed = 101

    # When: confirmation work is scheduled before those outcomes are observed.
    schedule = livecodebench_confirmation.remaining_seed_schedule(
        training_seeds=training_seeds,
        completed_seeds=(pilot_seed,),
    )

    # Then: every remaining seed is assigned once across both local GPUs.
    assert tuple(item.seed for item in schedule) == (131, 163, 197, 239)
    assert {item.required_gpu for item in schedule} == {
        "NVIDIA GeForce RTX 4060 Ti",
        "NVIDIA GeForce RTX 3070 Ti",
    }


def test_build_confirmation_preserves_the_pilot_task_bundle(tmp_path: Path) -> None:
    # Given: a completed seed-101 pilot and its original frozen task bundle.
    bundle = tmp_path / "frozen_tasks.json"
    bundle.write_text("[]", encoding="utf-8")
    source_freeze = tmp_path / "pilot.json"
    source_freeze.write_text(
        json.dumps(
            {
                "status": "frozen_before_outcomes",
                "dataset": {"frozen_task_bundle": {"path": str(bundle), "sha256": "task-hash"}},
                "execution": {"seed": 101, "required_gpu": "NVIDIA GeForce RTX 3070 Ti"},
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"confirmatory_training_recipe": {"confirmatory_training_seeds": [101, 131, 163]}}
        ),
        encoding="utf-8",
    )

    # When: the remaining confirmation contracts are frozen.
    report = livecodebench_confirmation.build_confirmation(
        source_freeze=source_freeze,
        training_plan=plan,
        output_dir=tmp_path / "confirmation",
    )

    # Then: only unused seeds are emitted and each references the original bundle.
    assert report["scheduled_seeds"] == [131, 163]
    frozen = json.loads((tmp_path / "confirmation" / "seed131.json").read_text(encoding="utf-8"))
    assert frozen["dataset"]["frozen_task_bundle"]["path"] == str(bundle)
    assert frozen["confirmation"]["selector_tuning_permission"] is False
    assert frozen["confirmation"]["training_output_root"].endswith("current_framework_rerun")


if __name__ == "__main__":
    test_remaining_seed_schedule_uses_both_approved_gpus()
    with tempfile.TemporaryDirectory() as temporary_dir:
        test_build_confirmation_preserves_the_pilot_task_bundle(Path(temporary_dir))
    print("[code-livecodebench-confirmation] seed schedule: pass")
