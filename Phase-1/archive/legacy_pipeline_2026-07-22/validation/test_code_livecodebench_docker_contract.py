from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_DIR / "validation" / "docker" / "livecodebench" / "Dockerfile"


def test_docker_image_pins_livecodebench_import_dependencies() -> None:
    # Given: the Docker recipe used for the pinned LiveCodeBench evaluator.
    recipe = DOCKERFILE.read_text(encoding="utf-8")

    # When: its installed Python packages are inspected.
    installed_packages = recipe.split("RUN pip install --no-cache-dir", maxsplit=1)[1]

    # Then: the official module's eager datasets import is reproducible offline.
    assert "datasets==3.6.0" in installed_packages


if __name__ == "__main__":
    test_docker_image_pins_livecodebench_import_dependencies()
    print("[code-livecodebench-docker] dependency contract: pass")
