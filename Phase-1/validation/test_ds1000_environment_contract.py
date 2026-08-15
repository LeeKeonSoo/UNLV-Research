from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARATION_SCRIPT = ROOT / "tmp" / "prepare_ds1000_environment.ps1"


def test_ds1000_environment_declares_all_evaluator_imports() -> None:
    # Given: the reproducible environment preparation script.
    source = PREPARATION_SCRIPT.read_text(encoding="utf-8")

    # When: its installed packages and readiness import probe are inspected.
    install_section, import_probe = source.split('& $Python -c "', maxsplit=1)

    # Then: tqdm is both installed and exercised before readiness is recorded.
    assert "tqdm" in install_section
    assert "tqdm" in import_probe.split('"', maxsplit=1)[0]


if __name__ == "__main__":
    test_ds1000_environment_declares_all_evaluator_imports()
    print("DS-1000 environment contract passed")
