#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_math_structural_heads import verify_model_hashes


def test_frozen_model_hashes_are_required_before_scoring() -> None:
    with TemporaryDirectory() as directory:
        model = Path(directory) / "model.joblib"
        model.write_bytes(b"frozen model")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()

        assert verify_model_hashes({"head": model}, {"head": digest}) == {"head": digest}


def test_model_hash_mismatch_is_rejected() -> None:
    with TemporaryDirectory() as directory:
        model = Path(directory) / "model.joblib"
        model.write_bytes(b"different model")

        try:
            verify_model_hashes({"head": model}, {"head": "0" * 64})
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("Expected a frozen model hash mismatch")


if __name__ == "__main__":
    test_frozen_model_hashes_are_required_before_scoring()
    test_model_hash_mismatch_is_rejected()
    print("[math-structural-scorer] frozen artifact gate: pass")
