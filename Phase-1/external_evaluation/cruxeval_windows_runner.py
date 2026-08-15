"""Run CRUXEval on Windows with its process-level timeout intact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from external_evaluation.evalplus_windows_runner import (
    install_windows_signal_compatibility,
)


DEFAULT_CRUX_ROOT = Path("D:/UNLV-Research/third_party/cruxeval/evaluation")


def install_cruxeval_windows_compatibility(
    crux_root: Path = DEFAULT_CRUX_ROOT,
) -> None:
    install_windows_signal_compatibility()
    root = str(crux_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


install_cruxeval_windows_compatibility()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations-path", type=Path, required=True)
    parser.add_argument("--scored-results-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("input", "output"), required=True)
    parser.add_argument("--crux-root", type=Path, default=DEFAULT_CRUX_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    install_cruxeval_windows_compatibility(args.crux_root)
    from evaluate_generations import evaluate_generations

    generations = json.loads(args.generations_path.read_text(encoding="utf-8"))
    previous_directory = Path.cwd()
    os.chdir(args.crux_root)
    try:
        results = evaluate_generations(generations, args.mode)
    finally:
        os.chdir(previous_directory)

    args.scored_results_path.parent.mkdir(parents=True, exist_ok=True)
    args.scored_results_path.write_text(
        json.dumps(results, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"pass@1: {results['pass_at_1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
