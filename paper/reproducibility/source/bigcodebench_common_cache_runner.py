import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--groundtruth-cache", type=Path)
    args = parser.parse_args()

    source = args.app.read_text(encoding="utf-8")
    evaluator_source, marker, _ = source.partition("# def run_gradio():")
    if not marker:
        raise RuntimeError(f"Cannot locate evaluator/UI boundary in {args.app}")
    evaluator_source = evaluator_source.replace("import gradio as gr\n", "")
    evaluator_source = evaluator_source.replace(
        "from apscheduler.schedulers.background import BackgroundScheduler\n", ""
    )
    module_name = "bigcodebench_evaluator_core"
    module = types.ModuleType(module_name)
    module.__file__ = str(args.app)
    sys.modules[module_name] = module
    exec(compile(evaluator_source, str(args.app), "exec"), module.__dict__)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise RuntimeError(f"Official evaluate function not found in {args.app}")

    dataset_hash = module.get_bigcodebench_hash(subset="full")
    cache_path = Path(module.CACHE_DIR) / f"{dataset_hash}.pkl"
    if args.groundtruth_cache is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.groundtruth_cache, cache_path)

    _, groundtruth = evaluate(
        split="complete",
        subset="full",
        samples="__dummy__.jsonl",
        pass_k="1",
        parallel=args.parallel,
        calibrated=True,
        check_gt_only=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_cache_path = args.output_dir / f"groundtruth_{dataset_hash}.pkl"
    shutil.copy2(cache_path, frozen_cache_path)
    cache_sha256 = hashlib.sha256(frozen_cache_path.read_bytes()).hexdigest()
    groundtruth_path = args.output_dir / "groundtruth_manifest.json"
    groundtruth_path.write_text(
        json.dumps(
            {
                "dataset_hash": dataset_hash,
                "cache_sha256": cache_sha256,
                "cache_path": str(frozen_cache_path),
                "groundtruth": groundtruth,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = []
    for samples_path in args.samples:
        results, pass_at_k = evaluate(
            split="complete",
            subset="full",
            samples=str(samples_path),
            pass_k="1",
            parallel=args.parallel,
            calibrated=True,
        )
        stem = samples_path.stem
        results_path = args.output_dir / f"{stem}_eval_results.json"
        pass_path = args.output_dir / f"{stem}_pass_at_k.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        pass_path.write_text(json.dumps(pass_at_k, indent=2), encoding="utf-8")
        outputs.append(
            {
                "samples": str(samples_path),
                "results": str(results_path),
                "pass_at_k": pass_at_k,
            }
        )

    print(
        json.dumps(
            {
                "groundtruth_manifest": str(groundtruth_path),
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
