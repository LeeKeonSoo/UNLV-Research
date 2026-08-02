# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.10", "transformers>=4.45"]
# ///
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus_profiler import TransformersTokenCounter, profile_jsonl
from model_provider_contract import load_provider_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile JSONL corpora without selecting or deleting records.")
    parser.add_argument("--input", action="append", required=True, type=Path, help="Input JSONL; repeat for multiple shards.")
    parser.add_argument("--text-field", action="append", default=None, help="Permitted text field; default: text.")
    parser.add_argument("--provider-registry", type=Path, default=ROOT / "configs" / "model_provider_registry_v1.json")
    parser.add_argument("--tokenizer", help="Frozen local tokenizer path. Omit only for a structural audit.")
    parser.add_argument("--tokenizer-revision", default="local-frozen-artifact")
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True, type=Path, help="Audit JSON output; never a curated dataset path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_counter = TransformersTokenCounter(args.tokenizer, args.tokenizer_revision, args.append_eos) if args.tokenizer else None
    report = profile_jsonl(
        input_paths=tuple(args.input),
        text_fields=tuple(args.text_field or ("text",)),
        provider_registry=load_provider_registry(args.provider_registry),
        token_counter=token_counter,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"[audit-only-corpus-profiler] records={report.invariants.records_read} output={args.output}")


if __name__ == "__main__":
    main()
