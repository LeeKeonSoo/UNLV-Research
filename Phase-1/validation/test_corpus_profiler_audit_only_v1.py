#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus_profiler import CorpusProfileError, TokenizerIdentity, profile_jsonl
from model_provider_contract import load_provider_registry


REGISTRY = ROOT / "configs" / "model_provider_registry_v1.json"
CONTRACT = ROOT / "configs" / "corpus_profiler_contract_v1.json"


class FixtureTokenCounter:
    identity = TokenizerIdentity(
        tokenizer_id="fixture-whitespace-tokenizer",
        revision="fixture-v1",
        add_special_tokens=False,
        append_eos_per_record=True,
    )

    def count(self, text: str) -> int:
        return len(text.split()) + 1


def test_profiler_reports_opportunities_without_selecting_or_mutating() -> None:
    rows = (
        b'{"record_id":"a","text":"def add(x, y):\\n    return x + y\\nimport math"}\n',
        b'{"record_id":"b","text":"def add(x, y):\\n    return x + y\\nimport math"}\n',
        b'{"record_id":"c","text":"The theorem follows from the matrix equation. Proof: use \\\\frac{1}{2}."}\n',
        '{"record_id":"d","text":"한글과 English가 함께 있는 설명 문서입니다. 이 문서는 두 문장입니다."}\n'.encode(),
    )
    with TemporaryDirectory() as directory:
        input_path = Path(directory) / "sample.jsonl"
        input_path.write_bytes(b"".join(rows))
        before = input_path.read_bytes()

        report = profile_jsonl(
            input_paths=(input_path,),
            text_fields=("text",),
            provider_registry=load_provider_registry(REGISTRY),
            token_counter=FixtureTokenCounter(),
        )

        assert input_path.read_bytes() == before
        assert report.status == "audit_only_complete"
        assert report.invariants.records_read == 4
        assert report.invariants.records_selected is None
        assert report.invariants.records_removed == 0
        assert report.invariants.output_dataset_written is False
        assert report.exact_duplicate_opportunity.family_count == 1
        assert report.exact_duplicate_opportunity.excess_record_count == 1
        assert report.target_tokenizer.total_tokens > 0
        assert report.routing.route_status["routed"] >= 2
        assert report.providers.selection_authority is False


def test_profiler_rejects_ambiguous_text_fields() -> None:
    with TemporaryDirectory() as directory:
        input_path = Path(directory) / "ambiguous.jsonl"
        input_path.write_text('{"text":"one","content":"two"}\n', encoding="utf-8")

        try:
            profile_jsonl(
                input_paths=(input_path,),
                text_fields=("text", "content"),
                provider_registry=load_provider_registry(REGISTRY),
                token_counter=None,
            )
        except CorpusProfileError as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("Ambiguous text fields must fail closed")


def test_machine_contract_forbids_every_selection_side_effect() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["authority"] == "measurement_only"
    assert contract["selection_authority"] is False
    assert {
        "rank_records",
        "select_records",
        "remove_records",
        "write_curated_dataset",
        "execute_model_provider_scores",
    } <= set(contract["forbidden_actions"])
    assert contract["token_count_boundary"]["release_profile_requires_exact_frozen_target_tokenizer"] is True


if __name__ == "__main__":
    test_profiler_reports_opportunities_without_selecting_or_mutating()
    test_profiler_rejects_ambiguous_text_fields()
    test_machine_contract_forbids_every_selection_side_effect()
    print("[corpus-profiler-audit-only-v1] no-selection invariants: pass")
