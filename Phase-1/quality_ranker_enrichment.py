from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reason_codes(result: dict[str, Any]) -> set[str]:
    passes = tuple(result.get("first_pass") or ()) + tuple(result.get("second_pass") or ())
    return {
        str(code)
        for vote in passes
        for code in vote.get("reason_codes") or ()
    }


def materialize_target_policy_observations(
    fixture_path: Path,
    observation_path: Path,
    output_path: Path,
) -> Path:
    fixtures = _read_jsonl(fixture_path)
    observations = _read_jsonl(observation_path)
    fixture_by_uid = {str(row["chunk_uid"]): row for row in fixtures}
    observation_by_uid = {str(row["chunk_uid"]): row for row in observations}
    if len(fixture_by_uid) != len(fixtures) or len(observation_by_uid) != len(observations):
        raise RuntimeError("target_policy_observation_uid_collision")
    if set(fixture_by_uid) != set(observation_by_uid):
        raise RuntimeError("target_policy_observation_universe_mismatch")

    filtered: list[dict[str, Any]] = []
    decision_counts: dict[str, Counter[str]] = defaultdict(Counter)
    expected_decision_matches = 0
    expected_reason_matches = 0
    for uid in sorted(fixture_by_uid):
        fixture = fixture_by_uid[uid]
        observation = observation_by_uid[uid]
        text_hash = _text_sha256(str(fixture["text"]))
        if str(observation["text_sha256"]) != text_hash:
            raise RuntimeError("target_policy_observation_text_mismatch")
        policy_id = str(fixture["fixture_policy_id"])
        target_results = [
            result
            for result in observation.get("policy_results") or ()
            if str(result.get("policy_id")) == policy_id
        ]
        if len(target_results) != 1:
            raise RuntimeError("target_policy_result_missing")
        target = target_results[0]
        decision = str(target["panel_decision"])
        decision_counts[policy_id][decision] += 1
        expected_decision_matches += int(decision == str(fixture["expected_decision"]))
        expected_reason_matches += int(
            str(fixture["expected_reason_code"]) in _reason_codes(target)
        )
        filtered.append({**observation, "policy_results": [target]})

    _write_jsonl(output_path, filtered)
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    audit = {
        "schema_version": "quality-ranker-target-policy-observations-v1",
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256_file(fixture_path),
        "source_observation_path": str(observation_path),
        "source_observation_sha256": _sha256_file(observation_path),
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "observation_count": len(filtered),
        "policy_results_per_observation": 1,
        "target_policy_decision_counts": {
            policy_id: dict(sorted(counts.items()))
            for policy_id, counts in sorted(decision_counts.items())
        },
        "expected_decision_match_count": expected_decision_matches,
        "expected_reason_match_count": expected_reason_matches,
        "ranker_training_authority": "target_policy_only",
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_path


def materialize_augmented_corpus(
    corpus_path: Path,
    enrichment_path: Path,
    output_path: Path,
) -> Path:
    corpus = _read_jsonl(corpus_path)
    enrichment = _read_jsonl(enrichment_path)
    rows = [*corpus, *enrichment]
    uids = [str(row.get("uid") or row.get("chunk_uid")) for row in rows]
    text_hashes = [_text_sha256(str(row["text"])) for row in rows]
    if len(set(uids)) != len(uids):
        raise RuntimeError("augmented_corpus_uid_overlap")
    if len(set(text_hashes)) != len(text_hashes):
        raise RuntimeError("augmented_corpus_text_overlap")
    _write_jsonl(output_path, rows)
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    audit = {
        "schema_version": "quality-ranker-augmented-corpus-v1",
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "corpus_count": len(corpus),
        "enrichment_path": str(enrichment_path),
        "enrichment_sha256": _sha256_file(enrichment_path),
        "enrichment_count": len(enrichment),
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "output_count": len(rows),
        "unique_uid_count": len(set(uids)),
        "unique_text_sha256_count": len(set(text_hashes)),
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize ranker enrichment artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    filter_parser = subparsers.add_parser("filter-observations")
    filter_parser.add_argument("--fixtures", type=Path, required=True)
    filter_parser.add_argument("--observations", type=Path, required=True)
    filter_parser.add_argument("--output", type=Path, required=True)
    augment_parser = subparsers.add_parser("augment-corpus")
    augment_parser.add_argument("--corpus", type=Path, required=True)
    augment_parser.add_argument("--enrichment", type=Path, required=True)
    augment_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "filter-observations":
        output = materialize_target_policy_observations(
            args.fixtures.resolve(), args.observations.resolve(), args.output.resolve()
        )
    else:
        output = materialize_augmented_corpus(
            args.corpus.resolve(), args.enrichment.resolve(), args.output.resolve()
        )
    print(json.dumps({"status": "complete", "audit": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
