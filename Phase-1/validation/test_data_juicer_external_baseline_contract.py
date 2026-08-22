#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(
        (ROOT / "protocols" / "code_7m_data_juicer_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = ROOT / protocol["published_recipe"]["local_adaptation"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_external_curation"
    assert protocol["implementation"]["version"] == "1.5.5"
    assert (
        sha256_file(config_path)
        == protocol["published_recipe"]["local_adaptation_sha256"]
    )
    assert protocol["execution"]["target_token_budget"] is None
    assert protocol["execution"]["benchmark_feedback"] is False
    assert config["text_keys"] == "text"
    assert config["open_tracer"] is True
    operators = [next(iter(item)) for item in config["process"]]
    assert operators == [
        "clean_email_mapper",
        "clean_links_mapper",
        "fix_unicode_mapper",
        "punctuation_normalization_mapper",
        "whitespace_normalization_mapper",
        "clean_copyright_mapper",
        "alphanumeric_filter",
        "alphanumeric_filter",
        "average_line_length_filter",
        "character_repetition_filter",
        "maximum_line_length_filter",
        "text_length_filter",
        "words_num_filter",
        "word_repetition_filter",
        "document_simhash_deduplicator",
    ]

    corrected_protocol = json.loads(
        (ROOT / "protocols" / "code_7m_data_juicer_baseline_v2.json").read_text(
            encoding="utf-8"
        )
    )
    corrected_config_path = (
        ROOT / corrected_protocol["published_recipe"]["local_adaptation"]
    )
    corrected_config = yaml.safe_load(
        corrected_config_path.read_text(encoding="utf-8")
    )
    compatibility_patch_path = (
        ROOT / corrected_protocol["compatibility_patch"]["path"]
    )

    assert corrected_protocol["implementation"] == protocol["implementation"]
    assert corrected_protocol["published_recipe"]["commit"] == protocol[
        "published_recipe"
    ]["commit"]
    assert corrected_config["process"] == config["process"]
    assert (
        sha256_file(corrected_config_path)
        == corrected_protocol["published_recipe"]["local_adaptation_sha256"]
    )
    assert (
        sha256_file(compatibility_patch_path)
        == corrected_protocol["compatibility_patch"]["sha256"]
    )
    assert corrected_protocol["compatibility_patch"]["operator_changes"] == []
    assert corrected_protocol["compatibility_patch"]["threshold_changes"] == []
    assert corrected_protocol["compatibility_patch"]["record_changes"] == []
    assert corrected_protocol["preflight_failure"]["exported_records"] == 0
    assert corrected_config["export_path"] != config["export_path"]
    print("[data-juicer-baseline] frozen official recipe adaptation: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
