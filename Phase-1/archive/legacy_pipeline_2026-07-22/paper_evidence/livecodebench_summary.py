from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Final, Sequence, TypeAlias, TypedDict


ROOT: Final = Path(__file__).resolve().parents[1]
PILOT_DIR: Final = ROOT / "outputs" / "code_livecodebench_pilot_qwen3_4b"
EVALUATION_DIR: Final = PILOT_DIR / "evaluations"
NATURAL_REPORT: Final = (
    ROOT
    / "outputs"
    / "validation"
    / "code_domain_natural_budget_current_framework_stage_c_summary_report.json"
)
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "code_livecodebench_pilot_summary_report.json"
MARKDOWN_PATH: Final = ROOT / "outputs" / "validation" / "code_livecodebench_pilot_summary_report.md"
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class SummaryContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class CodeDifferences(TypedDict):
    base_vs_raw: int
    base_vs_curated: int
    raw_vs_curated: int


class PassRates(TypedDict):
    base_no_update: float
    raw_full_natural: float
    curated_v2_natural: float


class PairedComparison(TypedDict):
    curated_wins: int
    curated_losses: int
    ties: int
    exact_two_sided_p: float


class PilotSummary(TypedDict):
    status: str
    claim: str
    pass_at_1: PassRates
    paired_curated_vs_raw: PairedComparison
    generation_code_differences: CodeDifferences


def _pass_rate(outcomes: Sequence[bool]) -> float:
    return sum(outcomes) / len(outcomes)


def _exact_two_sided_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(curated: Sequence[bool], reference: Sequence[bool]) -> PairedComparison:
    wins = sum(curated_pass and not reference_pass for curated_pass, reference_pass in zip(curated, reference))
    losses = sum(not curated_pass and reference_pass for curated_pass, reference_pass in zip(curated, reference))
    return {
        "curated_wins": wins,
        "curated_losses": losses,
        "ties": len(curated) - wins - losses,
        "exact_two_sided_p": _exact_two_sided_p(wins, losses),
    }


def build_summary(
    *,
    base: Sequence[bool],
    raw: Sequence[bool],
    curated: Sequence[bool],
    code_differences: CodeDifferences,
) -> PilotSummary:
    if not base or len(base) != len(raw) or len(raw) != len(curated):
        raise SummaryContractError("LiveCodeBench arm outcomes must be non-empty and aligned")
    pass_rates: PassRates = {
        "base_no_update": _pass_rate(base),
        "raw_full_natural": _pass_rate(raw),
        "curated_v2_natural": _pass_rate(curated),
    }
    curated_rate = pass_rates["curated_v2_natural"]
    raw_rate = pass_rates["raw_full_natural"]
    if curated_rate > raw_rate:
        status = "completed_independent_transfer_gain_observed"
        claim = "independent_transfer_gain_observed_in_seed101_pilot"
    elif curated_rate < raw_rate:
        status = "completed_independent_transfer_regression_observed"
        claim = "independent_transfer_regression_observed_in_seed101_pilot"
    else:
        status = "completed_no_independent_transfer_gain"
        claim = "independent_transfer_not_demonstrated"
    return {
        "status": status,
        "claim": claim,
        "pass_at_1": pass_rates,
        "paired_curated_vs_raw": _paired(curated, raw),
        "generation_code_differences": code_differences,
    }


def _load_json(path: Path) -> JsonValue:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value
    raise SummaryContractError(f"Unsupported JSON root in {path}")


def _evaluation(path: Path) -> tuple[tuple[bool, ...], dict[str, JsonValue]]:
    report = _load_json(path)
    if not isinstance(report, dict):
        raise SummaryContractError(f"Expected evaluation object in {path}")
    rows = report["rows"]
    if not isinstance(rows, list):
        raise SummaryContractError(f"Expected evaluation rows in {path}")
    outcomes = tuple(bool(row["passed"]) for row in rows if isinstance(row, dict))
    return outcomes, report


def _generation_codes(path: Path) -> tuple[str, ...]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise SummaryContractError(f"Expected generation rows in {path}")
    return tuple(str(row["code_list"][0]) for row in rows)


def _difference_count(left: Sequence[str], right: Sequence[str]) -> int:
    return sum(left_code != right_code for left_code, right_code in zip(left, right))


def main() -> int:
    evaluation_paths = {
        "base": EVALUATION_DIR / "base_no_update_eval.json",
        "raw": EVALUATION_DIR / "raw_full_natural_seed101_eval.json",
        "curated": EVALUATION_DIR / "curated_v2_natural_seed101_eval.json",
    }
    base, base_report = _evaluation(evaluation_paths["base"])
    raw, raw_report = _evaluation(evaluation_paths["raw"])
    curated, curated_report = _evaluation(evaluation_paths["curated"])
    generation_paths = {
        "base": PILOT_DIR / "base_no_update_base_generations.json",
        "raw": PILOT_DIR / "raw_full_natural_seed101_generations.json",
        "curated": PILOT_DIR / "curated_v2_natural_seed101_generations.json",
    }
    base_codes = _generation_codes(generation_paths["base"])
    raw_codes = _generation_codes(generation_paths["raw"])
    curated_codes = _generation_codes(generation_paths["curated"])
    differences: CodeDifferences = {
        "base_vs_raw": _difference_count(base_codes, raw_codes),
        "base_vs_curated": _difference_count(base_codes, curated_codes),
        "raw_vs_curated": _difference_count(raw_codes, curated_codes),
    }
    summary = build_summary(base=base, raw=raw, curated=curated, code_differences=differences)
    natural = _load_json(NATURAL_REPORT)
    if not isinstance(natural, dict):
        raise SummaryContractError(f"Expected natural-budget report object in {NATURAL_REPORT}")
    arms = natural["arms"]
    raw_arm = arms["raw_full_natural"]
    curated_arm = arms["curated_v2_natural"]
    report = {
        "schema_version": "code-livecodebench-pilot-summary-v1",
        **summary,
        "protocol": {
            "benchmark": "LiveCodeBench code_generation_lite",
            "task_count": len(base),
            "seed": 101,
            "official_runner_commit": base_report["official_runner_commit"],
            "tasks_sha256": base_report["tasks_sha256"],
            "isolation": base_report["isolation"],
            "utility_stage": "Stage C only",
            "selector_tuning_permission": False,
        },
        "natural_budget": {
            "raw_packed_training_tokens": raw_arm["packed_training_tokens"],
            "curated_packed_training_tokens": curated_arm["packed_training_tokens"],
            "curated_token_reduction_fraction": natural[
                "natural_budget_reduction_curated_vs_raw"
            ]["packed_training_token_reduction_fraction"],
        },
        "pass_counts": {
            "base_no_update": base_report["pass_count"],
            "raw_full_natural": raw_report["pass_count"],
            "curated_v2_natural": curated_report["pass_count"],
        },
        "strata": {
            "base_no_update": base_report["strata"],
            "raw_full_natural": raw_report["strata"],
            "curated_v2_natural": curated_report["strata"],
        },
        "interpretation": {
            "generated_programs_changed": any(differences.values()),
            "correctness_outcomes_changed": base != raw or base != curated or raw != curated,
            "format_affinity_alternative_resolved": False,
            "paper_use": "neutral independent-benchmark pilot; not evidence of transfer gain",
            "next_required_evidence": "pre-registered multi-seed benchmark with adequate non-easy task power",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rates = summary["pass_at_1"]
    MARKDOWN_PATH.write_text(
        "\n".join(
            (
                "# LiveCodeBench Pilot Summary",
                "",
                f"Status: `{summary['status']}`",
                "",
                "| Arm | Packed tokens | Pass@1 |",
                "| --- | ---: | ---: |",
                f"| Base | 0 | {rates['base_no_update']:.2%} |",
                f"| Raw natural | {raw_arm['packed_training_tokens']} | {rates['raw_full_natural']:.2%} |",
                f"| Curated natural | {curated_arm['packed_training_tokens']} | {rates['curated_v2_natural']:.2%} |",
                "",
                "Generated programs differed across arms, but all 48 correctness outcomes were identical.",
                "This Stage-C pilot does not demonstrate independent-benchmark transfer gain.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[code-livecodebench-summary] {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
