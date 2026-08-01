from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Final, Mapping, Sequence, TypedDict


ROOT: Final = Path(__file__).resolve().parents[1]
CONFIRMATION_DIR: Final = ROOT / "outputs" / "code_livecodebench_confirmation_qwen3_4b"
EVALUATION_DIR: Final = CONFIRMATION_DIR / "evaluations"
BASE_EVALUATION: Final = (
    ROOT / "outputs" / "code_livecodebench_pilot_qwen3_4b" / "evaluations" / "base_no_update_eval.json"
)
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "code_livecodebench_confirmation_summary_report.json"
MARKDOWN_PATH: Final = ROOT / "outputs" / "validation" / "code_livecodebench_confirmation_summary_report.md"
SEEDS: Final = (101, 131, 163, 197, 239)


@dataclass(frozen=True, slots=True)
class ConfirmationSummaryError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class PairedCounts(TypedDict):
    curated_wins: int
    curated_losses: int
    ties: int
    exact_two_sided_p: float


class SeedResult(TypedDict):
    seed: int
    raw_pass_count: int
    curated_pass_count: int
    raw_pass_rate: float
    curated_pass_rate: float
    pass_count_delta: int
    paired: PairedCounts


class ConfirmationSummary(TypedDict):
    status: str
    claim: str
    raw_mean_pass_rate: float
    curated_mean_pass_rate: float
    mean_pass_rate_delta: float
    pooled_paired: PairedCounts
    per_seed: tuple[SeedResult, ...]


class EvaluationRows(TypedDict):
    outcomes: dict[str, bool]
    task_count: int
    task_sha256: str
    isolation: str
    official_runner_commit: str


def _rate(outcomes: Sequence[bool]) -> float:
    return sum(outcomes) / len(outcomes)


def _exact_two_sided_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(curated: Sequence[bool], raw: Sequence[bool]) -> PairedCounts:
    wins = sum(curated_pass and not raw_pass for curated_pass, raw_pass in zip(curated, raw))
    losses = sum(not curated_pass and raw_pass for curated_pass, raw_pass in zip(curated, raw))
    return {
        "curated_wins": wins,
        "curated_losses": losses,
        "ties": len(curated) - wins - losses,
        "exact_two_sided_p": _exact_two_sided_p(wins, losses),
    }


def _aligned_seed_set(
    raw_by_seed: Mapping[int, Sequence[bool]],
    curated_by_seed: Mapping[int, Sequence[bool]],
) -> tuple[int, ...]:
    raw_seeds = tuple(sorted(raw_by_seed))
    curated_seeds = tuple(sorted(curated_by_seed))
    if not raw_seeds or raw_seeds != curated_seeds:
        raise ConfirmationSummaryError("Raw and curated confirmation seed sets must be identical and non-empty")
    return raw_seeds


def build_summary(
    *,
    raw_by_seed: Mapping[int, Sequence[bool]],
    curated_by_seed: Mapping[int, Sequence[bool]],
) -> ConfirmationSummary:
    seed_results: list[SeedResult] = []
    pooled_raw: list[bool] = []
    pooled_curated: list[bool] = []
    for seed in _aligned_seed_set(raw_by_seed, curated_by_seed):
        raw = tuple(raw_by_seed[seed])
        curated = tuple(curated_by_seed[seed])
        if not raw or len(raw) != len(curated):
            raise ConfirmationSummaryError(f"Seed {seed} outcomes must be non-empty and aligned")
        paired = _paired(curated, raw)
        seed_results.append(
            {
                "seed": seed,
                "raw_pass_count": sum(raw),
                "curated_pass_count": sum(curated),
                "raw_pass_rate": _rate(raw),
                "curated_pass_rate": _rate(curated),
                "pass_count_delta": sum(curated) - sum(raw),
                "paired": paired,
            }
        )
        pooled_raw.extend(raw)
        pooled_curated.extend(curated)
    raw_mean = sum(row["raw_pass_rate"] for row in seed_results) / len(seed_results)
    curated_mean = sum(row["curated_pass_rate"] for row in seed_results) / len(seed_results)
    pooled = _paired(pooled_curated, pooled_raw)
    return {
        "status": "completed_multiseed_external_transfer_inconclusive",
        "claim": "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation",
        "raw_mean_pass_rate": raw_mean,
        "curated_mean_pass_rate": curated_mean,
        "mean_pass_rate_delta": curated_mean - raw_mean,
        "pooled_paired": pooled,
        "per_seed": tuple(seed_results),
    }


def _load_evaluation(path: Path) -> EvaluationRows:
    payload = json.loads(path.read_text(encoding="utf-8"))
    match payload:
        case {
            "rows": list() as rows,
            "task_count": int() as task_count,
            "tasks_sha256": str() as task_sha256,
            "isolation": str() as isolation,
            "official_runner_commit": str() as official_runner_commit,
        }:
            outcomes: dict[str, bool] = {}
            for row in rows:
                match row:
                    case {"question_id": str() as question_id, "passed": bool() as passed}:
                        if question_id in outcomes:
                            raise ConfirmationSummaryError(f"Duplicate task {question_id} in {path}")
                        outcomes[question_id] = passed
                    case _:
                        raise ConfirmationSummaryError(f"Malformed evaluation row in {path}")
            if len(outcomes) != task_count:
                raise ConfirmationSummaryError(f"Task count mismatch in {path}")
            return {
                "outcomes": outcomes,
                "task_count": task_count,
                "task_sha256": task_sha256,
                "isolation": isolation,
                "official_runner_commit": official_runner_commit,
            }
        case _:
            raise ConfirmationSummaryError(f"Malformed evaluation report in {path}")


def _seed_evaluation_path(arm: str, seed: int) -> Path:
    return EVALUATION_DIR / f"{arm}_seed{seed}_eval.json"


def _load_seed_outcomes(arm: str) -> tuple[dict[int, tuple[bool, ...]], EvaluationRows]:
    by_seed: dict[int, tuple[bool, ...]] = {}
    reference: EvaluationRows | None = None
    for seed in SEEDS:
        evaluation = _load_evaluation(_seed_evaluation_path(arm, seed))
        task_ids = tuple(sorted(evaluation["outcomes"]))
        by_seed[seed] = tuple(evaluation["outcomes"][task_id] for task_id in task_ids)
        if reference is None:
            reference = evaluation
        elif (
            evaluation["task_sha256"] != reference["task_sha256"]
            or evaluation["task_count"] != reference["task_count"]
            or tuple(sorted(evaluation["outcomes"])) != tuple(sorted(reference["outcomes"]))
        ):
            raise ConfirmationSummaryError(f"Seed {seed} does not match frozen task bundle")
    if reference is None:
        raise ConfirmationSummaryError("No confirmation evaluations found")
    return by_seed, reference


def main() -> int:
    raw_by_seed, raw_reference = _load_seed_outcomes("raw_full_natural")
    curated_by_seed, curated_reference = _load_seed_outcomes("curated_v2_natural")
    if (
        raw_reference["task_sha256"] != curated_reference["task_sha256"]
        or tuple(sorted(raw_reference["outcomes"])) != tuple(sorted(curated_reference["outcomes"]))
    ):
        raise ConfirmationSummaryError("Raw and curated evaluations use different task bundles")
    base = _load_evaluation(BASE_EVALUATION)
    if base["task_sha256"] != raw_reference["task_sha256"]:
        raise ConfirmationSummaryError("Base evaluation uses a different task bundle")
    summary = build_summary(raw_by_seed=raw_by_seed, curated_by_seed=curated_by_seed)
    report = {
        "schema_version": "code-livecodebench-confirmation-summary-v1",
        **summary,
        "protocol": {
            "benchmark": "LiveCodeBench code_generation_lite",
            "task_count": raw_reference["task_count"],
            "seeds": list(SEEDS),
            "tasks_sha256": raw_reference["task_sha256"],
            "official_runner_commit": raw_reference["official_runner_commit"],
            "isolation": raw_reference["isolation"],
            "base_no_update_pass_rate": _rate(tuple(base["outcomes"].values())),
            "utility_stage": "Stage C only",
            "selector_tuning_permission": False,
        },
        "interpretation": {
            "paper_use": "external-transfer limitation; not evidence of LiveCodeBench transfer gain",
            "selector_policy_changed": False,
            "current_framework_confirmation": True,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LiveCodeBench Confirmation Summary",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Seed | Raw pass@1 | Curated pass@1 | Delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['raw_pass_rate']:.2%} | {row['curated_pass_rate']:.2%} | {row['pass_count_delta']:+d} |"
        )
    lines.extend(
        (
            "",
            f"Mean raw pass@1: `{summary['raw_mean_pass_rate']:.2%}`",
            f"Mean curated pass@1: `{summary['curated_mean_pass_rate']:.2%}`",
            f"Mean delta: `{summary['mean_pass_rate_delta']:.2%}`",
            f"Paired wins/losses/ties: `{summary['pooled_paired']['curated_wins']}/{summary['pooled_paired']['curated_losses']}/{summary['pooled_paired']['ties']}`",
            f"Exact two-sided p: `{summary['pooled_paired']['exact_two_sided_p']:.4f}`",
            "",
        )
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[code-livecodebench-confirmation-summary] {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
