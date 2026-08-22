# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0", "seaborn>=0.13"]
# ///
# --- How to run ---
# uv run paper/tools/build_results_tradeoff_figure.py

"""Build the paper's result-summary and mechanism-action figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

FULL_WIDTH: Final = 516.0
RESULTS_HEIGHT: Final = 190.0
COLUMN_WIDTH: Final = 246.0
MECHANISM_HEIGHT: Final = 152.0

BLUE: Final = (0.10, 0.32, 0.62)
ORANGE: Final = (0.90, 0.45, 0.08)
GREEN: Final = (0.08, 0.52, 0.32)
CODE_COLOR: Final = (0.15, 0.38, 0.67)
MATH_COLOR: Final = (0.86, 0.38, 0.10)


@dataclass(frozen=True, slots=True)
class Method:
    name: str
    color: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Retention:
    domain: str
    method: Method
    percent: float
    tokens: int


@dataclass(frozen=True, slots=True)
class BenchmarkDelta:
    domain: str
    benchmark: str
    ours: float
    data_juicer: float
    nemo: float


@dataclass(frozen=True, slots=True)
class MechanismAction:
    label: str
    code_count: int
    math_count: int


OURS: Final = Method("Ours", BLUE)
DATA_JUICER: Final = Method("Data-Juicer", ORANGE)
NEMO: Final = Method("NeMo Curator", GREEN)
METHODS: Final = (OURS, DATA_JUICER, NEMO)

RETENTION: Final = (
    Retention("Code", OURS, 89.44, 6_242_304),
    Retention("Code", DATA_JUICER, 78.87, 5_505_024),
    Retention("Code", NEMO, 86.38, 6_029_312),
    Retention("Math", OURS, 82.63, 5_767_168),
    Retention("Math", DATA_JUICER, 65.96, 4_603_904),
    Retention("Math", NEMO, 84.27, 5_881_856),
)

DELTAS: Final = (
    BenchmarkDelta("Code", "HumanEval+", 2.64, -0.61, 1.22),
    BenchmarkDelta("Code", "MBPP+", 5.73, -7.67, 2.03),
    BenchmarkDelta("Code", "BigCodeBench", -1.49, -0.26, -0.20),
    BenchmarkDelta("Code", "CRUXEval-I", -1.38, 2.62, 0.25),
    BenchmarkDelta("Code", "CRUXEval-O", 0.75, -0.25, 0.17),
    BenchmarkDelta("Code", "DS-1000", 0.93, 1.03, 1.43),
    BenchmarkDelta("Math", "GSM8K strict", -0.26, 0.93, 0.50),
    BenchmarkDelta("Math", "GSM8K flexible", -8.11, -3.59, -6.32),
    BenchmarkDelta("Math", "GSM8K normalized", -1.06, -0.33, 0.23),
    BenchmarkDelta("Math", "MATH-500", -1.60, -0.60, -0.60),
)

MECHANISM_ACTIONS: Final = (
    MechanismAction("Explicit exclusion", 40, 1),
    MechanismAction("Deduplication", 4, 10),
    MechanismAction("Quality not selected", 816, 1_223),
    MechanismAction("Coverage restoration", 104, 143),
)
STAGE_A_COUNTS: Final = {"Code": 8_026, "Math": 6_619}


def build_results_figure(output: Path) -> None:
    sns.set_theme(style="white", rc={"font.family": "Arial"})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.titlesize": 8.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
        }
    )
    figure, (retention_axis, delta_axis) = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH / 72.0, RESULTS_HEIGHT / 72.0),
        gridspec_kw={"width_ratios": (0.42, 0.58)},
    )
    figure.subplots_adjust(left=0.115, right=0.995, bottom=0.19, top=0.79, wspace=0.44)

    y_positions = np.array((5.15, 4.25, 3.35, 1.75, 0.85, -0.05))
    percentages = np.array(tuple(item.percent for item in RETENTION))
    method_colors = tuple(item.method.color for item in RETENTION)
    retention_axis.barh(
        y_positions,
        np.full(len(RETENTION), 100.0),
        height=0.54,
        color="#EEF1F4",
        edgecolor="#D6DCE2",
        linewidth=0.45,
    )
    retention_axis.barh(
        y_positions,
        percentages,
        height=0.54,
        color=method_colors,
        edgecolor="white",
        linewidth=0.45,
    )
    for y, item in zip(y_positions, RETENTION, strict=True):
        retention_axis.text(
            2.2,
            y,
            f"{item.percent:.1f}%",
            color="white",
            fontsize=6.4,
            fontweight="bold",
            ha="left",
            va="center",
        )
        retention_axis.text(
            item.percent + 1.5,
            y,
            f"{item.tokens / 1_000_000:.2f}M",
            color="#343A40",
            fontsize=6.1,
            ha="left",
            va="center",
        )
    retention_axis.set_yticks(y_positions, tuple(item.method.name for item in RETENTION))
    retention_axis.set_xlim(0, 116)
    retention_axis.set_ylim(-0.55, 5.75)
    retention_axis.set_xticks((0, 25, 50, 75, 100))
    retention_axis.set_xlabel("Raw packed tokens retained (%)", labelpad=2)
    retention_axis.set_title("(a) Packed-token retention", loc="left", pad=20)
    retention_axis.axvline(100, color="#636B74", linewidth=0.8, linestyle=(0, (2, 2)))
    retention_axis.axhline(2.55, color="#D6DCE2", linewidth=0.7)
    retention_axis.text(
        0.0,
        5.66,
        "CODE",
        color="#5B6570",
        fontsize=6.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    retention_axis.text(
        0.0,
        2.50,
        "MATH",
        color="#5B6570",
        fontsize=6.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    retention_axis.grid(axis="x", color="#E1E5E9", linewidth=0.45)
    retention_axis.grid(axis="y", visible=False)
    retention_axis.tick_params(axis="both", length=0)
    sns.despine(ax=retention_axis, left=True, bottom=True)

    delta_values = np.array(
        tuple((item.ours, item.data_juicer, item.nemo) for item in DELTAS)
    )
    delta_labels = np.array(
        tuple(tuple(f"{value:+.2f}" for value in row) for row in delta_values)
    )
    diverging_map = LinearSegmentedColormap.from_list(
        "paper_delta",
        ("#C5663E", "#F7F7F5", "#2A64A8"),
    )
    sns.heatmap(
        delta_values,
        ax=delta_axis,
        annot=delta_labels,
        fmt="",
        cmap=diverging_map,
        center=0.0,
        vmin=-8.5,
        vmax=8.5,
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        xticklabels=tuple(method.name for method in METHODS),
        yticklabels=(
            "HumanEval+",
            "MBPP+",
            "BigCodeBench",
            "CRUXEval-I",
            "CRUXEval-O",
            "DS-1000",
            "GSM8K strict",
            "GSM8K flexible",
            "GSM8K normalized",
            "MATH-500",
        ),
        annot_kws={"fontsize": 6.2, "fontweight": "bold"},
    )
    for text_label, value in zip(delta_axis.texts, delta_values.flat, strict=True):
        text_label.set_color("white" if abs(value) >= 4.5 else "#20252A")
    delta_axis.xaxis.tick_top()
    delta_axis.tick_params(axis="x", length=0, pad=4)
    delta_axis.tick_params(axis="y", length=0, pad=3)
    delta_axis.set_xticklabels(
        tuple(method.name for method in METHODS),
        rotation=0,
        fontsize=6.5,
        fontweight="bold",
    )
    delta_axis.set_yticklabels(delta_axis.get_yticklabels(), rotation=0, fontsize=6.1)
    delta_axis.set_title("(b) Score change from Raw (pp)", loc="left", pad=20)
    delta_axis.axhline(6, color="#6B737C", linewidth=1.15)
    delta_axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            len(DELTAS),
            fill=False,
            edgecolor="#1B4F8A",
            linewidth=1.25,
            clip_on=False,
        )
    )
    delta_axis.text(
        0.0,
        -0.11,
        "orange = lower than Raw",
        transform=delta_axis.transAxes,
        color="#8E4A2D",
        fontsize=5.7,
        ha="left",
    )
    delta_axis.text(
        1.0,
        -0.11,
        "blue = higher than Raw",
        transform=delta_axis.transAxes,
        color="#1B4F8A",
        fontsize=5.7,
        ha="right",
    )

    figure.savefig(output, format="pdf", facecolor="white")
    plt.close(figure)


def build_mechanism_figure(output: Path) -> None:
    sns.set_theme(style="white", rc={"font.family": "Arial"})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.4,
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.titlesize": 7.8,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.9,
        }
    )
    figure, axis = plt.subplots(
        figsize=(COLUMN_WIDTH / 72.0, MECHANISM_HEIGHT / 72.0)
    )
    figure.subplots_adjust(left=0.37, right=0.985, bottom=0.22, top=0.78)

    row_positions = np.arange(len(MECHANISM_ACTIONS), dtype=float)
    bar_height = 0.27
    code_percentages = np.array(
        tuple(
            action.code_count / STAGE_A_COUNTS["Code"] * 100.0
            for action in MECHANISM_ACTIONS
        )
    )
    math_percentages = np.array(
        tuple(
            action.math_count / STAGE_A_COUNTS["Math"] * 100.0
            for action in MECHANISM_ACTIONS
        )
    )
    code_bars = axis.barh(
        row_positions - bar_height / 1.7,
        code_percentages,
        height=bar_height,
        color=CODE_COLOR,
        edgecolor="white",
        linewidth=0.45,
        label="Code",
    )
    math_bars = axis.barh(
        row_positions + bar_height / 1.7,
        math_percentages,
        height=bar_height,
        color=MATH_COLOR,
        edgecolor="white",
        linewidth=0.45,
        label="Math",
    )
    for bars, counts, percentages, color in (
        (
            code_bars,
            tuple(action.code_count for action in MECHANISM_ACTIONS),
            code_percentages,
            CODE_COLOR,
        ),
        (
            math_bars,
            tuple(action.math_count for action in MECHANISM_ACTIONS),
            math_percentages,
            MATH_COLOR,
        ),
    ):
        for bar, count, percent in zip(bars, counts, percentages, strict=True):
            axis.text(
                max(percent + 0.38, 0.42),
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({percent:.2f}%)",
                color=color,
                fontsize=5.4,
                fontweight="bold",
                ha="left",
                va="center",
            )

    axis.set_yticks(row_positions, tuple(action.label for action in MECHANISM_ACTIONS))
    axis.invert_yaxis()
    axis.set_xlim(0, 23.5)
    axis.set_xticks((0, 5, 10, 15, 20))
    axis.set_xlabel("Share of Stage A chunks (%)", labelpad=3)
    axis.set_title("Membership-changing actions", loc="left", pad=18)
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.23),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        fontsize=5.9,
    )
    axis.grid(axis="x", color="#E0E4E8", linewidth=0.45)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", length=0)
    sns.despine(ax=axis, left=True, bottom=True)

    figure.savefig(output, format="pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    figures = Path(__file__).resolve().parents[1] / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    build_results_figure(figures / "results_tradeoff.pdf")
    build_mechanism_figure(figures / "mechanism_actions.pdf")


if __name__ == "__main__":
    main()
