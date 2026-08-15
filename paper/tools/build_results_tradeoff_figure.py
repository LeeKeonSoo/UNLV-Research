"""Build the vector results figure used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


PAGE_WIDTH: Final = 516.0
PAGE_HEIGHT: Final = 188.0
FONT: Final = "PaperArial"
FONT_BOLD: Final = "PaperArial-Bold"
FONT_PATH: Final = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH: Final = Path("C:/Windows/Fonts/arialbd.ttf")
BLUE: Final = (0.12, 0.34, 0.62)
BLUE_LIGHT: Final = (0.72, 0.82, 0.93)
ORANGE: Final = (0.88, 0.45, 0.10)
ORANGE_LIGHT: Final = (0.95, 0.69, 0.43)
GRID: Final = (0.86, 0.86, 0.86)


@dataclass(frozen=True, slots=True)
class ArmPoint:
    name: str
    retention: float
    macro: float
    standard_deviation: float
    color: tuple[float, float, float]


ARMS: Final = (
    ArmPoint("Hard", 72.05, 20.80, 0.44, ORANGE),
    ArmPoint("Normal", 87.70, 20.59, 0.83, BLUE),
    ArmPoint("Raw", 100.00, 20.86, 0.35, (0.08, 0.08, 0.08)),
)

BENCHMARKS: Final = (
    "HumanEval+",
    "MBPP+",
    "BigCodeBench",
    "CRUXEval-I",
    "CRUXEval-O",
    "DS-1000",
)
NORMAL_DELTAS: Final = (2.85, 4.76, -1.49, -0.83, 2.54, -1.30)
HARD_DELTAS: Final = (1.83, 5.56, -1.14, -1.00, 2.83, -0.93)


def draw_text(canvas: Canvas, x: float, y: float, text: str, size: float = 7.0) -> None:
    canvas.setFont(FONT, size)
    canvas.setFillColorRGB(0.08, 0.08, 0.08)
    canvas.drawString(x, y, text)


def draw_centered(canvas: Canvas, x: float, y: float, text: str, size: float = 7.0) -> None:
    canvas.setFont(FONT, size)
    canvas.setFillColorRGB(0.08, 0.08, 0.08)
    canvas.drawCentredString(x, y, text)


def draw_left_panel(canvas: Canvas) -> None:
    left, bottom, width, height = 40.0, 39.0, 198.0, 118.0
    x_min, x_max = 68.0, 102.0
    y_min, y_max = 18.0, 22.0
    x_pos = lambda value: left + (value - x_min) / (x_max - x_min) * width
    y_pos = lambda value: bottom + (value - y_min) / (y_max - y_min) * height

    canvas.setLineWidth(0.35)
    for tick in (70, 80, 90, 100):
        x = x_pos(float(tick))
        canvas.setStrokeColorRGB(*GRID)
        canvas.line(x, bottom, x, bottom + height)
        draw_centered(canvas, x, bottom - 11, str(tick), 6.5)
    for tick in (18, 19, 20, 21, 22):
        y = y_pos(float(tick))
        canvas.setStrokeColorRGB(*GRID)
        canvas.line(left, y, left + width, y)
        draw_text(canvas, left - 14, y - 2, str(tick), 6.5)

    canvas.setStrokeColorRGB(0.25, 0.25, 0.25)
    canvas.line(left, bottom, left + width, bottom)
    canvas.line(left, bottom, left, bottom + height)
    canvas.setDash(3, 2)
    canvas.line(left, y_pos(18.38), left + width, y_pos(18.38))
    canvas.setDash()
    draw_text(canvas, left + 3, y_pos(18.38) + 3, "Base 18.38", 6.5)

    for arm in ARMS:
        x, y = x_pos(arm.retention), y_pos(arm.macro)
        low, high = y_pos(arm.macro - arm.standard_deviation), y_pos(arm.macro + arm.standard_deviation)
        canvas.setStrokeColorRGB(*arm.color)
        canvas.setFillColorRGB(*arm.color)
        canvas.setLineWidth(0.8)
        canvas.line(x, low, x, high)
        canvas.line(x - 2.5, low, x + 2.5, low)
        canvas.line(x - 2.5, high, x + 2.5, high)
        canvas.circle(x, y, 2.8, stroke=1, fill=1)
        label_x = x + 4 if arm.name != "Raw" else x - stringWidth(arm.name, FONT, 6.5) - 4
        draw_text(canvas, label_x, y + 4, arm.name, 6.5)

    draw_centered(canvas, left + width / 2, 16, "Raw stream tokens retained (%)", 7.0)
    canvas.saveState()
    canvas.translate(12, bottom + height / 2)
    canvas.rotate(90)
    draw_centered(canvas, 0, 0, "Primary macro (%)", 7.0)
    canvas.restoreState()
    canvas.setFont(FONT_BOLD, 7.3)
    canvas.drawCentredString(left + width / 2, 176, "(a) Compression versus primary macro")


def draw_right_panel(canvas: Canvas) -> None:
    left, bottom, width, height = 332.0, 34.0, 172.0, 122.0
    x_min, x_max = -2.2, 6.2
    x_pos = lambda value: left + (value - x_min) / (x_max - x_min) * width
    row_step = height / len(BENCHMARKS)

    canvas.setLineWidth(0.35)
    for tick in (-2, 0, 2, 4, 6):
        x = x_pos(float(tick))
        canvas.setStrokeColorRGB(*GRID)
        canvas.line(x, bottom, x, bottom + height)
        draw_centered(canvas, x, bottom - 10, str(tick), 6.5)
    canvas.setStrokeColorRGB(0.25, 0.25, 0.25)
    canvas.setLineWidth(0.7)
    canvas.line(x_pos(0), bottom, x_pos(0), bottom + height)

    zero = x_pos(0)
    for index, name in enumerate(BENCHMARKS):
        y = bottom + height - (index + 0.5) * row_step
        label_width = stringWidth(name, FONT, 6.1)
        draw_text(canvas, left - label_width - 5, y - 2, name, 6.1)
        for offset, value, fill, stroke in (
            (2.6, NORMAL_DELTAS[index], BLUE_LIGHT, BLUE),
            (-3.6, HARD_DELTAS[index], ORANGE_LIGHT, ORANGE),
        ):
            endpoint = x_pos(value)
            canvas.setFillColorRGB(*fill)
            canvas.setStrokeColorRGB(*stroke)
            canvas.rect(min(zero, endpoint), y + offset, abs(endpoint - zero), 3.8, stroke=1, fill=1)

    legend_y = 163.0
    canvas.setFillColorRGB(*BLUE_LIGHT)
    canvas.setStrokeColorRGB(*BLUE)
    canvas.rect(394, legend_y, 9, 5, stroke=1, fill=1)
    draw_text(canvas, 406, legend_y, "Normal", 6.5)
    canvas.setFillColorRGB(*ORANGE_LIGHT)
    canvas.setStrokeColorRGB(*ORANGE)
    canvas.rect(447, legend_y, 9, 5, stroke=1, fill=1)
    draw_text(canvas, 459, legend_y, "Hard", 6.5)
    draw_centered(canvas, left + width / 2, 12, "Mean change from Raw (percentage points)", 7.0)
    canvas.setFont(FONT_BOLD, 7.3)
    canvas.setFillColorRGB(0.08, 0.08, 0.08)
    canvas.drawCentredString(left + width / 2, 176, "(b) Per-benchmark mean change from Raw")


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "figures" / "results_tradeoff.pdf"
    pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, FONT_BOLD_PATH))
    canvas = Canvas(
        str(output),
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        pageCompression=1,
        initialFontName=FONT,
        initialFontSize=7.0,
    )
    draw_left_panel(canvas)
    draw_right_panel(canvas)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    main()
