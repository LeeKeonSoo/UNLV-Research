from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


HEADING_RE: Final = re.compile(r"(?m)^\\(?:part|chapter|section|subsection|subsubsection)\*?\{")


@dataclass(frozen=True, slots=True)
class LatexHeadingUnit:
    unit_id: str
    text: str


def _normalized_lines(text: str) -> str:
    return "\n".join(line for raw in text.splitlines() if (line := " ".join(raw.split())))


def extract_latex_heading_units(text: str, minimum_characters: int) -> tuple[LatexHeadingUnit, ...]:
    """Split a complete LaTeX book at declared structural headings."""
    starts = tuple(match.start() for match in HEADING_RE.finditer(text))
    units = []
    for ordinal, start in enumerate(starts):
        end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(text)
        normalized = _normalized_lines(text[start:end])
        if len(normalized) >= minimum_characters:
            units.append(LatexHeadingUnit(f"heading-{ordinal:06d}", normalized))
    return tuple(units)
