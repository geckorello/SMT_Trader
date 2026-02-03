"""Confidence scoring for detected cycle lows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ScoreComponents:
    pivot_strength: float
    reversal: float
    drawdown: float
    atr_pattern: float
    length_match: float


def _normalize_drawdown(dd: float, target: float = -0.12) -> float:
    # dd is negative; deeper drawdown increases score
    if dd >= 0:
        return 0.0
    return min(1.0, abs(dd) / abs(target))


def length_score(length: int, expected: Tuple[int, int] | None) -> float:
    if expected is None:
        return 0.5
    lo, hi = expected
    if lo <= length <= hi:
        return 1.0
    # Penalize distance outside range
    if length < lo:
        return max(0.0, 1.0 - (lo - length) / lo)
    return max(0.0, 1.0 - (length - hi) / hi)


def score_cycle(
    pivot_strength: float,
    reversal: bool,
    drawdown: float,
    atr_pattern: float,
    length: int,
    expected: Tuple[int, int] | None,
) -> Tuple[float, str, ScoreComponents]:
    length_match = length_score(length, expected)
    comp = ScoreComponents(
        pivot_strength=float(pivot_strength),
        reversal=1.0 if reversal else 0.0,
        drawdown=_normalize_drawdown(drawdown),
        atr_pattern=float(atr_pattern),
        length_match=float(length_match),
    )

    # Weighted confidence
    weights = {
        "pivot_strength": 0.25,
        "reversal": 0.20,
        "drawdown": 0.20,
        "atr_pattern": 0.15,
        "length_match": 0.20,
    }
    score = (
        comp.pivot_strength * weights["pivot_strength"]
        + comp.reversal * weights["reversal"]
        + comp.drawdown * weights["drawdown"]
        + comp.atr_pattern * weights["atr_pattern"]
        + comp.length_match * weights["length_match"]
    )
    score = max(0.0, min(1.0, score))

    exp_str = "none" if expected is None else f"{expected[0]}-{expected[1]}"
    notes = (
        f"pivot={comp.pivot_strength:.2f}, reversal={comp.reversal:.0f}, "
        f"drawdown={comp.drawdown:.2f}, atr={comp.atr_pattern:.2f}, "
        f"length={length} vs expected {exp_str} (match={comp.length_match:.2f})"
    )
    return score, notes, comp
