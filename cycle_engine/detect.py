"""Cycle detection based on price action and SMT-calibrated ranges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import features
from .scoring import score_cycle


@dataclass
class CycleConfig:
    pivot_window: int = 3
    confirm_window: int = 5
    confirm_lookback: int = 5
    dd_lookback: int = 60
    atr_window: int = 14
    dcl_default: Tuple[int, int] = (20, 40)
    icl_default: Tuple[int, int] = (60, 120)


def spec_range(specs: Dict[str, dict] | None, instrument: str, timeframe: str, kind: str) -> Tuple[int, int] | None:
    if not specs:
        return None
    # Accept full payload with {"meta":..., "specs":...}
    if "specs" in specs:
        specs = specs.get("specs", {})
    inst = instrument.upper()
    inst_key = "GOLD" if inst in {"GLD", "XAU", "XAUUSD", "GOLD"} else "SILVER" if inst in {"SLV", "XAG", "XAGUSD", "SILVER"} else inst
    data = specs.get(inst_key, {})
    if not data:
        return None
    # daily specs map to DCL; weekly specs map to ICL (converted to trading days)
    if timeframe == "daily" and kind == "DCL":
        if "daily" in data:
            return data["daily"]["min"], data["daily"]["max"]
    if timeframe == "daily" and kind == "ICL":
        if "weekly" in data:
            lo = int(data["weekly"]["min"] * 5)
            hi = int(data["weekly"]["max"] * 5)
            return lo, hi
    if timeframe == "weekly" and kind == "DCL":
        if "weekly" in data:
            return data["weekly"]["min"], data["weekly"]["max"]
    return None


def _select_cycle_lows(
    candidates: List[int],
    lows: pd.Series,
    min_len: int,
    max_len: int,
) -> List[int]:
    if not candidates:
        return []
    accepted = []
    last = candidates[0]
    best = last
    accepted.append(last)
    for idx in candidates[1:]:
        length = idx - last
        # update best candidate between lows
        if lows.iloc[idx] <= lows.iloc[best]:
            best = idx
        if length < min_len:
            continue
        if length <= max_len:
            accepted.append(best)
            last = best
            best = idx
        elif length > max_len:
            accepted.append(best)
            last = best
            best = idx
    return accepted


def detect_cycles(df: pd.DataFrame, instrument: str, timeframe: str = "daily", specs: Dict[str, dict] | None = None) -> pd.DataFrame:
    """
    Detect cycle lows (DCL/HCL/ICL) and return a cycle table.
    Output columns: ['date','instrument','cycle_type','cycle_day','confidence','notes']
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "instrument", "cycle_type", "cycle_day", "confidence", "notes"])

    data = df.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    data = data.sort_index()

    cfg = CycleConfig()
    low = data["low"]
    close = data["close"]

    piv = features.pivot_lows(low, window=cfg.pivot_window)
    piv_strength = features.pivot_strength(low, window=cfg.pivot_window)
    dd = features.drawdown_from_high(close, lookback=cfg.dd_lookback)
    atr = features.atr(data, window=cfg.atr_window)

    candidates = []
    for i in range(len(data)):
        if not piv.iloc[i]:
            continue
        if not features.confirmation_after_low(close, i, lookback=cfg.confirm_lookback, window=cfg.confirm_window):
            continue
        candidates.append(i)

    dcl_range = spec_range(specs, instrument, timeframe, "DCL") or cfg.dcl_default
    icl_range = spec_range(specs, instrument, timeframe, "ICL") or cfg.icl_default

    dcl_indices = _select_cycle_lows(candidates, low, dcl_range[0], dcl_range[1])

    # ICL: select deeper lows with longer spacing
    icl_indices = []
    last_icl = None
    for idx in dcl_indices:
        if last_icl is None:
            last_icl = idx
            icl_indices.append(idx)
            continue
        length = idx - last_icl
        if length >= icl_range[0]:
            # require deeper drawdown
            if dd.iloc[idx] <= -0.08:
                icl_indices.append(idx)
                last_icl = idx

    # HCL: choose DCL roughly midpoint between ICLs
    hcl_indices = []
    for i in range(1, len(icl_indices)):
        start = icl_indices[i - 1]
        end = icl_indices[i]
        mid = (start + end) // 2
        between = [d for d in dcl_indices if start < d < end]
        if not between:
            continue
        # pick closest to midpoint
        hcl = min(between, key=lambda x: abs(x - mid))
        hcl_indices.append(hcl)

    rows = []

    def add_rows(indices: List[int], ctype: str, expected: Tuple[int, int] | None):
        last = None
        for idx in indices:
            if last is None:
                cycle_len = 1
            else:
                cycle_len = idx - last + 1
            score, notes, _ = score_cycle(
                pivot_strength=piv_strength.iloc[idx],
                reversal=True,
                drawdown=dd.iloc[idx],
                atr_pattern=features.atr_pattern(atr, idx),
                length=cycle_len,
                expected=expected,
            )
            rows.append(
                {
                    "date": data.index[idx],
                    "instrument": instrument,
                    "cycle_type": ctype,
                    "cycle_day": int(cycle_len),
                    "confidence": float(score),
                    "notes": notes,
                }
            )
            last = idx

    add_rows(dcl_indices, "DCL", dcl_range)
    add_rows(hcl_indices, "HCL", None)
    add_rows(icl_indices, "ICL", icl_range)

    out = pd.DataFrame(rows)
    return out.sort_values(["date", "cycle_type"]).reset_index(drop=True)
