"""Cycle state computation and manual overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_manual_cycles(path: str = "data/cycles/manual_cycle_points.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["date", "instrument", "cycle_type", "notes"])
    try:
        df = pd.read_csv(p, parse_dates=["date"], comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["date", "instrument", "cycle_type", "notes"])
    for col in ["instrument", "cycle_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
    return df


def apply_manual_overrides(cycles: pd.DataFrame, manual: pd.DataFrame, instrument: str) -> pd.DataFrame:
    if manual.empty:
        return cycles
    inst = instrument.upper()
    m = manual[manual["instrument"] == inst].copy()
    if m.empty:
        return cycles

    # Coerce columns
    m = m[["date", "instrument", "cycle_type"]].copy()
    cycles = cycles.copy()
    cycles["instrument"] = cycles["instrument"].astype(str).str.upper()
    cycles["cycle_type"] = cycles["cycle_type"].astype(str).str.upper()

    # Remove duplicates and append manual
    key_cols = ["date", "instrument", "cycle_type"]
    cycles = cycles[~cycles.set_index(key_cols).index.isin(m.set_index(key_cols).index)]
    out = pd.concat([cycles, m], ignore_index=True).sort_values(["date", "cycle_type"]).reset_index(drop=True)
    return out


def _cycle_day_series(index: pd.DatetimeIndex, low_dates: list[pd.Timestamp]) -> pd.Series:
    low_set = set(low_dates)
    days = []
    last_idx = None
    for i, dt in enumerate(index):
        if dt in low_set:
            last_idx = i
        if last_idx is None:
            days.append(float("nan"))
        else:
            days.append(i - last_idx + 1)
    return pd.Series(days, index=index)


def compute_cycle_state(df: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Return cycle_day series for DCL/HCL/ICL based on detected lows."""
    data = df.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    data = data.sort_index()

    state = pd.DataFrame(index=data.index)
    for ctype in ["DCL", "HCL", "ICL"]:
        lows = cycles[cycles["cycle_type"].str.upper() == ctype]
        low_dates = list(pd.to_datetime(lows["date"]))
        state[f"{ctype.lower()}_day"] = _cycle_day_series(data.index, low_dates)
    return state
