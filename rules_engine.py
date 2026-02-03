"""Rules engine implementing SMT proxies for GLD/SLV."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import indicators as ind
from cycle_engine.detect import detect_cycles, spec_range
from cycle_engine.state import load_manual_cycles, apply_manual_overrides, compute_cycle_state
from cycle_engine.pdf_specs import load_or_build_specs

RULES_PATH = Path("data/rules/rules.json")


@dataclass
class SignalConfig:
    risk_per_trade: float = 0.005  # 0.5% of equity
    atr_window: int = 14
    stop_atr_mult: float = 2.0
    ma_short: int = 20
    ma_mid: int = 50
    ma_long: int = 200
    trend_slope_lookback: int = 5
    trend_slope_min: float = 0.0
    trend_sep_min: float = 0.0
    use_cycles: bool = False
    cycle_specs_path: str = "data/cycle_specs.json"
    manual_cycles_path: str = "data/cycles/manual_cycle_points.csv"
    cycle_reversal_lookback: int = 5
    cycle_reversal_threshold: float = 0.66
    cycle_size_bonus: float = 0.5
    cycle_size_max_mult: float = 1.5
    cycle_size_penalty: float = 0.25


def _load_rules() -> Dict[str, dict]:
    if not RULES_PATH.exists():
        return {}
    data = json.loads(RULES_PATH.read_text())
    return {r["id"]: r for r in data.get("rules", [])}


def _load_cycle_specs(path: str) -> Dict[str, dict] | None:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _rule_meta(rule_id: str, rules: Dict[str, dict]) -> dict:
    base = rules.get(rule_id, {})
    return {
        "id": rule_id,
        "name": base.get("name"),
        "description": base.get("description"),
        "confidence": base.get("confidence"),
        "provenance": base.get("provenance"),
        "proxy": base.get("proxy"),
    }


def _cycle_rule_meta() -> dict:
    return {
        "id": "cycle_low_proxy_v1",
        "name": "Cycle Low Zone (Proxy)",
        "description": "Cycle low zone based on detected cycle day vs expected range and a basic reversal check.",
        "confidence": 0.45,
        "proxy": "Proxy derived from SMT cycle length ranges and price action; not a direct SMT rule.",
        "provenance": {"source": "SMT cycle specs + cycle detection"},
    }


def _cycle_high_rule_meta() -> dict:
    return {
        "id": "cycle_high_proxy_v1",
        "name": "Cycle High Zone (Proxy)",
        "description": "Late-cycle reversal based on cycle day exceeding expected range and bearish reversal.",
        "confidence": 0.45,
        "proxy": "Proxy derived from SMT cycle length ranges and price action; not a direct SMT rule.",
        "provenance": {"source": "SMT cycle specs + cycle detection"},
    }


def _cycle_size_meta() -> dict:
    return {
        "id": "cycle_size_adj_v1",
        "name": "Cycle Size Adjustment (Proxy)",
        "description": "Position sizing adjusted based on cycle proximity.",
        "confidence": 0.4,
        "proxy": "Proxy sizing adjustment using detected cycle proximity.",
        "provenance": {"source": "Cycle detection + SMT ranges"},
    }


def _position_size_pct(close: float, stop_dist: float, risk_per_trade: float) -> float:
    if isinstance(close, pd.Series):
        close = close.iloc[0] if len(close) > 0 else 0
    if close is None or (isinstance(close, float) and pd.isna(close)):
        close = 0
    if stop_dist <= 0 or close <= 0:
        return 0.0
    # Percent of equity = risk_per_trade / (stop_dist / close)
    return min(1.0, risk_per_trade / (stop_dist / close))


def _to_bool(val) -> bool:
    if isinstance(val, pd.Series):
        return bool(val.iloc[0]) if len(val) > 0 else False
    if isinstance(val, (list, tuple, np.ndarray)):
        return bool(val[0]) if len(val) > 0 else False
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return bool(val)


def _scalar(val):
    if isinstance(val, pd.Series):
        return val.iloc[0] if len(val) > 0 else 0
    if isinstance(val, (list, tuple, np.ndarray)):
        return val[0] if len(val) > 0 else 0
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    return val


def _proximity(day: float, rng: tuple[int, int] | None) -> float:
    if rng is None or day != day:
        return 0.0
    lo, hi = rng
    if hi <= lo:
        return 1.0
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return max(0.0, 1.0 - min(1.0, abs(day - mid) / half))


def generate_signals(df: pd.DataFrame, instrument: str, config: SignalConfig | None = None) -> pd.DataFrame:
    """
    Inputs df with columns: open, high, low, close, volume (daily).
    Outputs df with columns: signal, entry, stop, risk_per_trade, position_size_pct, reasons
    """
    if config is None:
        config = SignalConfig()

    rules = _load_rules()

    data = df.copy()
    data["ret"] = data["close"].pct_change()
    data["atr"] = ind.atr(data, window=config.atr_window)
    data["ma20"] = ind.sma(data["close"], config.ma_short)
    data["ma50"] = ind.sma(data["close"], config.ma_mid)
    data["ma200"] = ind.sma(data["close"], config.ma_long)
    data["trend_up"] = ind.trend_proxy(
        data,
        ma_mid=config.ma_mid,
        ma_long=config.ma_long,
        slope_lookback=config.trend_slope_lookback,
        slope_min=config.trend_slope_min,
        sep_min=config.trend_sep_min,
    )
    data["trend_down"] = ind.trend_down_proxy(
        data,
        ma_mid=config.ma_mid,
        ma_long=config.ma_long,
        slope_lookback=config.trend_slope_lookback,
        slope_min=config.trend_slope_min,
        sep_min=config.trend_sep_min,
    ) & data["ma200"].notna()

    # Streaks
    data["up_4"] = ind.consecutive_up(data["ret"], 4)
    data["down_4"] = ind.consecutive_down(data["ret"], 4)

    # One-day reversal after 4-day streak
    data["one_day_down"] = data["ret"] < 0
    data["one_day_up"] = data["ret"] > 0

    # Reversal signals (multi-feature)
    lookback = config.cycle_reversal_lookback
    prev_high = data["high"].shift(1).rolling(lookback).max()
    prev_low = data["low"].shift(1).rolling(lookback).min()
    data["bull_breakout"] = data["close"] > prev_high
    data["bear_breakdown"] = data["close"] < prev_low
    data["ma_cross_up"] = (data["close"] > data["ma20"]) & (data["close"].shift(1) <= data["ma20"].shift(1))
    data["ma_cross_down"] = (data["close"] < data["ma20"]) & (data["close"].shift(1) >= data["ma20"].shift(1))
    data["bull_engulf"] = (
        (data["close"] > data["open"]) &
        (data["close"].shift(1) < data["open"].shift(1)) &
        (data["close"] >= data["open"].shift(1)) &
        (data["open"] <= data["close"].shift(1))
    )
    data["bear_engulf"] = (
        (data["close"] < data["open"]) &
        (data["close"].shift(1) > data["open"].shift(1)) &
        (data["open"] >= data["close"].shift(1)) &
        (data["close"] <= data["open"].shift(1))
    )
    data["bull_rev_score"] = (
        data["bull_breakout"].astype(int)
        + data["ma_cross_up"].astype(int)
        + data["bull_engulf"].astype(int)
    ) / 3.0
    data["bear_rev_score"] = (
        data["bear_breakdown"].astype(int)
        + data["ma_cross_down"].astype(int)
        + data["bear_engulf"].astype(int)
    ) / 3.0
    data["bull_rev_score"] = data["bull_rev_score"].fillna(0.0)
    data["bear_rev_score"] = data["bear_rev_score"].fillna(0.0)

    # Cycle detection (optional)
    cycle_state = None
    dcl_range = (20, 40)
    icl_range = (60, 120)
    if config.use_cycles:
        specs = _load_cycle_specs(config.cycle_specs_path)
        if specs is None:
            specs = load_or_build_specs(out_path=config.cycle_specs_path)
        cycles = detect_cycles(data.reset_index().rename(columns={"index": "date"}), instrument, specs=specs)
        manual = load_manual_cycles(config.manual_cycles_path)
        cycles = apply_manual_overrides(cycles, manual, instrument)
        cycle_state = compute_cycle_state(data.reset_index().rename(columns={"index": "date"}), cycles)
        # Align to data index
        cycle_state = cycle_state.reindex(data.index)
        data["dcl_day"] = cycle_state["dcl_day"]
        data["icl_day"] = cycle_state["icl_day"]
        dcl_range = spec_range(specs, instrument, "daily", "DCL") or dcl_range
        icl_range = spec_range(specs, instrument, "daily", "ICL") or icl_range

    signals = []

    for idx, row in data.iterrows():
        reasons: List[dict] = []
        rule_checks: List[dict] = []
        signal = "HOLD"
        cycle_low_trigger = False
        cycle_high_trigger = False
        dcl_day = float("nan")
        icl_day = float("nan")

        trend_up = _to_bool(row.get("trend_up"))
        trend_down = _to_bool(row.get("trend_down"))
        up_4 = _to_bool(row.get("up_4"))
        down_4 = _to_bool(row.get("down_4"))
        one_day_down = _to_bool(row.get("one_day_down"))
        one_day_up = _to_bool(row.get("one_day_up"))

        # 4-day corollary proxy
        corollary_up = trend_up and up_4 and one_day_down
        corollary_down = trend_down and down_4 and one_day_up
        rule_checks.append(
            {
                "rule": _rule_meta("smt_4day_corollary_v1", rules),
                "triggered": bool(corollary_up or corollary_down),
                "direction": "SELL" if corollary_up else "BUY" if corollary_down else None,
                "note": "trend + 4-day streak then 1-day reversal",
            }
        )
        if corollary_up or corollary_down:
            meta = _rule_meta("smt_4day_corollary_v1", rules)
            reasons.append({"rule": meta, "trigger": True})
            signal = "SELL" if corollary_up else "BUY"

        # 4-day rule proxy (confirmation)
        rule_up = trend_up and down_4
        rule_down = trend_down and up_4
        rule_checks.append(
            {
                "rule": _rule_meta("smt_4day_rule_v1", rules),
                "triggered": bool(rule_up or rule_down),
                "direction": "SELL" if rule_up else "BUY" if rule_down else None,
                "note": "trend + 4-day counter-trend closes",
            }
        )
        if rule_up or rule_down:
            meta = _rule_meta("smt_4day_rule_v1", rules)
            reasons.append({"rule": meta, "trigger": True})
            signal = "SELL" if rule_up else "BUY"

        # MA touch proxy (mean reversion within uptrend)
        close = _scalar(row.get("close"))
        ma20 = _scalar(row.get("ma20"))
        ma50 = _scalar(row.get("ma50"))
        ma_touch = trend_up and (close <= ma20 or close <= ma50)
        rule_checks.append(
            {
                "rule": _rule_meta("smt_ma_touch_proxy_v1", rules),
                "triggered": bool(ma_touch),
                "direction": "BUY" if ma_touch else None,
                "note": "uptrend + close <= MA20/MA50",
            }
        )
        if ma_touch:
            meta = _rule_meta("smt_ma_touch_proxy_v1", rules)
            reasons.append({"rule": meta, "trigger": True})
            if signal == "HOLD":
                signal = "BUY"

        # Cycle low zone proxy (optional)
        if config.use_cycles:
            dcl_day = _scalar(row.get("dcl_day"))
            icl_day = _scalar(row.get("icl_day"))
            in_dcl_zone = dcl_day >= dcl_range[0] and dcl_day <= dcl_range[1]
            in_icl_zone = icl_day >= icl_range[0] and icl_day <= icl_range[1]
            late_dcl = dcl_day >= dcl_range[1] if dcl_day == dcl_day else False
            late_icl = icl_day >= icl_range[1] if icl_day == icl_day else False

            bull_rev_score = _scalar(row.get("bull_rev_score"))
            bear_rev_score = _scalar(row.get("bear_rev_score"))
            bull_rev = bull_rev_score >= config.cycle_reversal_threshold
            bear_rev = bear_rev_score >= config.cycle_reversal_threshold

            cycle_low_trigger = (in_dcl_zone or in_icl_zone) and bull_rev
            cycle_high_trigger = (late_dcl or late_icl) and bear_rev
            rule_checks.append(
                {
                    "rule": _cycle_rule_meta(),
                    "triggered": bool(cycle_low_trigger),
                    "direction": "BUY" if cycle_low_trigger else None,
                    "note": f"dcl_day={dcl_day:.0f} (exp {dcl_range[0]}-{dcl_range[1]}), "
                    f"icl_day={icl_day:.0f} (exp {icl_range[0]}-{icl_range[1]}), "
                    f"bull_rev_score={bull_rev_score:.2f} (thr {config.cycle_reversal_threshold:.2f})",
                }
            )
            rule_checks.append(
                {
                    "rule": _cycle_high_rule_meta(),
                    "triggered": bool(cycle_high_trigger),
                    "direction": "SELL" if cycle_high_trigger else None,
                    "note": f"dcl_day={dcl_day:.0f} (exp {dcl_range[0]}-{dcl_range[1]}), "
                    f"icl_day={icl_day:.0f} (exp {icl_range[0]}-{icl_range[1]}), "
                    f"bear_rev_score={bear_rev_score:.2f} (thr {config.cycle_reversal_threshold:.2f})",
                }
            )

            if cycle_high_trigger:
                meta = _cycle_high_rule_meta()
                reasons.append(
                    {
                        "rule": meta,
                        "trigger": True,
                        "detail": f"Late-cycle reversal; dcl_day={dcl_day:.0f}, icl_day={icl_day:.0f}",
                    }
                )
                signal = "SELL"

            if cycle_low_trigger and signal != "SELL":
                meta = _cycle_rule_meta()
                reasons.append(
                    {
                        "rule": meta,
                        "trigger": True,
                        "detail": f"Cycle low zone + bullish reversal; dcl_day={dcl_day:.0f}, icl_day={icl_day:.0f}",
                    }
                )
                if signal == "HOLD":
                    signal = "BUY"

        atr_val = _scalar(row.get("atr"))
        stop_dist = atr_val * config.stop_atr_mult
        entry = close if signal != "HOLD" else None
        stop = (close - stop_dist) if signal == "BUY" else (close + stop_dist if signal == "SELL" else None)
        pos_pct = _position_size_pct(close, stop_dist, config.risk_per_trade) if signal != "HOLD" else 0.0

        # Cycle-aware position sizing
        size_mult = 1.0
        if config.use_cycles and signal == "BUY":
            proximity = max(_proximity(dcl_day, dcl_range), _proximity(icl_day, icl_range))
            if cycle_low_trigger:
                size_mult = min(config.cycle_size_max_mult, 1.0 + config.cycle_size_bonus * proximity)
            else:
                # Penalize late-cycle buys
                if (dcl_day == dcl_day and dcl_day > dcl_range[1]) or (icl_day == icl_day and icl_day > icl_range[1]):
                    size_mult = max(0.1, 1.0 - config.cycle_size_penalty)
            pos_pct = pos_pct * size_mult
            if size_mult != 1.0:
                reasons.append(
                    {
                        "rule": _cycle_size_meta(),
                        "trigger": True,
                        "detail": f"size_mult={size_mult:.2f} (proximity={proximity:.2f})",
                    }
                )

        signals.append(
            {
                "signal": signal,
                "entry": entry,
                "stop": stop,
                "risk_per_trade": config.risk_per_trade,
                "position_size_pct": pos_pct,
                "reasons": reasons,
                "rule_checks": rule_checks,
                "instrument": instrument,
            }
        )

    out = pd.concat([data, pd.DataFrame(signals, index=data.index)], axis=1)
    return out
