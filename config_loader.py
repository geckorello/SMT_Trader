"""Load YAML config and map to engine/backtest settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json

from rules_engine import SignalConfig


@dataclass
class BacktestConfig:
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 100000.0
    walk_forward_years: int = 3
    train_years: int = 5


@dataclass
class AppConfig:
    raw: Dict[str, Any]


def load_config(path: str = "config.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to read config.yaml. Install with: pip install pyyaml") from exc
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(raw=raw)


def signal_config(app: AppConfig) -> SignalConfig:
    risk = app.raw.get("risk", {})
    trend = app.raw.get("trend", {})
    cycles = app.raw.get("cycles", {}) or {}
    return SignalConfig(
        risk_per_trade=risk.get("risk_per_trade", 0.005),
        atr_window=risk.get("atr_window", 14),
        stop_atr_mult=risk.get("stop_atr_mult", 2.0),
        ma_short=trend.get("ma_short", 20),
        ma_mid=trend.get("ma_mid", 50),
        ma_long=trend.get("ma_long", 200),
        trend_slope_lookback=trend.get("slope_lookback", 5),
        trend_slope_min=trend.get("slope_min", 0.0),
        trend_sep_min=trend.get("sep_min", 0.0),
        use_cycles=cycles.get("enable_detection", False),
        cycle_specs_path=cycles.get("specs_path", "data/cycle_specs.json"),
        manual_cycles_path=cycles.get("manual_points_path", "data/cycles/manual_cycle_points.csv"),
        cycle_reversal_lookback=cycles.get("cycle_reversal_lookback", 5),
        cycle_reversal_threshold=cycles.get("cycle_reversal_threshold", 0.66),
        cycle_size_bonus=cycles.get("cycle_size_bonus", 0.5),
        cycle_size_max_mult=cycles.get("cycle_size_max_mult", 1.5),
        cycle_size_penalty=cycles.get("cycle_size_penalty", 0.25),
    )


def backtest_config(app: AppConfig) -> BacktestConfig:
    costs = app.raw.get("costs", {})
    return BacktestConfig(
        cost_bps=costs.get("cost_bps", 10.0),
        slippage_bps=costs.get("slippage_bps", 5.0),
    )
