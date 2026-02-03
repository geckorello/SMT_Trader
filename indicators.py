"""Indicator utilities for SMT signal engine."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def trend_proxy(
    df: pd.DataFrame,
    ma_mid: int = 50,
    ma_long: int = 200,
    slope_lookback: int = 5,
    slope_min: float = 0.0,
    sep_min: float = 0.0,
) -> pd.Series:
    """Trend proxy: MA_mid above MA_long with positive slope and optional separation."""
    ma_mid_s = sma(df["close"], ma_mid)
    ma_long_s = sma(df["close"], ma_long)
    slope_base = ma_long_s.shift(slope_lookback)
    slope_pct = (ma_long_s - slope_base) / slope_base
    sep_ok = ma_mid_s > (ma_long_s * (1.0 + sep_min))
    return sep_ok & (slope_pct > slope_min)


def trend_down_proxy(
    df: pd.DataFrame,
    ma_mid: int = 50,
    ma_long: int = 200,
    slope_lookback: int = 5,
    slope_min: float = 0.0,
    sep_min: float = 0.0,
) -> pd.Series:
    """Downtrend proxy: MA_mid below MA_long with negative slope and optional separation."""
    ma_mid_s = sma(df["close"], ma_mid)
    ma_long_s = sma(df["close"], ma_long)
    slope_base = ma_long_s.shift(slope_lookback)
    slope_pct = (ma_long_s - slope_base) / slope_base
    sep_ok = ma_mid_s < (ma_long_s * (1.0 - sep_min))
    return sep_ok & (slope_pct < -slope_min)


def consecutive_up(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).apply(lambda x: all(x > 0), raw=True).astype(bool)


def consecutive_down(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).apply(lambda x: all(x < 0), raw=True).astype(bool)
