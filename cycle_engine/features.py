"""Feature extraction for cycle detection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - close).abs(), (low - close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def pivot_lows(low: pd.Series, window: int = 3) -> pd.Series:
    """Return boolean Series of pivot lows using a symmetric window."""
    w = int(window)
    if w < 1:
        w = 1
    rolling_min = low.rolling(2 * w + 1, center=True).min()
    piv = (low == rolling_min)
    return piv.fillna(False)


def pivot_strength(low: pd.Series, window: int = 3) -> pd.Series:
    """Normalized depth of pivot low vs local max."""
    w = int(window)
    local_max = low.rolling(2 * w + 1, center=True).max()
    strength = (local_max - low) / local_max.replace(0, np.nan)
    return strength.fillna(0.0)


def drawdown_from_high(close: pd.Series, lookback: int = 60) -> pd.Series:
    rolling_high = close.rolling(lookback, min_periods=lookback).max()
    dd = (close - rolling_high) / rolling_high.replace(0, np.nan)
    return dd.fillna(0.0)


def confirmation_after_low(close: pd.Series, idx: int, lookback: int = 5, window: int = 5) -> bool:
    """Check if close breaks above prior N-day high within confirmation window."""
    if idx <= lookback:
        return False
    prior_high = close.iloc[idx - lookback: idx].max()
    end = min(len(close), idx + window + 1)
    return (close.iloc[idx + 1: end] > prior_high).any()


def atr_pattern(atr_series: pd.Series, idx: int, lookback: int = 3, forward: int = 3) -> float:
    """Score ATR expansion into low then contraction after low."""
    if idx < lookback or idx + forward >= len(atr_series):
        return 0.0
    before = atr_series.iloc[idx - lookback: idx + 1]
    after = atr_series.iloc[idx: idx + forward + 1]
    if before.isna().any() or after.isna().any():
        return 0.0
    expand = before.iloc[-1] > before.iloc[0]
    contract = after.iloc[-1] < after.iloc[0]
    return 1.0 if (expand and contract) else 0.0
