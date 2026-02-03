"""Market data provider interface with yfinance backend for GLD/SLV."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas as pd


@dataclass
class DataProviderConfig:
    interval: str = "1d"
    auto_adjust: bool = True


def fetch_yfinance(symbol: str, start: str | None = None, end: str | None = None, config: Optional[DataProviderConfig] = None) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from exc

    config = config or DataProviderConfig()
    if start or end:
        df = yf.download(symbol, start=start, end=end, interval=config.interval, auto_adjust=config.auto_adjust, progress=False)
    else:
        # Ensure enough history for long MAs
        df = yf.download(symbol, period="max", interval=config.interval, auto_adjust=config.auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        # If multi-index, try to select the requested symbol level
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)
        else:
            df = df.droplevel(-1, axis=1)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    # Drop any duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]
