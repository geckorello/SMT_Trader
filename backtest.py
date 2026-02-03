"""Simple vectorized backtest with walk-forward splits for SMT signal engine."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from rules_engine import generate_signals, SignalConfig
from data_provider import fetch_yfinance
from config_loader import load_config, backtest_config, signal_config, BacktestConfig


def _apply_costs(returns: pd.Series, trades: pd.Series, config: BacktestConfig) -> pd.Series:
    cost = (config.cost_bps + config.slippage_bps) / 10000.0
    return returns - trades * cost


def _calc_metrics(equity: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    if equity.empty:
        return {}
    ann = 252
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(1e-6, len(equity) / ann)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = daily_returns.std() * np.sqrt(ann)
    sharpe = (daily_returns.mean() * ann) / vol if vol > 0 else 0.0
    downside = daily_returns[daily_returns < 0].std() * np.sqrt(ann)
    sortino = (daily_returns.mean() * ann) / downside if downside > 0 else 0.0
    drawdown = (equity / equity.cummax()) - 1.0
    max_dd = drawdown.min()

    trades = daily_returns[daily_returns != 0]
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    win_rate = len(wins) / max(1, len(trades))
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_daily_return": daily_returns.mean(),
        "volatility": vol,
    }


def _as_series(val) -> pd.Series:
    if isinstance(val, pd.DataFrame):
        return val.iloc[:, 0]
    return val


def backtest_signals(
    df: pd.DataFrame,
    instrument: str,
    config: BacktestConfig,
    sig_cfg: SignalConfig | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    signals = generate_signals(df, instrument, config=sig_cfg)
    pos = signals["signal"].map({"BUY": 1, "SELL": 0, "HOLD": np.nan}).astype("float64").ffill().fillna(0.0)
    pos = _as_series(pos)
    close = _as_series(df["close"])
    daily_ret = close.pct_change().fillna(0)
    strategy_ret = pos.shift(1).fillna(0) * daily_ret
    trades = pos.diff().abs().fillna(0)
    net_ret = _apply_costs(strategy_ret, trades, config)

    equity = (1 + net_ret).cumprod() * config.initial_capital
    metrics = _calc_metrics(equity, net_ret)

    out = signals.copy()
    out["position"] = pos
    out["strategy_ret"] = net_ret
    out["equity"] = equity
    return out, metrics


def walk_forward(
    df: pd.DataFrame,
    instrument: str,
    config: BacktestConfig,
    sig_cfg: SignalConfig | None = None,
) -> Dict[str, Dict[str, float]]:
    results = {}
    years = sorted(df.index.year.unique())
    if len(years) < config.train_years + config.walk_forward_years:
        return {"full": backtest_signals(df, instrument, config, sig_cfg)[1]}

    for start in range(0, len(years) - config.train_years, config.walk_forward_years):
        train_years = years[start:start + config.train_years]
        test_years = years[start + config.train_years:start + config.train_years + config.walk_forward_years]
        if not test_years:
            break
        test_df = df[df.index.year.isin(test_years)]
        key = f"{test_years[0]}-{test_years[-1]}"
        _, metrics = backtest_signals(test_df, instrument, config, sig_cfg)
        results[key] = metrics
    return results


def _format_metrics_table(wf: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if not wf:
        return pd.DataFrame()
    df = pd.DataFrame(wf).T
    df.index.name = "period"
    # Format key columns for readability
    pct_cols = ["total_return", "cagr", "max_drawdown", "win_rate", "volatility", "avg_daily_return"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)
    for col in ["sharpe", "sortino", "profit_factor"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtest with walk-forward evaluation.")
    parser.add_argument("--data", help="CSV file with daily OHLCV")
    parser.add_argument("--yahoo", action="store_true", help="Fetch data from yfinance instead of CSV")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--instrument", default="GLD")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--show-table", action="store_true", help="Print walk-forward table to stdout")
    args = parser.parse_args()

    app = load_config(args.config)
    cfg = backtest_config(app)
    sig_cfg = signal_config(app)

    if args.yahoo:
        df = fetch_yfinance(args.instrument, start=args.start, end=args.end)
    else:
        if not args.data:
            raise SystemExit("--data is required unless --yahoo is used")
        df = pd.read_csv(args.data, parse_dates=["date"], index_col="date")

    wf = walk_forward(df, args.instrument, cfg, sig_cfg)
    table = _format_metrics_table(wf)
    if args.show_table:
        print(f"Walk-forward results ({args.instrument}):")
        if table.empty:
            print("No results.")
        else:
            # Console-friendly view
            cols = ["total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "profit_factor"]
            cols = [c for c in cols if c in table.columns]
            print(table[cols].to_string())

    # Save a readable table to reports
    output_dir = app.raw.get("report", {}).get("output_dir", "reports")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().date().isoformat()
    out_csv = out_dir / f"backtest_{args.instrument}_{date}.csv"
    out_md = out_dir / f"backtest_{args.instrument}_{date}.md"
    if not table.empty:
        table.to_csv(out_csv)
        out_md.write_text(
            "\n".join(
                [
                    f"# Walk-Forward Backtest — {args.instrument} ({date})",
                    "",
                    "All returns/volatility are in percent.",
                    "",
                    table.to_markdown(),
                ]
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out_csv}")
        print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
