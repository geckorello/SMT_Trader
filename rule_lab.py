"""Rule lab: evaluate single rules and rule combinations via backtesting."""
from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import load_config, backtest_config, signal_config, BacktestConfig
from data_provider import fetch_yfinance
from rules_engine import generate_signals


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


def _signal_to_position(signal: pd.Series) -> pd.Series:
    pos = signal.map({"BUY": 1, "SELL": 0, "HOLD": np.nan}).astype("float64").ffill().fillna(0.0)
    return pos


def backtest_from_signal(close: pd.Series, signal: pd.Series, config: BacktestConfig) -> Dict[str, float]:
    pos = _signal_to_position(signal)
    daily_ret = close.pct_change().fillna(0)
    strategy_ret = pos.shift(1).fillna(0) * daily_ret
    trades = pos.diff().abs().fillna(0)
    net_ret = _apply_costs(strategy_ret, trades, config)
    equity = (1 + net_ret).cumprod() * config.initial_capital
    return _calc_metrics(equity, net_ret)


def walk_forward_periods(index: pd.DatetimeIndex, train_years: int, test_years: int) -> List[tuple]:
    years = sorted(index.year.unique())
    periods = []
    if len(years) < train_years + test_years:
        return [("full", index)]
    for start in range(0, len(years) - train_years, test_years):
        test = years[start + train_years:start + train_years + test_years]
        if not test:
            break
        mask = index.year.isin(test)
        periods.append((f"{test[0]}-{test[-1]}", index[mask]))
    return periods


def extract_rule_signals(signals_df: pd.DataFrame) -> Dict[str, pd.Series]:
    rule_ids = set()
    for checks in signals_df["rule_checks"].values:
        for chk in checks:
            rule = chk.get("rule", {})
            rid = rule.get("id")
            if rid:
                rule_ids.add(rid)

    rule_signals = {rid: pd.Series(["HOLD"] * len(signals_df), index=signals_df.index) for rid in rule_ids}

    for idx, checks in signals_df["rule_checks"].items():
        for chk in checks:
            rule = chk.get("rule", {})
            rid = rule.get("id")
            if not rid:
                continue
            if not chk.get("triggered"):
                continue
            direction = chk.get("direction")
            if direction not in {"BUY", "SELL"}:
                continue
            # Priority: SELL > BUY
            cur = rule_signals[rid].loc[idx]
            if cur == "SELL":
                continue
            if direction == "SELL":
                rule_signals[rid].loc[idx] = "SELL"
            elif direction == "BUY" and cur != "SELL":
                rule_signals[rid].loc[idx] = "BUY"

    return rule_signals


def combine_signals(signals: List[pd.Series]) -> pd.Series:
    idx = signals[0].index
    out = pd.Series(["HOLD"] * len(idx), index=idx)
    for i in idx:
        vals = [s.loc[i] for s in signals]
        if "SELL" in vals:
            out.loc[i] = "SELL"
        elif "BUY" in vals:
            out.loc[i] = "BUY"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule lab: evaluate single rules and combinations.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="CSV with OHLCV")
    parser.add_argument("--yahoo", action="store_true")
    parser.add_argument("--instrument", default="GLD")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--max-combo", type=int, default=3)
    args = parser.parse_args()

    app = load_config(args.config)
    bt_cfg = backtest_config(app)
    sig_cfg = signal_config(app)

    if args.yahoo:
        df = fetch_yfinance(args.instrument, start=args.start, end=args.end)
    else:
        if not args.data:
            raise SystemExit("--data is required unless --yahoo is used")
        df = pd.read_csv(args.data, parse_dates=["date"], index_col="date")

    signals_df = generate_signals(df, args.instrument, config=sig_cfg)
    rule_signals = extract_rule_signals(signals_df)

    rule_ids = sorted(rule_signals.keys())
    combos = []
    for k in range(1, min(args.max_combo, len(rule_ids)) + 1):
        combos.extend(list(itertools.combinations(rule_ids, k)))

    index = signals_df.index
    periods = walk_forward_periods(index, bt_cfg.train_years, bt_cfg.walk_forward_years)

    rows = []
    close = df["close"] if "close" in df.columns else df["Close"]
    data_start = close.index.min()
    data_end = close.index.max()

    for combo in combos:
        if len(combo) == 1:
            signal = rule_signals[combo[0]]
        else:
            signal = combine_signals([rule_signals[r] for r in combo])

        full_metrics = backtest_from_signal(close, signal, bt_cfg)

        wf_metrics = []
        for period, idx in periods:
            signal_p = signal.loc[idx]
            close_p = close.loc[idx]
            m = backtest_from_signal(close_p, signal_p, bt_cfg)
            m["period"] = period
            wf_metrics.append(m)

        wf_df = pd.DataFrame(wf_metrics)
        row = {
            "rules": "+".join(combo),
            "rule_count": len(combo),
            "full_total_return": full_metrics.get("total_return", 0),
            "full_sharpe": full_metrics.get("sharpe", 0),
            "full_cagr": full_metrics.get("cagr", 0),
            "full_mdd": full_metrics.get("max_drawdown", 0),
            "wf_total_return_mean": wf_df["total_return"].mean() if not wf_df.empty else 0,
            "wf_sharpe_mean": wf_df["sharpe"].mean() if not wf_df.empty else 0,
            "wf_cagr_mean": wf_df["cagr"].mean() if not wf_df.empty else 0,
            "wf_mdd_mean": wf_df["max_drawdown"].mean() if not wf_df.empty else 0,
        }
        rows.append(row)

    out_dir = Path(app.raw.get("report", {}).get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().date().isoformat()
    rule_lab_cfg = app.raw.get("rule_lab", {}) or {}
    chart_top_n = int(rule_lab_cfg.get("chart_top_n", 10))
    chart_lookback_years = rule_lab_cfg.get("chart_lookback_years", 5)
    if chart_lookback_years is not None:
        chart_lookback_years = float(chart_lookback_years)

    result = pd.DataFrame(rows)
    result = result.sort_values(["wf_sharpe_mean", "wf_cagr_mean"], ascending=False)

    out_csv = out_dir / f"rule_lab_{args.instrument}_{date}.csv"
    result.to_csv(out_csv, index=False)

    # HTML report
    top = result.head(50)
    html = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:24px auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;text-align:right}th{text-align:left;background:#f5f5f5}</style>",
        "</head><body>",
        f"<h1>Rule Lab — {args.instrument} ({date})</h1>",
        f"<p>Dataset: {data_start.date()} to {data_end.date()} ({len(close)} bars)</p>",
        "<p>Ranked by walk-forward Sharpe, then CAGR.</p>",
        "<table>",
        "<tr><th>Rules</th><th>#</th><th>Full Return</th><th>Full CAGR</th><th>Full Sharpe</th><th>Full MDD</th><th>WF Return</th><th>WF Sharpe</th><th>WF CAGR</th><th>WF MDD</th></tr>",
    ]
    for row_idx, r in top.iterrows():
        anchor = f"rule-{row_idx}"
        row_html = (
            f"<tr><td style='text-align:left'><a href='#{anchor}'>{r['rules']}</a></td>"
            f"<td>{int(r['rule_count'])}</td>"
            f"<td>{r['full_total_return']:.2%}</td>"
            f"<td>{r['full_cagr']:.2%}</td>"
            f"<td>{r['full_sharpe']:.2f}</td>"
            f"<td>{r['full_mdd']:.2%}</td>"
            f"<td>{r['wf_total_return_mean']:.2%}</td>"
            f"<td>{r['wf_sharpe_mean']:.2f}</td>"
            f"<td>{r['wf_cagr_mean']:.2%}</td>"
            f"<td>{r['wf_mdd_mean']:.2%}</td></tr>"
        )
        html.append(row_html)
    html.append("</table>")
    html.append("<h2>Signal Charts</h2>")
    html.append("<p>Charts show price with BUY/SELL markers for each rule set.</p>")

    def _build_signal_for_rules(rules_str: str) -> pd.Series:
        parts = rules_str.split("+") if rules_str else []
        if len(parts) == 1:
            return rule_signals[parts[0]]
        return combine_signals([rule_signals[r] for r in parts])

    def _plot_signal_chart(rules_str: str, rank: int) -> str:
        signal = _build_signal_for_rules(rules_str)
        plot_close = close
        if chart_lookback_years is not None:
            cutoff = close.index.max() - pd.DateOffset(years=chart_lookback_years)
            plot_close = close.loc[close.index >= cutoff]
            signal_plot = signal.loc[plot_close.index]
        else:
            signal_plot = signal

        buys = signal_plot == "BUY"
        sells = signal_plot == "SELL"

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(plot_close.index, plot_close.values, label="Close", color="#1f2937")
        if buys.any():
            ax.scatter(plot_close.index[buys], plot_close.values[buys], marker="^", color="#16a34a", s=30, label="BUY")
        if sells.any():
            ax.scatter(plot_close.index[sells], plot_close.values[sells], marker="v", color="#dc2626", s=30, label="SELL")
        ax.set_title(f"{args.instrument} — {rules_str}")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", frameon=False)

        img_name = f"rule_lab_{args.instrument}_{date}_{rank:02d}.png"
        img_path = out_dir / img_name
        fig.tight_layout()
        fig.savefig(img_path, dpi=140)
        plt.close(fig)
        return img_name

    for idx, r in top.head(chart_top_n).iterrows():
        anchor = f"rule-{idx}"
        img_name = _plot_signal_chart(r["rules"], rank=idx + 1)
        html.append(f"<h3 id='{anchor}'>{r['rules']}</h3>")
        html.append(f"<p>Full Return: {r['full_total_return']:.2%} | Full CAGR: {r['full_cagr']:.2%} | WF Sharpe: {r['wf_sharpe_mean']:.2f}</p>")
        html.append(f"<img src='{img_name}' style='width:100%;max-width:980px' />")
    html.append("<h2>Glossary</h2>")
    html.append("<ul>")
    html.append("<li><strong>Return</strong>: total % return over the period.</li>")
    html.append("<li><strong>CAGR</strong>: compound annual growth rate.</li>")
    html.append("<li><strong>Sharpe</strong>: risk-adjusted return (higher is better).</li>")
    html.append("<li><strong>MDD</strong>: maximum drawdown (worst peak-to-trough % loss).</li>")
    html.append("<li><strong>WF</strong>: walk-forward (out-of-sample) averages.</li>")
    html.append("</ul>")
    html.append("</body></html>")
    out_html = out_dir / f"rule_lab_{args.instrument}_{date}.html"
    out_html.write_text("\n".join(html), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
