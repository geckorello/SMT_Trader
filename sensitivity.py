"""Parameter sensitivity sweep for SMT signals (educational only)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import backtest_signals, BacktestConfig
from config_loader import load_config, backtest_config, signal_config
from data_provider import fetch_yfinance


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parameter sensitivity sweep.")
    parser.add_argument("--instrument", default="GLD")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    app = load_config(args.config)
    bt_cfg = backtest_config(app)
    sig_cfg = signal_config(app)

    df = fetch_yfinance(args.instrument, start=args.start, end=args.end)

    ma_short_grid = [10, 20, 30]
    ma_mid_grid = [40, 50, 60]
    ma_long_grid = [150, 200, 250]
    stop_mult_grid = [1.5, 2.0, 2.5]

    rows = []
    for ma_s in ma_short_grid:
        for ma_m in ma_mid_grid:
            for ma_l in ma_long_grid:
                if not (ma_s < ma_m < ma_l):
                    continue
                for stop_mult in stop_mult_grid:
                    cfg = replace(
                        sig_cfg,
                        ma_short=ma_s,
                        ma_mid=ma_m,
                        ma_long=ma_l,
                        stop_atr_mult=stop_mult,
                    )
                    _, metrics = backtest_signals(df, args.instrument, bt_cfg, cfg)
                    rows.append(
                        {
                            "ma_short": ma_s,
                            "ma_mid": ma_m,
                            "ma_long": ma_l,
                            "stop_atr_mult": stop_mult,
                            **metrics,
                        }
                    )

    out_dir = Path(app.raw.get("report", {}).get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().date()

    out_csv = out_dir / f"sensitivity_{args.instrument}_{date}.csv"
    pd.DataFrame(rows).sort_values("cagr", ascending=False).to_csv(out_csv, index=False)

    out_md = out_dir / f"sensitivity_{args.instrument}_{date}.md"
    top = pd.DataFrame(rows).sort_values("cagr", ascending=False).head(10)

    lines = [
        f"# Sensitivity Sweep — {args.instrument} ({date})",
        "",
        "Educational analysis only. This sweep is for robustness checks, not optimization.",
        "",
        f"Rows: {len(rows)}",
        "",
        "Top 10 by CAGR:",
        "",
    ]
    lines.append(top.to_markdown(index=False))
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
