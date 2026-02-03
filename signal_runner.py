"""Generate daily signal JSON for configured instruments."""
from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path

from data_provider import fetch_yfinance
from rules_engine import generate_signals
from config_loader import load_config, signal_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily SMT signals JSON.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", help="YYYY-MM-DD (defaults to today)")
    args = parser.parse_args()

    app = load_config(args.config)
    cfg = signal_config(app)

    instruments = app.raw.get("instruments", ["GLD", "SLV"])
    output_dir = Path(app.raw.get("signals", {}).get("output_dir", "reports"))
    output_csv = app.raw.get("signals", {}).get("output_csv", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()

    payload = {
        "date": str(report_date),
        "signals": [],
        "notes": "Educational signal generation only. Not financial advice.",
    }

    for inst in instruments:
        df = fetch_yfinance(inst)
        signals = generate_signals(df, inst, config=cfg)
        last = signals.iloc[-1]
        confidence = 0.0
        if isinstance(last.get("reasons"), list) and last["reasons"]:
            vals = []
            for r in last["reasons"]:
                rule = r.get("rule", {})
                conf = rule.get("confidence")
                if isinstance(conf, (int, float)):
                    vals.append(float(conf))
            if vals:
                confidence = sum(vals) / len(vals)
        payload["signals"].append(
            {
                "instrument": inst,
                "signal": last["signal"],
                "entry": last["entry"],
                "stop": last["stop"],
                "risk_per_trade": last["risk_per_trade"],
                "position_size_pct": last["position_size_pct"],
                "reasons": last["reasons"],
                "rule_checks": last.get("rule_checks", []),
                "confidence_score": confidence,
                "dcl_day": last.get("dcl_day"),
                "icl_day": last.get("icl_day"),
            }
        )

    out_path = output_dir / f"{report_date}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if output_csv:
        csv_path = output_dir / f"{report_date}_signals.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "date",
                    "instrument",
                    "signal",
                    "entry",
                    "stop",
                    "risk_per_trade",
                    "position_size_pct",
                    "reasons",
                ]
            )
            for s in payload["signals"]:
                reasons = "; ".join(
                    [
                        (r.get("rule", {}).get("name") or r.get("rule", {}).get("id") or "rule")
                        for r in s.get("reasons", [])
                    ]
                )
                writer.writerow(
                    [
                        payload["date"],
                        s["instrument"],
                        s["signal"],
                        s["entry"],
                        s["stop"],
                        s["risk_per_trade"],
                        s["position_size_pct"],
                        reasons,
                    ]
                )
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
