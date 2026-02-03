"""Daily report generator for SMT signals (GLD/SLV)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from config_loader import load_config, signal_config
from data_provider import fetch_yfinance
from rules_engine import generate_signals
from cycle_engine.detect import detect_cycles, spec_range
from cycle_engine.state import load_manual_cycles, apply_manual_overrides
import indicators as ind


def _format_reasons(reasons: List[dict]) -> List[str]:
    lines = []
    for r in reasons:
        rule = r.get("rule", {})
        name = rule.get("name") or rule.get("id")
        confidence = rule.get("confidence")
        proxy = rule.get("proxy")
        prov = rule.get("provenance", {})
        src = prov.get("origin") or prov.get("title")
        detail = r.get("detail")
        msg = f"{name} (conf {confidence})"
        if proxy:
            msg += " [proxy]"
        if src:
            msg += f" | source: {src}"
        if detail:
            msg += f" | {detail}"
        lines.append(msg)
    return lines


def _aggregate_confidence(reasons: List[dict]) -> float:
    if not reasons:
        return 0.0
    vals = []
    for r in reasons:
        rule = r.get("rule", {})
        conf = rule.get("confidence")
        if isinstance(conf, (int, float)):
            vals.append(float(conf))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _regime_summary(df: pd.DataFrame, signals: pd.DataFrame) -> dict:
    last = signals.iloc[-1]
    trend_up = bool(last.get("trend_up"))
    trend = "uptrend" if trend_up else "downtrend/flat"

    # Volatility bucket (20d)
    vol = df["close"].pct_change().rolling(20).std().iloc[-1]
    vol_desc = "low" if vol < 0.012 else "medium" if vol < 0.02 else "high"

    # Trend reasoning from MA50/MA200 + MA200 slope
    ma50 = ind.sma(df["close"], 50).iloc[-1]
    ma200 = ind.sma(df["close"], 200).iloc[-1]
    ma200_slope = ind.sma(df["close"], 200).diff(5).iloc[-1]

    if pd.isna(ma50) or pd.isna(ma200) or pd.isna(ma200_slope):
        reason = "Insufficient data for MA200 trend filter."
    else:
        cond1 = ma50 > ma200
        cond2 = ma200_slope > 0
        reason = (
            f"MA50 ({ma50:.2f}) {'>' if cond1 else '<='} MA200 ({ma200:.2f}) "
            f"and MA200 slope(5d) {ma200_slope:.2f} "
            f"({'positive' if cond2 else 'non-positive'})."
        )

    return {"trend": trend, "volatility": vol_desc, "reason": reason}


def _signal_block(instrument: str, row: pd.Series) -> dict:
    reasons = _format_reasons(row.get("reasons", []))
    conf = _aggregate_confidence(row.get("reasons", []))
    return {
        "instrument": instrument,
        "signal": row["signal"],
        "entry": row.get("entry"),
        "stop": row.get("stop"),
        "risk_per_trade": row.get("risk_per_trade"),
        "position_size_pct": row.get("position_size_pct"),
        "reasons": reasons,
        "rule_checks": row.get("rule_checks", []),
        "confidence_score": conf,
        "chart_paths": [],
        "regime_reason": None,
        "cycle_note": None,
        "cycle_state": None,
        "cycle_markers": None,
        "cycle_flags": {},
    }


def _load_cycle_notes(path: Path, instrument: str) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    notes = data.get("notes", [])
    best = None
    for n in notes:
        if n.get("instrument") != instrument:
            continue
        if best is None or n.get("confidence", 0) > best.get("confidence", 0):
            best = n
    if not best:
        return None
    return best.get("label")


def _load_cycle_specs(path: str = "data/cycle_specs.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def _load_cycle_notes_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("notes", [])


def _cycle_summary_lines(blocks: list[dict]) -> list[str]:
    lines = []
    for b in blocks:
        if b.get("cycle_state"):
            flags = b.get("cycle_flags", {})
            dcl_flag = "yes" if flags.get("dcl_in_range") else "no"
            icl_flag = "yes" if flags.get("icl_in_range") else "no"
            lines.append(f"- {b['instrument']}: {b.get('cycle_state')} | DCL in range: {dcl_flag} | ICL in range: {icl_flag}")
    return lines


def _ocr_notes_lines(notes: list[dict], instruments: list[str] | None = None) -> list[str]:
    lines = []
    allowed = {i.upper() for i in instruments} if instruments else None
    filtered = []
    for n in notes:
        inst = (n.get("instrument") or "UNKNOWN").upper()
        if allowed and inst not in allowed:
            continue
        filtered.append(n)
    for n in filtered[:30]:
        inst = n.get("instrument") or "UNKNOWN"
        label = n.get("label") or ""
        text = n.get("text") or ""
        source = n.get("source") or ""
        conf = n.get("confidence", 0)
        tag = "HIGH" if conf >= 0.6 else "MED" if conf >= 0.4 else "LOW"
        lines.append(f"- [{tag}] {inst}: {label} | {text} | {source}")
    if len(filtered) > 30:
        lines.append(f"- ... plus {len(filtered) - 30} more")
    return lines


def _conf_color(conf: float, thresholds: dict, colors: dict) -> str:
    high = thresholds.get("high", 0.7)
    med = thresholds.get("med", 0.4)
    if conf >= high:
        return colors.get("high", "#2ca02c")
    if conf >= med:
        return colors.get("med", "#ff7f0e")
    return colors.get("low", "#d62728")


def _build_cycle_markers(
    cycles: pd.DataFrame,
    limit: int,
    thresholds: dict,
    colors: dict,
    type_colors: dict | None = None,
) -> list[dict]:
    if cycles.empty:
        return []
    markers = []
    type_marker = {"DCL": "v", "HCL": "o", "ICL": "^"}
    base_size = {"DCL": 70, "HCL": 10, "ICL": 140}
    for ctype, marker in type_marker.items():
        subset = cycles[cycles["cycle_type"].str.upper() == ctype].sort_values("date")
        subset = subset.tail(limit)
        for _, row in subset.iterrows():
            conf = float(row.get("confidence", 0.0))
            size = base_size.get(ctype, 40) * (0.6 + 0.8 * conf)
            color = (type_colors or {}).get(ctype, "#1f77b4")
            markers.append(
                {
                    "label": ctype,
                    "date": pd.to_datetime(row["date"]),
                    "marker": marker,
                    "color": color,
                    "conf": conf,
                    "size": size,
                }
            )
    return markers


def _narrative(block: dict) -> str:
    signal = block.get("signal", "HOLD")
    reasons = block.get("reasons", [])
    checks = block.get("rule_checks", [])
    regime = block.get("regime")

    triggered = []
    proxies = False
    for r in reasons:
        name = r.split(" (conf")[0] if isinstance(r, str) else None
        if name:
            triggered.append(name)
        if isinstance(r, str) and "[proxy]" in r:
            proxies = True

    if signal == "HOLD":
        if checks:
            parts = []
            for chk in checks:
                rule = chk.get("rule", {})
                name = rule.get("name") or rule.get("id") or "rule"
                trig = "yes" if chk.get("triggered") else "no"
                parts.append(f"{name}: {trig}")
            detail = "; ".join(parts)
            base = f"Holding because no SMT rule triggered a BUY/SELL signal ({detail})."
        else:
            base = "Holding because no SMT rule triggered a BUY/SELL signal."
    else:
        if triggered:
            base = f"{signal} because " + ", ".join(triggered) + " triggered."
        else:
            base = f"{signal} because SMT rule conditions were met."
    if proxies:
        base += " Note: one or more triggered rules are proxy interpretations of SMT language."
    if regime:
        base += f" Regime: {regime}."
    if block.get("cycle_state"):
        base += f" Cycle state: {block.get('cycle_state')}."
    if block.get("cycle_note"):
        base += f" OCR cycle note: {block.get('cycle_note')}."
    return base


def _render_markdown(report_date: str, blocks: list[dict], cycle_summary: list[str] | None = None, ocr_notes: list[str] | None = None) -> str:
    lines = [
        f"# Daily Signals — {report_date} (NY Close)",
        "",
        "Educational signal generation only. Not financial advice.",
        "Signals are generated from SMT-derived rules with explicit proxies where SMT language is discretionary.",
        "",
    ]
    if cycle_summary:
        lines.append("## Cycle Summary")
        lines.extend(cycle_summary)
        lines.append("")
    if ocr_notes:
        lines.append("## Cycle OCR Notes")
        lines.extend(ocr_notes)
        lines.append("")
    for b in blocks:
        lines.append(f"## {b['instrument']}")
        lines.append(f"Signal: {b['signal']}")
        lines.append(f"Signal confidence: {b['confidence_score']:.2f}")
        lines.append(f"Regime: {b.get('regime')}")
        if b.get("regime_reason"):
            lines.append(f"Regime reason: {b.get('regime_reason')}")
        if b.get("cycle_note"):
            lines.append(f"Cycle note (OCR): {b.get('cycle_note')}")
        if b.get("cycle_state"):
            lines.append(f"Cycle state: {b.get('cycle_state')}")
        lines.append(f"Narrative: {_narrative(b)}")
        if b.get("entry") is not None:
            lines.append(f"Entry: {b['entry']:.2f}")
        if b.get("stop") is not None:
            lines.append(f"Stop: {b['stop']:.2f}")
        if b.get("entry") is not None and b.get("stop") is not None:
            stop_dist_pct = abs(b["entry"] - b["stop"]) / b["entry"] * 100
            lines.append(f"Stop distance: {stop_dist_pct:.2f}%")
        lines.append(f"Risk per trade: {b['risk_per_trade']:.3%}")
        lines.append(f"Position size (pct equity): {b['position_size_pct']:.3%}")
        if b.get("reasons"):
            lines.append("Reasons (SMT rules triggered):")
            for r in b["reasons"]:
                lines.append(f"- {r}")
        else:
            lines.append("Reasons: No SMT rule triggered for a BUY/SELL signal.")
        # Always show rule checks for transparency
        if b.get("rule_checks"):
            lines.append("Rule checks:")
            for chk in b["rule_checks"]:
                rule = chk.get("rule", {})
                name = rule.get("name") or rule.get("id") or "rule"
                trig = "YES" if chk.get("triggered") else "no"
                note = chk.get("note") or ""
                lines.append(f"- {name}: {trig} {note}".rstrip())
        lines.append("")
        if b.get("chart_paths"):
            lines.append("Charts:")
            labels = ["Daily", "Weekly", "Monthly"]
            for label, path in zip(labels, b["chart_paths"]):
                lines.append(f"{label} chart:")
                lines.append(f"![{b['instrument']} {label.lower()} chart]({path})")
            lines.append("")
    return "\n".join(lines)


def _render_html(report_date: str, blocks: list[dict], cycle_summary: list[str] | None = None, ocr_notes: list[str] | None = None) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Daily Signals</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:24px auto;line-height:1.4}h1{margin-bottom:8px}h2{margin-top:24px}ul{margin-top:6px}img{max-width:100%;height:auto;border:1px solid #ddd}</style>",
        "</head><body>",
        f"<h1>Daily Signals — {report_date} (NY Close)</h1>",
        "<p><strong>Educational signal generation only. Not financial advice.</strong></p>",
    ]
    if cycle_summary:
        parts.append("<h2>Cycle Summary</h2><ul>")
        for line in cycle_summary:
            parts.append(f"<li>{line[2:] if line.startswith('- ') else line}</li>")
        parts.append("</ul>")
    if ocr_notes:
        parts.append("<h2>Cycle OCR Notes</h2><ul>")
        for line in ocr_notes:
            parts.append(f"<li>{line[2:] if line.startswith('- ') else line}</li>")
        parts.append("</ul>")
    for b in blocks:
        parts.append(f"<h2>{b['instrument']}</h2>")
        parts.append(f"<p><strong>Signal:</strong> {b['signal']}</p>")
        parts.append(f"<p><strong>Signal confidence:</strong> {b['confidence_score']:.2f}</p>")
        parts.append(f"<p><strong>Regime:</strong> {b.get('regime')}</p>")
        if b.get("regime_reason"):
            parts.append(f"<p><strong>Regime reason:</strong> {b.get('regime_reason')}</p>")
        if b.get("cycle_note"):
            parts.append(f"<p><strong>Cycle note (OCR):</strong> {b.get('cycle_note')}</p>")
        if b.get("cycle_state"):
            parts.append(f"<p><strong>Cycle state:</strong> {b.get('cycle_state')}</p>")
        parts.append(f"<p><strong>Narrative:</strong> {_narrative(b)}</p>")
        if b.get("entry") is not None:
            parts.append(f"<p><strong>Entry:</strong> {b['entry']:.2f}</p>")
        if b.get("stop") is not None:
            parts.append(f"<p><strong>Stop:</strong> {b['stop']:.2f}</p>")
        if b.get("entry") is not None and b.get("stop") is not None:
            stop_dist_pct = abs(b["entry"] - b["stop"]) / b["entry"] * 100
            parts.append(f"<p><strong>Stop distance:</strong> {stop_dist_pct:.2f}%</p>")
        parts.append(f"<p><strong>Risk per trade:</strong> {b['risk_per_trade']:.3%}</p>")
        parts.append(f"<p><strong>Position size (pct equity):</strong> {b['position_size_pct']:.3%}</p>")
        if b.get("reasons"):
            parts.append("<p><strong>Reasons (SMT rules triggered):</strong></p><ul>")
            for r in b["reasons"]:
                parts.append(f"<li>{r}</li>")
            parts.append("</ul>")
        else:
            parts.append("<p><strong>Reasons:</strong> No SMT rule triggered for a BUY/SELL signal.</p>")
        if b.get("rule_checks"):
            parts.append("<p><strong>Rule checks:</strong></p><ul>")
            for chk in b["rule_checks"]:
                rule = chk.get("rule", {})
                name = rule.get("name") or rule.get("id") or "rule"
                trig = "YES" if chk.get("triggered") else "no"
                note = chk.get("note") or ""
                parts.append(f"<li>{name}: {trig} {note}</li>")
            parts.append("</ul>")
        if b.get("chart_paths"):
            labels = ["Daily", "Weekly", "Monthly"]
            for label, path in zip(labels, b["chart_paths"]):
                parts.append(f"<p><strong>{label} chart:</strong></p>")
                parts.append(f"<img alt='{b['instrument']} {label.lower()} chart' src='{path}' />")
    parts.append("</body></html>")
    return "\n".join(parts)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out


def _save_chart(
    df: pd.DataFrame,
    instrument: str,
    out_dir: Path,
    timeframe: str,
    lookback: int | None = None,
    ma_windows: list[int] | None = None,
    cycle_note: str | None = None,
    cycle_markers: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_plot = df.tail(lookback) if lookback else df
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_plot.index, df_plot["close"], label="Close", linewidth=1.3, color="#111111")
    ma_windows = ma_windows or [20, 50, 200]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, w in enumerate(ma_windows):
        ma = ind.sma(df["close"], int(w))
        ax.plot(
            df_plot.index,
            ma.loc[df_plot.index],
            label=f"MA{w}",
            linewidth=1.0,
            color=colors[i % len(colors)],
            linestyle="--" if i < 2 else "-",
        )
    if cycle_markers:
        close_series = df_plot["close"]
        for m in cycle_markers:
            date = pd.to_datetime(m.get("date"))
            if date not in close_series.index:
                try:
                    idx = close_series.index.get_indexer([date], method="nearest")[0]
                    date = close_series.index[idx]
                except Exception:
                    continue
            y = close_series.loc[date]
            ax.scatter(
                date,
                y,
                label=m.get("label"),
                color=m.get("color"),
                marker=m.get("marker"),
                s=m.get("size", 40),
                zorder=3,
                edgecolors="#111111",
                linewidths=0.6,
                alpha=0.85,
            )
    if cycle_note:
        ax.text(
            0.02,
            0.95,
            f"Cycle note: {cycle_note}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#999999"),
        )
    if cycle_markers:
        ax.text(
            0.02,
            0.80,
            "Markers: DCL=blue v, HCL=yellow o, ICL=green ^",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="#999999"),
        )
        ax.text(
            0.02,
            0.72,
            "Size: ICL > DCL > HCL; larger = higher confidence",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="#999999"),
        )
    ax.set_title(f"{instrument} {timeframe.title()} Price + MAs")
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / f"{instrument}_{timeframe}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_report(date_str: str | None = None, config_path: str = "config.yaml") -> Path:
    if date_str:
        report_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        report_date = datetime.now().date()

    app = load_config(config_path)
    cfg = signal_config(app)
    instruments = app.raw.get("instruments", ["GLD", "SLV"])
    report_cfg = app.raw.get("report", {}) or {}
    include_charts = report_cfg.get("include_charts", True)
    lookback_days = report_cfg.get("chart_lookback_days")
    lookback_weeks = report_cfg.get("chart_lookback_weeks")
    lookback_months = report_cfg.get("chart_lookback_months")
    ma_daily = report_cfg.get("chart_ma_daily", [20, 50, 200])
    ma_weekly = report_cfg.get("chart_ma_weekly", [10, 20, 40])
    ma_monthly = report_cfg.get("chart_ma_monthly", [6, 12, 24])
    marker_limit = report_cfg.get("cycle_marker_limit", 5)
    thresholds = {
        "high": report_cfg.get("cycle_conf_high", 0.7),
        "med": report_cfg.get("cycle_conf_med", 0.4),
    }
    colors = report_cfg.get("cycle_conf_colors", {}) or {}
    type_colors = report_cfg.get("cycle_type_colors", {}) or {}
    output_dir = Path(app.raw.get("report", {}).get("output_dir", "reports"))
    blocks = []

    cycles_cfg = app.raw.get("cycles", {}) or {}
    cycles_path = Path(cycles_cfg.get("output_dir", "data/cycles")) / "cycle_notes.json"
    cycle_specs = _load_cycle_specs(cycles_cfg.get("specs_path", "data/cycle_specs.json"))
    manual_path = cycles_cfg.get("manual_points_path", "data/cycles/manual_cycle_points.csv")

    for inst in instruments:
        df = fetch_yfinance(inst)
        signals = generate_signals(df, inst, config=cfg)
        last = signals.iloc[-1]
        block = _signal_block(inst, last)
        regime = _regime_summary(df, signals)
        block["regime"] = f"{regime['trend']} | volatility: {regime['volatility']}"
        block["regime_reason"] = regime["reason"]
        block["cycle_note"] = _load_cycle_notes(cycles_path, inst)

        # Compute cycle markers for charts
        if cfg.use_cycles:
            cycles = detect_cycles(df.reset_index(), inst, specs=cycle_specs)
            manual = load_manual_cycles(manual_path)
            cycles = apply_manual_overrides(cycles, manual, inst)
            block["cycle_markers"] = _build_cycle_markers(cycles, marker_limit, thresholds, colors, type_colors=type_colors)
        if "dcl_day" in signals.columns:
            dcl_day = _safe_float(last.get("dcl_day"))
            icl_day = _safe_float(last.get("icl_day"))
            dcl_range = spec_range(cycle_specs, inst, "daily", "DCL") if cycle_specs else None
            icl_range = spec_range(cycle_specs, inst, "daily", "ICL") if cycle_specs else None
            dcl_txt = f"DCL day {dcl_day:.0f}" if dcl_day == dcl_day else "DCL day n/a"
            icl_txt = f"ICL day {icl_day:.0f}" if icl_day == icl_day else "ICL day n/a"
            if dcl_range:
                dcl_txt += f" (exp {dcl_range[0]}-{dcl_range[1]})"
            if icl_range:
                icl_txt += f" (exp {icl_range[0]}-{icl_range[1]})"
            block["cycle_state"] = f"{dcl_txt}; {icl_txt}"
            block["cycle_flags"] = {
                "dcl_in_range": bool(dcl_range and dcl_day == dcl_day and dcl_range[0] <= dcl_day <= dcl_range[1]),
                "icl_in_range": bool(icl_range and icl_day == icl_day and icl_range[0] <= icl_day <= icl_range[1]),
            }
            # Collect cycle low markers for charts
            if {"cycle_type", "date"}.issubset(signals.columns):
                cycle_points = signals[["cycle_type", "date"]].dropna()
                dcl_dates = list(pd.to_datetime(cycle_points[cycle_points["cycle_type"] == "DCL"]["date"]))
                hcl_dates = list(pd.to_datetime(cycle_points[cycle_points["cycle_type"] == "HCL"]["date"]))
                icl_dates = list(pd.to_datetime(cycle_points[cycle_points["cycle_type"] == "ICL"]["date"]))
                block["cycle_markers"] = {
                    "points": [
                        ("DCL", "#1f77b4", "v", dcl_dates),
                        ("HCL", "#ff7f0e", "o", hcl_dates),
                        ("ICL", "#2ca02c", "^", icl_dates),
                    ]
                }
        if include_charts:
            # Daily chart
            daily_path = _save_chart(
                df,
                inst,
                output_dir,
                "daily",
                lookback=lookback_days,
                ma_windows=ma_daily,
                cycle_note=block["cycle_note"],
                cycle_markers=block.get("cycle_markers"),
            )
            # Weekly chart
            weekly_df = _resample_ohlcv(df, "W-FRI")
            weekly_path = _save_chart(
                weekly_df,
                inst,
                output_dir,
                "weekly",
                lookback=lookback_weeks,
                ma_windows=ma_weekly,
                cycle_note=block["cycle_note"],
                cycle_markers=block.get("cycle_markers"),
            )
            # Monthly chart
            monthly_df = _resample_ohlcv(df, "ME")
            monthly_path = _save_chart(
                monthly_df,
                inst,
                output_dir,
                "monthly",
                lookback=lookback_months,
                ma_windows=ma_monthly,
                cycle_note=block["cycle_note"],
                cycle_markers=block.get("cycle_markers"),
            )

            block["chart_paths"] = [p.name for p in [daily_path, weekly_path, monthly_path]]
        blocks.append(block)

    cycle_summary = _cycle_summary_lines(blocks)
    ocr_notes = _ocr_notes_lines(_load_cycle_notes_all(cycles_path), instruments=instruments)
    content = _render_markdown(str(report_date), blocks, cycle_summary=cycle_summary, ocr_notes=ocr_notes)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{report_date}.md"
    out_path.write_text(content, encoding="utf-8")

    output_html = app.raw.get("report", {}).get("output_html", False)
    if output_html:
        html_path = output_dir / f"{report_date}.html"
        html_path.write_text(_render_html(str(report_date), blocks, cycle_summary=cycle_summary, ocr_notes=ocr_notes), encoding="utf-8")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily SMT signal report.")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    out = generate_report(args.date, args.config)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
