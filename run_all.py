"""Run full pipeline: ingest (optional), extract rules, backtest, report, signals."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
import subprocess

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def _ocr_ready() -> bool:
    has_fitz = importlib.util.find_spec("fitz") is not None
    has_pytesseract = importlib.util.find_spec("pytesseract") is not None
    has_tesseract_bin = shutil.which("tesseract") is not None
    return has_fitz and has_pytesseract and has_tesseract_bin


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SMT pipeline end-to-end.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ingest-public", action="store_true", help="Run public SMT ingestion")
    parser.add_argument("--urls-file", default="data/smt_urls.txt")
    parser.add_argument("--ingest-local", action="store_true", help="Run local exports ingestion")
    parser.add_argument("--backtest", action="store_true", help="Run backtest (Yahoo) for GLD/SLV)")
    parser.add_argument("--start", help="Backtest start YYYY-MM-DD")
    parser.add_argument("--end", help="Backtest end YYYY-MM-DD")
    args = parser.parse_args()

    if args.ingest_public:
        run(["python3", "scripts/ingest_smt.py", "--urls-file", args.urls_file])
    if args.ingest_local:
        run(["python3", "scripts/ingest_local.py"])

    run(["python3", "scripts/extract_rules.py"])

    # Optional cycle OCR
    cycles_cfg = {}
    if yaml is not None:
        try:
            raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            cycles_cfg = raw.get("cycles", {}) or {}
        except Exception:
            cycles_cfg = {}
    if cycles_cfg.get("enable_detection"):
        out_path = cycles_cfg.get("specs_path", "data/cycle_specs.json")
        run(["python3", "scripts/build_cycle_specs.py", "--out", out_path])

    if cycles_cfg.get("enable_ocr"):
        out_dir = cycles_cfg.get("output_dir", "data/cycles")
        if _ocr_ready():
            run(["python3", "scripts/ocr_cycles.py", "--out", out_dir])
        else:
            print("OCR skipped: missing dependencies (pymupdf, pytesseract, and/or tesseract binary).")

    if args.backtest:
        for inst in ["GLD", "SLV"]:
            cmd = ["python3", "backtest.py", "--yahoo", "--instrument", inst, "--config", args.config]
            if args.start:
                cmd += ["--start", args.start]
            if args.end:
                cmd += ["--end", args.end]
            run(cmd)

    run(["python3", "report.py", "--config", args.config])
    run(["python3", "signal_runner.py", "--config", args.config])

    # Optional rule lab
    rule_lab_cfg = {}
    if yaml is not None:
        try:
            raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            rule_lab_cfg = raw.get("rule_lab", {}) or {}
        except Exception:
            rule_lab_cfg = {}
    if rule_lab_cfg.get("enable"):
        max_combo = str(rule_lab_cfg.get("max_combo", 3))
        for inst in ["GLD", "SLV"]:
            cmd = ["python3", "rule_lab.py", "--yahoo", "--instrument", inst, "--config", args.config, "--max-combo", max_combo]
            if args.start:
                cmd += ["--start", args.start]
            if args.end:
                cmd += ["--end", args.end]
            run(cmd)

    # Print one-line summary from latest signal JSON
    output_dir = "reports"
    if yaml is not None:
        try:
            raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            output_dir = raw.get("signals", {}).get("output_dir", "reports")
        except Exception:
            pass
    report_date = datetime.now().date().isoformat()
    json_path = Path(output_dir) / f"{report_date}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        parts = []
        for s in data.get("signals", []):
            conf = s.get("confidence_score", 0.0)
            parts.append(f"{s.get('instrument')}={s.get('signal')} (conf {conf:.2f})")
        if parts:
            print("Latest signals: " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
