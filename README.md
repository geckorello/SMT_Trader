# SMT_Trader

Rules-based GOLD/SILVER timing system driven by SmartMoneyTrackerPremium (SMT) exports plus market data.
This project produces daily BUY/SELL/HOLD signals, walk-forward backtests, and human-readable reports.

[![CI](https://github.com/geckorello/SMT_Trader/actions/workflows/ci.yml/badge.svg)](https://github.com/geckorello/SMT_Trader/actions/workflows/ci.yml)

## What This Does
- Ingests public SMT pages and local PDF/HTML/TXT exports.
- Extracts a structured rule book from SMT text (`data/rules/rules.json`).
- Implements SMT proxy rules + cycle detection (DCL/HCL/ICL).
- Runs walk-forward backtests and produces daily reports.
- Generates a Rule Lab report to compare single rules and combos.

Educational signal generation only. Not financial advice.

## Quickstart
1) Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

2) Place SMT exports
```
data/smt_exports/
```

3) Run the pipeline
```bash
python3 run_all.py --config config.yaml --ingest-local --backtest --start 2010-01-01 --end 2026-02-03
```

Outputs:
- Daily reports: `reports/YYYY-MM-DD.md` and `reports/YYYY-MM-DD.html`
- Signals JSON/CSV
- Walk-forward backtests: `reports/backtest_*`
- Rule Lab: `reports/rule_lab_*`

## Reproducibility
- All configuration lives in `config.yaml`.
- Market data source is `yfinance` (Yahoo); specify `--start/--end` for reproducible runs.
- SMT content is only ingested from local exports or public pages (no auth bypass).
- Generated outputs are date-stamped in `reports/`.
- To fully reproduce historical outputs, snapshot the input exports and data.

## Cycle Detection (Module)
This module adds production-ready cycle detection (DCL/HCL/ICL) and PDF-based calibration from SMT exports.

### Build Cycle Specs From PDFs
```bash
python3 - <<'PY'
from cycle_engine.pdf_specs import load_or_build_specs
load_or_build_specs(exports_dir='/smt_exports', out_path='data/cycle_specs.json')
PY
```

### Run Cycle Detection On a Sample OHLCV CSV
```bash
python3 - <<'PY'
import pandas as pd
from cycle_engine.detect import detect_cycles

# CSV with columns: date,open,high,low,close,volume
_df = pd.read_csv('data/GLD.csv', parse_dates=['date'])
cycles = detect_cycles(_df, 'GLD', timeframe='daily')
print(cycles.head())
PY
```

### Generate Diagnostics Plot (Optional)
```bash
python3 - <<'PY'
import pandas as pd
from cycle_engine.diagnostics import plot_cycles

_df = pd.read_csv('data/GLD.csv', parse_dates=['date'])
plot_cycles(_df, 'GLD', out_path='reports/cycle_diagnostics.png')
PY
```

### Notes
- PDF calibration only extracts **text** (no chart image inference).
- If no SMT specs are available, default cycle ranges are used.
- Confidence scoring is explainable in the `notes` column.

## Manual Cycle Overrides
You can override detected cycle lows by editing:
`data/cycles/manual_cycle_points.csv`

Format:
```
date,instrument,cycle_type,notes
2026-01-15,GLD,DCL,Manual daily cycle low
```
Manual points are merged into detection results and used for cycle-day state.

## Optional OCR (Cycle Charts)
If you want OCR on cycle chart PDFs:
```bash
python3 -m pip install pymupdf pytesseract
brew install tesseract
```

## Runbook
See `RUNBOOK.md` for weekly ingestion, daily signal generation, and deployment notes.
