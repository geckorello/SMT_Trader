# SMT Cycle Detection Module

This module adds production-ready cycle detection (DCL/HCL/ICL) and PDF-based calibration from SMT exports.

## How To Run

1) Build cycle specs from PDFs
```bash
python3 - <<'PY'
from cycle_engine.pdf_specs import load_or_build_specs
load_or_build_specs(exports_dir='/smt_exports', out_path='data/cycle_specs.json')
PY
```

2) Run cycle detection on a sample OHLCV CSV
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

3) Generate diagnostics plot (optional)
```bash
python3 - <<'PY'
import pandas as pd
from cycle_engine.diagnostics import plot_cycles

_df = pd.read_csv('data/GLD.csv', parse_dates=['date'])
plot_cycles(_df, 'GLD', out_path='reports/cycle_diagnostics.png')
PY
```

## Notes
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
