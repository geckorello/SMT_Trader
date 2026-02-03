# Runbook — SMT Gold/Silver Timing Tool

Educational signal generation only. Not financial advice.

## Weekly ingestion (public SMT pages)
1. Update the URL list in `data/smt_urls.txt`.
2. Run:
   ```bash
   python3 scripts/ingest_smt.py --urls-file data/smt_urls.txt
   ```

## Ingest member exports (local PDFs/HTML/TXT)
1. Place exports in `data/smt_exports/`.
2. Run:
   ```bash
   python3 scripts/ingest_local.py
   ```

## Update rules + glossary
```bash
python3 scripts/extract_rules.py
```
Review `data/rules/rules_review.md` for accuracy.

## Build cycle specs (from SMT PDFs)
```bash
python3 scripts/build_cycle_specs.py --out data/cycle_specs.json
```

## Manual cycle overrides
Edit `data/cycles/manual_cycle_points.csv` to add known cycle lows.

## Backtest (CSV or Yahoo)
CSV:
```bash
python3 backtest.py --data data/GLD.csv --instrument GLD
```
Yahoo:
```bash
python3 backtest.py --yahoo --instrument GLD --start 2010-01-01 --end 2026-02-03 --config config.yaml
```

## Generate daily signals report
```bash
python3 report.py --config config.yaml
```

## Generate daily signals JSON
```bash
python3 signal_runner.py --config config.yaml
```

## Run full pipeline
```bash
python3 run_all.py --config config.yaml --ingest-local --backtest --start 2010-01-01 --end 2026-02-03
```

## Rule Lab (best single rule / combo)
```bash
python3 rule_lab.py --yahoo --instrument GLD --start 2010-01-01 --end 2026-02-03 --config config.yaml --max-combo 3
```
Outputs HTML/CSV in `reports/` as `rule_lab_<instrument>_<date>.*`.

## Parameter sensitivity sweep
```bash
python3 sensitivity.py --instrument GLD --start 2010-01-01 --end 2026-02-03 --config config.yaml
```
Output is saved to `reports/YYYY-MM-DD.md`.

## Deployment options
- Cron (local): schedule `python3 report.py` daily after NY close.
- GitHub Actions: run on a daily schedule with a cron job and upload report artifacts.

## Notes
- Do not bypass SMT paywalls. Only ingest public URLs or user-provided exports.
- Rules labeled `proxy` are approximations for discretionary SMT language.
- All signals include reason lists and risk controls.
