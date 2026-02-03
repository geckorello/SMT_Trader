#!/usr/bin/env python3
"""Build cycle_specs.json from SMT PDF exports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cycle_engine.pdf_specs import load_or_build_specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SMT cycle specs from PDFs.")
    parser.add_argument("--exports", default="/smt_exports")
    parser.add_argument("--out", default="data/cycle_specs.json")
    args = parser.parse_args()

    load_or_build_specs(exports_dir=args.exports, out_path=args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
