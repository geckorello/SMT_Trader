"""Extract SMT cycle length specs from PDF exports."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _read_pdf_text(path: Path) -> str:
    # Try pdfplumber, fallback to PyPDF2
    try:
        import pdfplumber  # type: ignore

        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""


def parse_specs_from_text(text: str, source: str) -> Dict[str, dict]:
    """Parse SMT cycle length specs from raw text."""
    out: Dict[str, dict] = {}
    lower = text.lower()

    # Common patterns like: "Gold daily cycle count: Day 22 (average duration 30–40 days)"
    pattern = re.compile(
        r"(?P<inst>gold|silver|gld|slv|stock market|stocks|spx|s&p ?500)[^\n]{0,120}?"
        r"(?P<tf>daily|weekly)[^\n]{0,120}?"
        r"average duration\s*(?P<min>\d+)\s*[–-]\s*(?P<max>\d+)\s*(?P<unit>days|weeks)",
        re.IGNORECASE,
    )

    # Secondary: sometimes "average duration 35-45 days" appears without daily/weekly nearby
    pattern2 = re.compile(
        r"(?P<inst>gold|silver|gld|slv|stock market|stocks|spx|s&p ?500)[^\n]{0,120}?"
        r"average duration\s*(?P<min>\d+)\s*[–-]\s*(?P<max>\d+)\s*(?P<unit>days|weeks)",
        re.IGNORECASE,
    )

    def norm_inst(inst: str) -> str:
        inst = inst.lower()
        if inst in {"gold", "gld"}:
            return "GOLD"
        if inst in {"silver", "slv"}:
            return "SILVER"
        if inst in {"stock market", "stocks", "spx", "s&p 500", "s&p500"}:
            return "STOCKS"
        return inst.upper()

    def apply(inst: str, tf: str, min_v: int, max_v: int, unit: str) -> None:
        inst_key = norm_inst(inst)
        tf_key = "daily" if tf.lower().startswith("day") else "weekly"
        # If unit mismatches timeframe, still store but record unit in notes
        out.setdefault(inst_key, {})
        out[inst_key][tf_key] = {
            "min": int(min_v),
            "max": int(max_v),
            "unit": unit.lower(),
            "source": source,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }

    for m in pattern.finditer(lower):
        apply(m.group("inst"), m.group("tf"), int(m.group("min")), int(m.group("max")), m.group("unit"))

    # Use pattern2 only if nothing found for that instrument
    if not out:
        for m in pattern2.finditer(lower):
            apply(m.group("inst"), "daily", int(m.group("min")), int(m.group("max")), m.group("unit"))

    return out


def extract_specs_from_pdfs(exports_dir: str) -> Dict[str, dict]:
    export_path = Path(exports_dir)
    if not export_path.exists():
        return {}

    out: Dict[str, dict] = {}
    for pdf in export_path.glob("*.pdf"):
        text = _read_pdf_text(pdf)
        if not text:
            continue
        specs = parse_specs_from_text(text, pdf.name)
        for inst, val in specs.items():
            out.setdefault(inst, {})
            out[inst].update(val)
    return out


def load_or_build_specs(exports_dir: str = "/smt_exports", out_path: str = "data/cycle_specs.json") -> Dict[str, dict]:
    # Allow relative fallback to data/smt_exports if /smt_exports doesn't exist
    exp = Path(exports_dir)
    if not exp.exists():
        exp = Path("data/smt_exports")
    specs = extract_specs_from_pdfs(str(exp))

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "exports_dir": str(exp),
        },
        "specs": specs,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
