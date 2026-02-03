#!/usr/bin/env python3
"""OCR cycle chart images from SMT PDF exports and extract cycle notes.

Requires:
- pymupdf (fitz)
- pytesseract
- system tesseract binary installed (e.g., `brew install tesseract`)
"""
from __future__ import annotations

import argparse
import json
import re
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEYWORDS = ["cycle", "day", "dcl", "icl", "hcl", "ycl", "cycle count"]


def _load_pymupdf(strict: bool):
    try:
        import fitz  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError("pymupdf is required. Install with: pip install pymupdf") from exc
        print("OCR skipped: pymupdf not installed (pip install pymupdf)")
        return None
    return fitz


def _load_tesseract(strict: bool):
    try:
        import pytesseract  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError("pytesseract is required. Install with: pip install pytesseract") from exc
        print("OCR skipped: pytesseract not installed (pip install pytesseract)")
        return None
    return pytesseract


def _ocr_page(pix, pytesseract) -> str:
    from PIL import Image

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text


def _fingerprint_pdfs(pdfs: list[Path]) -> str:
    parts = []
    for p in pdfs:
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    blob = "|".join(sorted(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_existing_fingerprint(out_dir: Path) -> str | None:
    path = out_dir / "cycle_notes.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("meta", {}).get("source_fingerprint")


def _extract_notes(text: str, source: str) -> list[dict]:
    notes = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if not any(k in low for k in KEYWORDS):
            continue
        # Find cycle day
        m = re.search(r"\bday\s*(\d+)\b", low)
        if not m:
            continue
        day = int(m.group(1))
        inst = None
        if "gold" in low or "gld" in low or "xau" in low:
            inst = "GLD"
        if "silver" in low or "slv" in low or "xag" in low:
            inst = "SLV" if inst is None else inst
        confidence = 0.4 if inst else 0.25
        notes.append(
            {
                "instrument": inst,
                "label": f"Cycle day {day} (OCR)",
                "confidence": confidence,
                "source": source,
                "text": ln,
            }
        )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR cycle charts from SMT PDF exports.")
    parser.add_argument("--exports", default="data/smt_exports")
    parser.add_argument("--out", default="data/cycles")
    parser.add_argument("--filter", default="cycle", help="Only OCR PDFs containing this substring")
    parser.add_argument("--force", action="store_true", help="Force OCR even if PDFs unchanged")
    parser.add_argument("--strict", action="store_true", help="Fail if OCR dependencies are missing")
    args = parser.parse_args()

    fitz = _load_pymupdf(args.strict)
    pytesseract = _load_tesseract(args.strict)
    if fitz is None or pytesseract is None:
        return 0

    export_dir = Path(args.exports)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in export_dir.glob("*.pdf") if args.filter.lower() in p.name.lower()]
    if not pdfs:
        print("No matching PDFs found for OCR.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _fingerprint_pdfs(pdfs)
    existing_fp = _load_existing_fingerprint(out_dir)
    if (existing_fp == fingerprint) and not args.force:
        print("No changes detected in cycle PDFs. Skipping OCR.")
        return 0

    ocr_dump = []
    notes = []

    for pdf in pdfs:
        doc = fitz.open(pdf)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            text = _ocr_page(pix, pytesseract)
            ocr_dump.append(
                {
                    "file": pdf.name,
                    "page": i + 1,
                    "text": text,
                }
            )
            notes.extend(_extract_notes(text, f"{pdf.name}#page={i+1}"))
        doc.close()

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": fingerprint,
            "files": [p.name for p in pdfs],
        },
        "notes": notes,
        "ocr_dump": ocr_dump,
    }

    out_json = out_dir / "cycle_notes.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human review summary
    lines = [
        f"# Cycle OCR Review ({datetime.now(timezone.utc).date()})",
        "",
        f"Files scanned: {len(pdfs)}",
        f"Notes extracted: {len(notes)}",
        "",
    ]
    for n in notes[:30]:
        lines.append(f"- {n.get('instrument') or 'UNKNOWN'} | {n['label']} | {n['text']} | {n['source']}")
    if len(notes) > 30:
        lines.append(f"- ... plus {len(notes) - 30} more")

    (out_dir / "cycle_notes_review.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_dir / 'cycle_notes_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
