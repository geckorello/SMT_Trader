#!/usr/bin/env python3
"""
Ingest local SMT member exports placed in data/smt_exports/.
Supports HTML/PDF/TXT/MD. Saves cleaned text + metadata to data/smt_pages/*.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

DATA_DIR = Path("data/smt_pages")
EXPORT_DIR = Path("data/smt_exports")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_html(path: Path) -> tuple[str, str]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(" ")
    return title, clean_text(text)


def extract_text_file(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return path.stem, clean_text(text)


def extract_pdf(path: Path) -> tuple[str, str] | None:
    # Try pdfplumber first, then PyPDF2
    try:
        import pdfplumber  # type: ignore

        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return path.stem, clean_text("\n".join(texts))
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return path.stem, clean_text("\n".join(texts))
    except Exception:
        pass

    # Fallback to system pdftotext if available
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_txt = Path(tmpdir) / "out.txt"
                subprocess.run(
                    [pdftotext, str(path), str(out_txt)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if out_txt.exists():
                    text = out_txt.read_text(encoding="utf-8", errors="ignore")
                    return path.stem, clean_text(text)
        except Exception:
            return None

    return None


def save_page(payload: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{path_safe(payload['file_name'])}_{ts}.json"
    out_path = DATA_DIR / name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def path_safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")[:120] or "export"


def ingest_file(path: Path) -> Path | None:
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        title, text = extract_html(path)
    elif ext in {".txt", ".md"}:
        title, text = extract_text_file(path)
    elif ext == ".pdf":
        result = extract_pdf(path)
        if not result:
            print(f"SKIP (unable to parse PDF): {path}")
            return None
        title, text = result
    else:
        print(f"SKIP (unsupported extension): {path}")
        return None

    if not text:
        print(f"SKIP (empty text): {path}")
        return None

    stat = path.stat()
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = {
        "source": "local_export",
        "file_name": path.name,
        "original_path": str(path.resolve()),
        "title": title,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "file_mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "text": text,
        "content_hash": content_hash,
    }
    return save_page(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local SMT exports to JSON.")
    parser.add_argument("--dir", default=str(EXPORT_DIR))
    args = parser.parse_args()

    in_dir = Path(args.dir)
    if not in_dir.exists():
        print(f"Directory not found: {in_dir}")
        return 1

    saved = 0
    for path in sorted(in_dir.iterdir()):
        if path.is_file():
            out = ingest_file(path)
            if out:
                print(f"SAVED: {out}")
                saved += 1

    print(f"Done. Saved {saved} export(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
