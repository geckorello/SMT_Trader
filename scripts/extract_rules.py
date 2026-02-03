#!/usr/bin/env python3
"""
Extract SMT rules + glossary from ingested pages.
Outputs:
- data/rules/rules.json
- data/rules/glossary.json
- data/rules/rules_review.md (human review summary)
"""
from __future__ import annotations

import argparse
import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("data/smt_pages")
OUT_DIR = Path("data/rules")

DISCRETIONARY_WORDS = [
    "often",
    "usually",
    "likely",
    "watch",
    "prepare",
    "may",
    "might",
    "should",
    "beware",
    "expect",
    "possible",
    "generally",
]


@dataclass
class SourceRef:
    source_id: str
    title: str
    origin: str
    location_hint: str


def load_pages() -> list[dict]:
    pages = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source_id"] = path.name
        pages.append(data)
    return pages


def fingerprint_pages(pages: list[dict]) -> str:
    parts = []
    for p in pages:
        sid = p.get("_source_id", "")
        ch = p.get("content_hash", "")
        parts.append(f"{sid}:{ch}")
    blob = "|".join(sorted(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_existing_fingerprint() -> str | None:
    rules_path = OUT_DIR / "rules.json"
    if not rules_path.exists():
        return None
    try:
        data = json.loads(rules_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("meta", {}).get("source_fingerprint")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def find_page(pages: list[dict], title_contains: str) -> dict | None:
    for p in pages:
        if title_contains.lower() in (p.get("title") or "").lower():
            return p
    return None


def label_source(page: dict, hint: str) -> SourceRef:
    origin = page.get("url") or page.get("original_path") or page.get("file_name") or "unknown"
    return SourceRef(
        source_id=page.get("_source_id", "unknown"),
        title=page.get("title", ""),
        origin=origin,
        location_hint=hint,
    )


def extract_labeled_rules(text: str) -> list[dict]:
    rules = []
    cleaned = normalize(text)
    # Capture tokens like T1., G2., or 1.
    token_re = re.compile(r"(T\d+\.|G\d+\.|\b\d+\.)")
    positions = [(m.group(1), m.start()) for m in token_re.finditer(cleaned)]
    if not positions:
        return rules

    for i, (label, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(cleaned)
        segment = cleaned[start:end].strip()
        # Remove label prefix from segment
        segment = re.sub(r"^(T\d+\.|G\d+\.|\d+\.)\s*", "", segment)
        if len(segment) < 20:
            continue
        rules.append({"label": label.strip("."), "text": segment})
    return rules


def extract_four_day_rules(text: str) -> list[str]:
    snippets = []
    for m in re.finditer(r".{0,120}4[- ]day.{0,220}", text, flags=re.IGNORECASE):
        snippet = normalize(m.group(0))
        if "rule" in snippet.lower() or "corollary" in snippet.lower():
            snippets.append(snippet)
    # Deduplicate similar snippets
    uniq = []
    for s in snippets:
        if all(s not in u for u in uniq):
            uniq.append(s)
    return uniq


def extract_glossary_terms(text: str) -> list[tuple[str, str]]:
    items = []
    cleaned = normalize(text)

    # Abbreviations block
    m = re.search(r"Abbreviations: (.{0,500})", cleaned, flags=re.IGNORECASE)
    if m:
        block = m.group(1)
        for abbr, definition in re.findall(r"([A-Z]{2,5})\s*=\s*([^A-Z]{2,80})(?=\s+[A-Z]{2,5}\s*=|$)", block):
            term = abbr.strip()
            defn = normalize(definition)
            if defn:
                items.append((term, defn))

    # Term: definition pattern
    term_re = re.compile(r"([A-Z][A-Za-z0-9\-/ ]{1,40}):")
    matches = list(term_re.finditer(cleaned))
    for i, m in enumerate(matches):
        term = m.group(1).strip()
        if term.lower() in {"important terminology definitions", "dg’s strategies and indicators"}:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        defn = normalize(cleaned[start:end])
        if len(defn) < 30:
            continue
        if len(term) > 40:
            continue
        items.append((term, defn))

    # Deduplicate by term
    seen = {}
    for term, defn in items:
        if term not in seen or len(defn) > len(seen[term]):
            seen[term] = defn
    return list(seen.items())


def flag_discretionary(text: str) -> list[str]:
    flags = []
    lower = text.lower()
    for w in DISCRETIONARY_WORDS:
        if w in lower:
            flags.append(w)
    return flags


def build_rules(pages: list[dict]) -> list[dict]:
    rules = []

    trading = find_page(pages, "Trading Rules")
    if trading:
        source = label_source(trading, "Trading Rules section")
        for rule in extract_labeled_rules(trading.get("text", "")):
            label = rule["label"]
            category = "technical" if label.startswith("T") else "general" if label.startswith("G") else "commonsense"
            text = rule["text"]
            flags = flag_discretionary(text)
            rules.append(
                {
                    "id": f"smt_{label.lower()}_v1",
                    "name": f"SMT {label}",
                    "category": category,
                    "description": text,
                    "if": None,
                    "then": None,
                    "parameters": {},
                    "mechanical": False,
                    "proxy": None,
                    "confidence": 0.3 if flags else 0.4,
                    "discretionary_flags": flags,
                    "provenance": {
                        "source_id": source.source_id,
                        "title": source.title,
                        "origin": source.origin,
                        "location_hint": source.location_hint,
                    },
                }
            )

    terminology = find_page(pages, "Terminology")
    if terminology:
        source = label_source(terminology, "Terminology section")
        snippets = extract_four_day_rules(terminology.get("text", ""))
        if snippets:
            # Build two explicit rules
            rules.append(
                {
                    "id": "smt_4day_corollary_v1",
                    "name": "4-Day Corollary (Proxy)",
                    "category": "trend_change",
                    "description": "After a long intermediate move, the first day counter to the trend following 4 or more days in a row often signals a trend change.",
                    "if": "trend_is_up AND prior_4days_up AND today_down (or vice versa)",
                    "then": "Signal potential trend change (sell/cover in uptrend; buy in downtrend)",
                    "parameters": {
                        "trend_proxy": "MA_50 vs MA_200 slope",
                        "lookback_days": 4,
                    },
                    "mechanical": False,
                    "proxy": "Use MA(50) > MA(200) as long intermediate uptrend; detect 4 consecutive up days then 1 down day.",
                    "confidence": 0.55,
                    "discretionary_flags": ["often"],
                    "provenance": {
                        "source_id": source.source_id,
                        "title": source.title,
                        "origin": source.origin,
                        "location_hint": "4 day corollary snippet",
                        "snippet": snippets[0],
                    },
                }
            )
            rules.append(
                {
                    "id": "smt_4day_rule_v1",
                    "name": "4-Day Rule (Proxy)",
                    "category": "trend_change",
                    "description": "4 or more days in a row counter to a long intermediate trend is often confirmation of a trend change.",
                    "if": "trend_is_up AND last_4days_down (or trend_is_down AND last_4days_up)",
                    "then": "Confirm trend change (sell in uptrend, buy in downtrend)",
                    "parameters": {
                        "trend_proxy": "MA_50 vs MA_200 slope",
                        "lookback_days": 4,
                    },
                    "mechanical": False,
                    "proxy": "Use MA(50) > MA(200) as long intermediate uptrend; detect 4 consecutive counter-trend closes.",
                    "confidence": 0.6,
                    "discretionary_flags": ["often"],
                    "provenance": {
                        "source_id": source.source_id,
                        "title": source.title,
                        "origin": source.origin,
                        "location_hint": "4 day rule snippet",
                        "snippet": snippets[-1],
                    },
                }
            )

            # Add MA touch proxy only if moving average references exist
            text = terminology.get("text", "")
            if re.search(r"moving average|\bDMA\b", text, flags=re.IGNORECASE):
                rules.append(
                    {
                        "id": "smt_ma_touch_proxy_v1",
                        "name": "Moving Average Touch (Proxy)",
                        "category": "mean_reversion",
                        "description": "Moving averages (DMA/WMA) are referenced in SMT terminology. A mechanical proxy is to treat a touch/close at a key MA in the direction of the prevailing trend as a signal.",
                        "if": "trend_is_up AND close <= MA_20 OR MA_50",
                        "then": "BUY (mean reversion within trend)",
                        "parameters": {"ma_short": 20, "ma_mid": 50, "trend_proxy": "MA_50 vs MA_200"},
                        "mechanical": False,
                        "proxy": "Approximation only; SMT references MAs but does not specify a mechanical 'touch' rule in the provided text.",
                        "confidence": 0.35,
                        "discretionary_flags": ["proxy"],
                        "provenance": {
                            "source_id": source.source_id,
                            "title": source.title,
                            "origin": source.origin,
                            "location_hint": "DMA/WMA definition mention",
                        },
                    }
                )

    return rules


def build_glossary(pages: list[dict]) -> list[dict]:
    glossary = []
    terminology = find_page(pages, "Terminology")
    if not terminology:
        return glossary

    source = label_source(terminology, "Terminology definitions")
    terms = extract_glossary_terms(terminology.get("text", ""))
    for term, definition in terms:
        glossary.append(
            {
                "term": term,
                "definition": definition,
                "provenance": {
                    "source_id": source.source_id,
                    "title": source.title,
                    "origin": source.origin,
                    "location_hint": source.location_hint,
                },
            }
        )
    return glossary


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_review(path: Path, rules: list[dict], glossary: list[dict]) -> None:
    lines = []
    lines.append(f"# SMT Rule Extraction Review ({datetime.now(timezone.utc).date()})")
    lines.append("")
    lines.append(f"Rules extracted: {len(rules)}")
    lines.append(f"Glossary terms extracted: {len(glossary)}")
    lines.append("")

    if rules:
        lines.append("## Rules")
        for r in rules:
            lines.append(f"- {r['id']}: {r['name']}")
            lines.append(f"  - Category: {r['category']}")
            lines.append(f"  - Description: {r['description'][:200]}{'...' if len(r['description'])>200 else ''}")
            if r.get("proxy"):
                lines.append(f"  - Proxy: {r['proxy']}")
            if r.get("discretionary_flags"):
                lines.append(f"  - Discretionary flags: {', '.join(r['discretionary_flags'])}")
            prov = r.get("provenance", {})
            lines.append(f"  - Source: {prov.get('origin')} ({prov.get('title')})")
        lines.append("")

    if glossary:
        lines.append("## Glossary (Sample)")
        for g in glossary[:20]:
            lines.append(f"- {g['term']}: {g['definition'][:160]}{'...' if len(g['definition'])>160 else ''}")
        if len(glossary) > 20:
            lines.append(f"- ... plus {len(glossary) - 20} more")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SMT rules and glossary from ingested pages.")
    parser.add_argument("--force", action="store_true", help="Rebuild rules even if sources unchanged")
    args = parser.parse_args()

    pages = load_pages()
    if not pages:
        print("No ingested pages found in data/smt_pages.")
        return 1

    fingerprint = fingerprint_pages(pages)
    existing_fp = load_existing_fingerprint()
    if (existing_fp == fingerprint) and not args.force:
        print("No changes detected in SMT source pages. Skipping rule extraction.")
        return 0

    rules = build_rules(pages)
    glossary = build_glossary(pages)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pages": [p.get("_source_id") for p in pages],
        "source_fingerprint": fingerprint,
        "notes": "Rules include proxies where SMT language is discretionary or non-mechanical.",
    }

    write_json(OUT_DIR / "rules.json", {"meta": meta, "rules": rules})
    write_json(OUT_DIR / "glossary.json", {"meta": meta, "glossary": glossary})
    write_review(OUT_DIR / "rules_review.md", rules, glossary)

    print(f"Wrote {OUT_DIR / 'rules.json'}")
    print(f"Wrote {OUT_DIR / 'glossary.json'}")
    print(f"Wrote {OUT_DIR / 'rules_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
