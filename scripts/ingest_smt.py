#!/usr/bin/env python3
"""
Ingest public SmartMoneyTrackerPremium (SMT) pages.
Respects robots.txt and avoids any authentication bypass.
Saves cleaned text + metadata to data/smt_pages/*.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path("data/smt_pages")
DEFAULT_UA = "SMT-Edu-Ingest/1.0 (+educational use; respects robots.txt)"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Remove common boilerplate tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(" ")
    return title, clean_text(text)


def robots_allows(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt is unreachable, be conservative
        return False
    return rp.can_fetch(user_agent, url)


def slugify(url: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", url)
    safe = safe.strip("-")
    return safe[:120] or "page"


def save_page(payload: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{slugify(payload['url'])}_{ts}.json"
    out_path = DATA_DIR / name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def ingest_url(url: str, user_agent: str, timeout: int) -> Path | None:
    if not robots_allows(url, user_agent):
        print(f"SKIP (robots.txt disallows): {url}")
        return None

    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        print(f"ERROR fetching {url}: {exc}")
        return None

    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code != 200:
        print(f"ERROR {resp.status_code} fetching {url}")
        return None

    if "text/html" not in content_type:
        print(f"SKIP (non-HTML content): {url} ({content_type})")
        return None

    title, text = extract_text(resp.text)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    payload = {
        "source": "public_web",
        "url": url,
        "title": title,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "http_status": resp.status_code,
        "content_type": content_type,
        "text": text,
        "content_hash": content_hash,
    }
    return save_page(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest public SMT pages to JSON.")
    parser.add_argument("--urls", nargs="*", help="List of SMT URLs to fetch")
    parser.add_argument("--urls-file", help="Text file with one URL per line")
    parser.add_argument("--user-agent", default=DEFAULT_UA)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()

    urls = []
    if args.urls:
        urls.extend(args.urls)
    if args.urls_file:
        with open(args.urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    urls = list(dict.fromkeys(urls))
    if not urls:
        print("No URLs provided. Use --urls or --urls-file.")
        return 1

    saved = 0
    for i, url in enumerate(urls, start=1):
        out = ingest_url(url, args.user_agent, args.timeout)
        if out:
            print(f"SAVED: {out}")
            saved += 1
        if i < len(urls):
            time.sleep(args.sleep)

    print(f"Done. Saved {saved} page(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
