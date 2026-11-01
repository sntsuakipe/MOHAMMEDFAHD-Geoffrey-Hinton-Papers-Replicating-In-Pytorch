#!/usr/bin/env python3
"""Download legally accessible PDF copies for the papers in this repository.

This script intentionally avoids mirroring paywalled publisher PDFs. It only tries
URLs that are marked as open-access in the manifest. If a paper has no verified
open-access URL, it is skipped.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdf"
MANIFEST_PATH = ROOT / "manifest.json"


def load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(entries: List[Dict[str, Any]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def fetch_with_retries(url: str, timeout: int = 20, max_retries: int = 3) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as resp:
                body = resp.read()
                if not body:
                    raise RuntimeError(f"Empty response from {url}")
                return body
        except Exception as exc:  # pragma: no cover - network errors are expected
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def download_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    title = entry.get("title", "unknown")
    slug = entry.get("slug") or title.lower().replace(" ", "-")
    pdf_url = entry.get("pdf_url")

    if not pdf_url:
        entry["status"] = "skipped_no_open_pdf"
        entry["downloaded_pdf"] = None
        print(f"[SKIP] {title}: no verified open-access PDF URL")
        return entry

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / f"{slug}.pdf"

    try:
        data = fetch_with_retries(pdf_url)
    except Exception as exc:
        entry["status"] = "failed_download"
        entry["downloaded_pdf"] = None
        entry["error"] = str(exc)
        print(f"[FAIL] {title}: {exc}")
        return entry

    destination.write_bytes(data)
    entry["status"] = "downloaded"
    entry["downloaded_pdf"] = str(destination.name)
    entry["downloaded_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    print(f"[OK] {title} -> {destination.name}")
    return entry


def main() -> None:
    entries = load_manifest()
    changed = False

    for entry in entries:
        if entry.get("status") == "downloaded":
            print(f"[KEEP] {entry.get('title')}: already downloaded")
            continue
        updated = download_entry(entry)
        if updated != entry:
            changed = True

    if changed:
        save_manifest(entries)
        print(f"\nUpdated manifest: {MANIFEST_PATH}")
    else:
        print("\nNo changes needed.")

    print(f"PDFs saved under: {PDF_DIR}")


if __name__ == "__main__":
    main()
