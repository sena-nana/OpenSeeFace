#!/usr/bin/env python3
"""Download Wikimedia Commons thumbs listed in fixtures/manifest.json."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "fixtures" / "manifest.json"
CACHE = ROOT / "fixtures" / "cache"
API = "https://commons.wikimedia.org/w/api.php"
UA = "OpenSeeFace-bench/1.0 (https://github.com/emilianavt/OpenSeeFace; local A/B fixtures)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _title(photo: dict) -> str:
    if photo.get("title"):
        t = photo["title"]
        return t if t.startswith("File:") else f"File:{t}"
    return f"File:{photo['file']}"


def fetch_one(photo: dict, dest: Path) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{photo['id']}.jpg"
    if out.is_file() and out.stat().st_size > 0:
        return out
    params = {
        "action": "query",
        "titles": _title(photo),
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "640",
        "format": "json",
    }
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    pages = data.get("query", {}).get("pages", {})
    info = None
    for page in pages.values():
        ii = page.get("imageinfo") or []
        if ii:
            info = ii[0]
            break
    if not info:
        print(f"skip {photo['id']}: no imageinfo", file=sys.stderr)
        return None
    url = info.get("thumburl") or info.get("url")
    if not url:
        print(f"skip {photo['id']}: no url", file=sys.stderr)
        return None
    out.write_bytes(_get(url))
    attr = dest / f"{photo['id']}.json"
    attr.write_text(
        json.dumps(
            {
                "id": photo["id"],
                "license": photo.get("license"),
                "artist": photo.get("artist"),
                "page": photo.get("page"),
                "tags": photo.get("tags", []),
                "source_url": url,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def fetch_all(manifest: Path = MANIFEST, dest: Path = CACHE) -> list[Path]:
    photos = json.loads(manifest.read_text())["photos"]
    got = []
    for photo in photos:
        try:
            path = fetch_one(photo, dest)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"skip {photo.get('id')}: {e}", file=sys.stderr)
            continue
        if path is not None:
            got.append(path)
    return got


def cached_photos(dest: Path = CACHE) -> list[Path]:
    if not dest.is_dir():
        return []
    return sorted(p for p in dest.glob("*.jpg") if p.stat().st_size > 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--out-dir", type=Path, default=CACHE)
    args = p.parse_args()
    got = fetch_all(args.manifest, args.out_dir)
    print(f"fetched {len(got)} photos into {args.out_dir}")
    for g in got:
        print(f"  {g.name}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
