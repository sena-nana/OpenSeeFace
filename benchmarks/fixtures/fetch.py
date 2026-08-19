#!/usr/bin/env python3
"""Download CC/PD Wikimedia portraits listed in manifest.json into cache/.

Wikimedia requires a descriptive User-Agent. Cache is gitignored.
Tight crops with webcam_pad=true are letterboxed onto 640x480 so RetinaFace
sees a webcam-sized face.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
MANIFEST = ROOT / "manifest.json"
UA = {
    "User-Agent": "OpenSeeFace-bench/0.1 (https://github.com/emilianavt/OpenSeeFace; glasses fixtures)"
}


def wiki_info(title: str) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": "800",
            "titles": title,
        }
    )
    req = urllib.request.Request(url, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30))
    page = next(iter(data["query"]["pages"].values()))
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        raise SystemExit(f"missing on Commons: {title}")
    return info


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url.split("?")[0], headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        blob = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if not blob or "html" in ctype.lower():
        raise SystemExit(f"bad download {url} ctype={ctype} n={len(blob)}")
    return blob


def webcam_pad(path: Path, margin: float = 0.55) -> None:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    px, py = int(w * margin), int(h * margin)
    bg = Image.new("RGB", (w + 2 * px, h + 2 * py), (48, 52, 58))
    bg.paste(im, (px, py))
    bg.thumbnail((560, 400), Image.Resampling.LANCZOS)
    cam = Image.new("RGB", (640, 480), (48, 52, 58))
    bw, bh = bg.size
    cam.paste(bg, ((640 - bw) // 2, (480 - bh) // 2))
    cam.save(path, quality=92)


def main() -> None:
    photos = json.loads(MANIFEST.read_text())["photos"]
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, photo in enumerate(photos):
        dest = CACHE / photo["file"]
        title = photo.get("commons") or f"File:{photo['file']}"
        info = wiki_info(title)
        url = info.get("thumburl") or info["url"]
        print(f"{photo['id']}: {info['width']}x{info['height']} -> {dest.name}")
        dest.write_bytes(fetch(url))
        if photo.get("webcam_pad"):
            webcam_pad(dest)
            print("  webcam_pad 640x480")
        if i + 1 < len(photos):
            time.sleep(0.4)


if __name__ == "__main__":
    main()
