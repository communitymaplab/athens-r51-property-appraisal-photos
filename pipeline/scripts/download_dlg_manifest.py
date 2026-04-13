#!/usr/bin/env python3
"""
Download full-resolution JPEGs from a DLG / IIIF Presentation manifest.

Standalone utility (not used by the appraisal pipeline). Mirrors the prior R
workflow: fetch manifest.json, collect Image service base URLs, then GET each
``{service_id}/full/full/0/default.jpg?download=true``.

Example::

    python scripts/download_dlg_manifest.py \\
        --url 'https://...' \\
        --out-path ~/test/my_book
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()

import requests

USER_AGENT = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
            )


def _body_service_id(body: Any) -> str | None:
    if body is None:
        return None
    if isinstance(body, list):
        for b in body:
            sid = _body_service_id(b)
            if sid:
                return sid
        return None
    if not isinstance(body, dict):
        return None
    svc = body.get("service")
    if isinstance(svc, dict):
        return svc.get("id")
    if isinstance(svc, list):
        for s in svc:
            if isinstance(s, dict) and s.get("id"):
                return str(s["id"])
    return None


def _extract_image_service_ids(manifest: dict[str, Any]) -> list[str]:
    """
    Match R logic::

        my_json$items$items %>% map(~ .x$items %>% map_chr(~ .x$body$service$id))

    plus one extra level when ``items`` holds AnnotationPages (IIIF 3).
    """
    out: list[str] = []
    items = manifest.get("items")
    if items is None:
        return out

    # R ``my_json$items$items``: nested dict or list of nodes with ``items``
    if isinstance(items, dict) and "items" in items:
        outer = items["items"]
    elif isinstance(items, list):
        outer = items
    else:
        return out

    if not isinstance(outer, list):
        outer = [outer]

    for x in outer:
        if not isinstance(x, dict):
            continue
        inner = x.get("items") or []
        if not isinstance(inner, list):
            inner = [inner]
        for y in inner:
            if not isinstance(y, dict):
                continue
            sid = _body_service_id(y.get("body"))
            if sid:
                out.append(str(sid).rstrip("/"))
                continue
            # AnnotationPage -> annotations with body
            for z in y.get("items") or []:
                if isinstance(z, dict):
                    s2 = _body_service_id(z.get("body"))
                    if s2:
                        out.append(str(s2).rstrip("/"))
    return out


def manifest_url_from_document_url(document_url: str) -> str:
    base = document_url.split("?", 1)[0].rstrip("/")
    return f"{base}/presentation/manifest.json"


def download_dlg_document(
    url: str,
    *,
    out_path: Path,
    start: int = 0,
    end: int = 99999,
    session: requests.Session | None = None,
) -> None:
    save_path = out_path.expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    m_url = manifest_url_from_document_url(url)
    sess = session or requests.Session()

    r = sess.get(m_url, timeout=120)
    r.raise_for_status()
    my_json = r.json()

    items_root = my_json.get("items")
    if items_root is None:
        print("Warning: Couldn't find items array in manifest json", file=sys.stderr)
        return

    image_items = _extract_image_service_ids(my_json)
    if not image_items:
        print("Warning: No image service ids extracted from manifest", file=sys.stderr)
        return

    in_range = [i for i in range(1, len(image_items) + 1) if start <= i <= end]
    total_dl = len(in_range)
    j = 0
    for i in range(1, len(image_items) + 1):
        if i < start or i > end:
            continue
        service_id = image_items[i - 1]
        new_str = f"{service_id.rstrip('/')}/full/full/0/default.jpg?download=true"
        dest = save_path / f"{i}.jpg"
        j += 1
        with sess.get(new_str, stream=True, timeout=300) as img_resp:
            img_resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in img_resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded image {j}/{total_dl}")
    print(f"Downloaded {j} images to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IIIF full images from a DLG-style document URL."
    )
    parser.add_argument(
        "--url",
        required=False,
        help="Document page URL (query string stripped; manifest path appended).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=False,
        help="Directory to write numbered JPEGs (created if missing).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First 1-based manifest index to download (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=99999,
        help="Last 1-based manifest index to download (inclusive).",
    )
    args = parser.parse_args()

    BLOCK4_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_058-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B4",
        "start": 1,
        "end": 48,
    }

    BLOCK5_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_058-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B5",
        "start": 49,
        "end": 220,
    }

    BLOCK6_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_058-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B6",
        "start": 221,
        "end": 285,
    }

    BLOCK7_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-001",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B7",
        "start": 1,
        "end": 26,
    }

    BLOCK8_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-001",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B8",
        "start": 27,
        "end": 96,
    }

    BLOCK9_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-001",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B9",
        "start": 97,
        "end": 142,
    }

    BLOCK10_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-001",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B10",
        "start": 143,
        "end": 261,
    }

    BLOCK11_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B11",
        "start": 1,
        "end": 50,
    }

    BLOCK12_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B12",
        "start": 51,
        "end": 176,
    }
    
    BLOCK13_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B13",
        "start": 177,
        "end": 186,
    }

    BLOCK14_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B14",
        "start": 187,
        "end": 227,
    }

    BLOCK15_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B15",
        "start": 228,
        "end": 247,
    }

    BLOCK16_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-002",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B16",
        "start": 248,
        "end": 277,
    }

    BLOCK17_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B17",
        "start": 1,
        "end": 23,
    }

    BLOCK18_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B18",
        "start": 24,
        "end": 53,
    }

    BLOCK19_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B19",
        "start": 54,
        "end": 99,
    }

    BLOCK20_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B20",
        "start": 100,
        "end": 145,
    }

    BLOCK21_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B21",
        "start": 146,
        "end": 171,
    }

    BLOCK22_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B22",
        "start": 172,
        "end": 217,
    }

    BLOCK23_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-003",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B23",
        "start": 218,
        "end": 225,
    }

    BLOCK24_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-004",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B24",
        "start": 1,
        "end": 39,
    }

    BLOCK25_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-004",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B25",
        "start": 40,
        "end": 100,
    }

    BLOCK27_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-004",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B27",
        "start": 101,
        "end": 105,
    }

    BLOCK28_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-004",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B28",
        "start": 106,
        "end": 134,
    }

    BLOCK30_CONFIG = {
        "url": "https://dlg.usg.edu/record/guan_1633_059-004",
        "out_path": Path(os.environ.get("PHOTOS_DOWNLOAD_DIRECTORY")) / "B30",
        "start": 135,
        "end": 154,
    }

   
    # Manually download
    if not args.url or not args.out_path:
        current_config = BLOCK30_CONFIG
        print(f"No args passed, manually downloading {current_config['out_path'].name}")

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        download_dlg_document(**current_config, session=session)

    else:

        if not args.url.strip():
            parser.error("--url must be non-empty")

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        download_dlg_document(
            args.url,
            out_path=args.out_path,
            start=args.start,
            end=args.end,
            session=session,
        )


if __name__ == "__main__":
    main()
