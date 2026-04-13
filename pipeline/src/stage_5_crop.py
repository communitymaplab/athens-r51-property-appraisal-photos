from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

# Tolerance for nested-box checks (Roboflow floating-point + rounding noise)
_NEST_EPS = 0.5


def _bbox_to_rect(
    bbox: dict[str, Any] | None,
    margin: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """
    Convert Roboflow center (x,y) + (width,height) → (left, top, right, bottom).

    ``margin`` expands the rectangle by that many pixels on each side (left/up
    shrink, right/down grow). Clamping to the image happens in ``_bbox_to_pil_crop_box``.
    """
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["width"])
        h = float(bbox["height"])
        m = float(margin)
    except (KeyError, TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    left = x - w / 2.0 - m
    top = y - h / 2.0 - m
    right = x + w / 2.0 + m
    bottom = y + h / 2.0 + m
    return (left, top, right, bottom)


def _is_strictly_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    eps: float = _NEST_EPS,
) -> bool:
    """Return True if inner is wholly inside outer and not identical."""
    il, it, ir, ib = inner
    ol, ot, or_, ob = outer

    # Inside with tolerance
    if not (ol - eps <= il and ot - eps <= it and ir <= or_ + eps and ib <= ob + eps):
        return False

    # Not the same box (with tolerance)
    if (
        abs(il - ol) < eps
        and abs(it - ot) < eps
        and abs(ir - or_) < eps
        and abs(ib - ob) < eps
    ):
        return False

    return True


def drop_nested_detection_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove regions whose bbox is strictly nested inside another."""
    # Build rects once, keep original regions
    rects = [
        _bbox_to_rect(r.get("bbox") if isinstance(r, dict) else None, margin=0.0)
        for r in regions
    ]

    remove: set[int] = set()
    n = len(regions)

    for i in range(n):
        ri = rects[i]
        if ri is None:
            continue
        for j in range(n):
            if i == j or rects[j] is None:
                continue
            if _is_strictly_inside(ri, rects[j]):
                remove.add(i)
                break

    return [regions[k] for k in range(n) if k not in remove]


def _bbox_to_pil_crop_box(
    bbox: dict[str, Any],
    img_w: int,
    img_h: int,
    margin: float = 0.0,
) -> tuple[int, int, int, int] | None:
    """Convert bbox to PIL crop box (left, top, right, bottom) with clamping."""
    rect = _bbox_to_rect(bbox, margin=margin)
    if rect is None:
        return None

    left, top, right, bottom = rect

    l = max(0, min(img_w, math.floor(left)))
    t = max(0, min(img_h, math.floor(top)))
    r = max(l + 1, min(img_w, math.ceil(right)))
    b = max(t + 1, min(img_h, math.ceil(bottom)))

    return (l, t, r, b)


_PARCEL_SPLIT_RE = re.compile(
    r"\s*(?:,|&|\band\b)\s*",
    re.IGNORECASE,
)


def _normalize_parcel_token(token: str) -> str:
    t = token.strip()
    t = re.sub(r"\s+", "", t)
    if not t:
        return ""
    t = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", t)
    return t


def _split_parcel_field(parcel: Any) -> list[str]:
    """
    ``"2 & 3"``, ``"1, 2 & 3"`` → ``["2", "3"]``, ``["1", "2", "3"]``.
    Single parcel unchanged.
    """
    if parcel is None:
        return []
    s = str(parcel).strip()
    if not s:
        return []
    raw = [p for p in _PARCEL_SPLIT_RE.split(s) if p.strip()]
    out: list[str] = []
    for p in raw:
        n = _normalize_parcel_token(p)
        if n:
            out.append(n)
    return out


def _filename_core_from_block_parcel(block: str, parcel: Any) -> str:
    """
    Stem before ``__photo{n}.jpg``: ``B18_P5`` or ``B18_P2P3`` (several parcels).
    """
    parts = _split_parcel_field(parcel)
    if not parts:
        raise ValueError("extracted_id parcel is empty or unparsable")
    p_seg = "".join(f"P{p}" for p in parts)
    return f"B{block}_{p_seg}"


def _allocate_photo_dest(
    out_dir: Path,
    filename_core: str,
    preferred_index: int,
) -> Path:
    """
    ``{core}__photo{n}.jpg`` — always two underscores before ``photo``. First
    ``n >= preferred_index`` with no existing file.
    """
    n = preferred_index
    while True:
        dest = out_dir / f"{filename_core}__photo{n}.jpg"
        if not dest.exists():
            return dest
        n += 1


def crop_and_save_regions(
    image_path: Path,
    regions: list[dict[str, Any]],
    *,
    output_dir: Path,
    cache_key: str,
    extracted_id: dict[str, Any] | None = None,
    block_folder_name: str = "unknown_block",
    bbox_margin: float = 0.0,
    classification: str | None = None,
) -> list[Path]:
    """
    Crop detected regions and save as JPEGs under processed/<block_folder>/.
    Drops nested duplicate regions (inner bbox inside outer).

    For ``loose_photo``, the original file is copied (same bytes) — no Stage 4 regions
    and no decode/re-encode.

    Filenames: ``B{block}_P{p}`` or ``B{block}_P{p1}P{p2}…`` (split parcel strings like
    ``"2 & 3"``), then always ``__photo{n}.jpg`` (two underscores before ``photo``)
    so R can slice between the first ``_`` after the block number and ``__``.

    Per-page region order picks a preferred ``n``; if that path exists (e.g. another
    photo sheet for the same B/P), ``n`` is increased until free.

    ``extracted_id`` supplies block and parcel when present.

    ``bbox_margin``: expand each crop by this many pixels on every side before
    clamping to the image (nested dedupe still uses margin=0).
    """
    if extracted_id is not None:
        block = extracted_id.get("block")
        parcel = extracted_id.get("parcel")
        if block and parcel:
            filename_core = _filename_core_from_block_parcel(
                str(block).strip(), parcel
            )
        else:
            raise ValueError("extracted_id must contain both block and parcel")
    else:
        filename_core = cache_key

    out_dir = output_dir / block_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    if classification == "loose_photo":
        dest = _allocate_photo_dest(out_dir, filename_core, 1)
        shutil.copy2(image_path, dest)
        saved.append(dest)
    else:
        regions = drop_nested_detection_regions(regions)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size

            for i, region in enumerate(regions, 1):
                if not isinstance(region, dict):
                    continue

                if region.get("detector") == "roboflow_missing_api_key":
                    continue

                bbox = region.get("bbox")
                if not isinstance(bbox, dict):
                    continue

                box = _bbox_to_pil_crop_box(bbox, w, h, margin=bbox_margin)
                if box is None:
                    continue

                crop = im.crop(box)
                dest = _allocate_photo_dest(out_dir, filename_core, i)
                crop.save(dest, format="JPEG", quality=95)
                saved.append(dest)

    n = len(saved)
    if n == 0:
        print(f"[WARN] Stage 5: [{cache_key}] produced no saved crops")
    elif classification == "loose_photo" and n == 1:
        # print(f"Stage 5: copied original for [{cache_key}] (loose_photo)")
        pass
    elif n == 1:
        # print(f"Stage 5: saved and cropped [{cache_key}] into 1 photo")
        pass
    else:
        print(f"[WARN] Stage 5: saved and cropped [{cache_key}] into {n} photos")

    return saved