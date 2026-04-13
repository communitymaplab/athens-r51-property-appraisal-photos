from __future__ import annotations

import base64
import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from stage_3_claude_batch import load_page_cache, save_page_cache

# Max edge length for saved Roboflow visualization PNGs (aspect ratio preserved).
_VISUALIZATION_THUMBNAIL_MAX = (2000, 2000)


def _roboflow_input_hash_legacy(
    image_path: Path,
    workspace_name: str,
    workflow_id: str,
    api_url: str,
) -> str:
    """Invalidate on absolute path + mtime + size (legacy; path changes break cache hits)."""
    st = image_path.stat()
    payload = (
        f"{api_url}\n{workspace_name}\n{workflow_id}\n"
        f"{image_path.resolve()}\n{st.st_mtime_ns}\n{st.st_size}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _roboflow_input_hash_for_dataset_image(
    image_path: Path,
    workspace_name: str,
    workflow_id: str,
    api_url: str,
    logical_image_key: str,
) -> str:
    """
    Stable across ``--dataset-dir``: ``logical_image_key`` is ``{block}/{file}`` plus
    SHA-256 of file bytes (same identity as the cache filename stem).
    """
    h = hashlib.sha256()
    with image_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    payload = (
        f"{api_url}\n{workspace_name}\n{workflow_id}\n"
        f"{logical_image_key}\n{digest}\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_visualization_to_png_bytes(viz: Any) -> bytes | None:
    """Turn Roboflow `visualization` field into raw image bytes (usually PNG)."""
    if viz is None or viz == "removed":
        return None
    if isinstance(viz, (bytes, bytearray)):
        return bytes(viz)
    if isinstance(viz, str):
        s = viz.strip()
        if s.startswith("data:"):
            comma = s.find(",")
            if comma >= 0:
                s = s[comma + 1 :]
        try:
            return base64.b64decode(s, validate=False)
        except (ValueError, TypeError):
            return None
    if isinstance(viz, dict):
        vtype = str(viz.get("type", "")).lower()
        val = viz.get("value")
        if val is None:
            val = viz.get("data") or viz.get("image") or viz.get("base64")
        if isinstance(val, list) and val and isinstance(val[0], int):
            try:
                return bytes(val)
            except (ValueError, TypeError):
                return None
        if isinstance(val, list) and val and isinstance(val[0], str):
            val = "".join(val)
        if isinstance(val, str):
            return _decode_visualization_to_png_bytes(val)
        if vtype == "base64" and isinstance(val, str):
            return _decode_visualization_to_png_bytes(val)
    return None


def _save_workflow_visualization_pngs(
    raw_result: Any,
    viz_dir: Path,
    file_stem: str,
) -> list[str]:
    """
    Write each step's visualization to viz_dir, thumbnailed to fit within 2000×2000
    (Pillow ``Image.thumbnail``, aspect ratio preserved). Smaller images are unchanged.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    roots: list[Any]
    if isinstance(raw_result, dict):
        roots = [raw_result]
    elif isinstance(raw_result, list):
        roots = raw_result
    else:
        return saved

    part = 0
    for root in roots:
        if not isinstance(root, dict):
            continue
        viz = root.get("visualization")
        data = _decode_visualization_to_png_bytes(viz)
        if not data:
            continue
        try:
            with Image.open(BytesIO(data)) as im:
                im.load()
                im.thumbnail(_VISUALIZATION_THUMBNAIL_MAX)
                out = BytesIO()
                im.save(out, format="JPEG", optimize=True)
                scaled = out.getvalue()
        except OSError:
            continue
        filename = f"{file_stem}.jpg" if part == 0 else f"{file_stem}__viz{part}.jpg"
        part += 1
        out_path = viz_dir / filename
        out_path.write_bytes(scaled)
        roboflow_root = viz_dir.parent
        try:
            rel = str(out_path.relative_to(roboflow_root))
        except ValueError:
            rel = str(out_path)
        saved.append(rel)
    return saved


def _drop_visualization(node: Any) -> Any:
    """Remove visualization payloads from workflow JSON before caching."""
    if isinstance(node, list):
        return [_drop_visualization(x) for x in node]
    if isinstance(node, dict):
        return {
            k: _drop_visualization(v)
            for k, v in node.items()
            if k != "visualization"
        }
    return node


def _normalize_prediction(
    pred: dict[str, Any],
    *,
    source_image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    One detection from workflow `predictions.predictions[]` (bbox center x,y in image pixels).
    Optional: points[], rle_mask as {size, counts} or other SDK shape.
    """
    return {
        "detector": "roboflow_workflow",
        "class": pred.get("class"),
        "class_id": pred.get("class_id"),
        "confidence": pred.get("confidence"),
        "detection_id": pred.get("detection_id"),
        "parent_id": pred.get("parent_id"),
        "source_image": source_image,
        "bbox": {
            "x": pred.get("x"),
            "y": pred.get("y"),
            "width": pred.get("width"),
            "height": pred.get("height"),
        },
        "points": pred.get("points"),
        "rle_mask": pred.get("rle_mask"),
    }


def _extract_predictions_from_workflow_result(result: Any) -> list[dict[str, Any]]:
    """
    Roboflow workflow shape (typical):

        [
          {
            "predictions": {
              "image": {"width": int, "height": int},
              "predictions": [
                {
                  "width", "height", "x", "y", "confidence",
                  "class_id", "class", "detection_id", "parent_id",
                  optional "points", "rle_mask"
                },
                ...
              ]
            },
            "visualization": ...   # saved as PNG under roboflow_workflow/visualizations/, then stripped in JSON
          },
          ...
        ]

    Also accepts a single dict instead of a one-element list.
    """
    found: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    if isinstance(result, dict):
        roots: list[Any] = [result]
    elif isinstance(result, list):
        roots = result
    else:
        roots = []

    for root in roots:
        if not isinstance(root, dict):
            continue
        block = root.get("predictions")
        if not isinstance(block, dict):
            continue
        frame = block.get("image")
        source_image = frame if isinstance(frame, dict) else None
        inner = block.get("predictions")
        if not isinstance(inner, list):
            continue
        bbox_keys = ("width", "height", "x", "y")
        for item in inner:
            if not isinstance(item, dict):
                continue
            if not all(k in item and item.get(k) is not None for k in bbox_keys):
                continue
            found.append((item, source_image))

    return [_normalize_prediction(p, source_image=img) for p, img in found]


def detect_photo_regions(
    image_path: Path,
    page_type: str,
    *,
    dataset_name: str | None = None,
    cache_file: Path | None = None,
    workspace_name: str | None = None,
    workflow_id: str | None = None,
    api_url: str | None = None,
    cache_stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Stage 4: Roboflow workflow photo detection with per-image JSON cache.
    Loose-photo pages skip Stage 4; Stage 5 copies the original file.

    When ``dataset_name`` is set, cache validity uses block folder + filename (not
    ``--dataset-dir``) plus a hash of image bytes, so the same scan reused under
    another directory still hits cache. Cache files are still per ``--dataset-name``
    via the ``roboflow_workflow`` directory layout.

    Expects `run_workflow` output: list of steps, each with
    `predictions` -> { `image`, `predictions` -> [detections] }; see
    `_extract_predictions_from_workflow_result` docstring.

    Environment variables (defaults in parentheses):
    - ROBOFLOW_API_KEY (required for API unless cache hit)
    - ROBOFLOW_WORKSPACE (liams-workspace-r51)
    - ROBOFLOW_WORKFLOW_ID (find-photos)
    - ROBOFLOW_API_URL (https://serverless.roboflow.com)
    """
    workspace_name = workspace_name or os.environ.get(
        "ROBOFLOW_WORKSPACE", "liams-workspace-r51"
    )
    workflow_id = workflow_id or os.environ.get("ROBOFLOW_WORKFLOW_ID", "find-photos")
    api_url = api_url or os.environ.get(
        "ROBOFLOW_API_URL", "https://serverless.roboflow.com"
    )
    if dataset_name is not None:
        logical_image_key = f"{image_path.parent.name}/{image_path.name}"
        input_hash = _roboflow_input_hash_for_dataset_image(
            image_path, workspace_name, workflow_id, api_url, logical_image_key
        )
        # Mock parent directory to match hash
        if dataset_name == "Bradberry appraisals 1962":
            image_path_mock = Path(f"/Users/liam/Documents/R-51 appraisal reports_files/Bradberry appraisals 1962/{image_path.parent.name}/{image_path.name}")
            input_hash_fallback = _roboflow_input_hash_legacy(
                image_path_mock, workspace_name, workflow_id, api_url
            )
        else:
            input_hash_fallback = None
    else:
        logical_image_key = str(image_path.resolve())
        input_hash = _roboflow_input_hash_legacy(
            image_path, workspace_name, workflow_id, api_url
        )
        input_hash_fallback = None

    if cache_file is not None:
        cached = load_page_cache(cache_file)
        if cached and (cached.get("input_hash") == input_hash or (input_hash_fallback is not None and cached.get("input_hash") == input_hash_fallback)):
            regions = cached.get("regions")
            if isinstance(regions, list):
                if cache_stats is not None:
                    cache_stats["roboflow_cache_hits"] = (
                        cache_stats.get("roboflow_cache_hits", 0) + 1
                    )
                return regions

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        if cache_stats is not None:
            cache_stats["roboflow_skipped_no_key"] = (
                cache_stats.get("roboflow_skipped_no_key", 0) + 1
            )
        return [
            {
                "detector": "roboflow_missing_api_key",
                "bbox": None,
                "confidence": 0.0,
                "image_path": str(image_path),
                "page_type": page_type,
            }
        ]

    cache_key = cache_file.stem if cache_file is not None else image_path.name
    print(f"Running roboflow on image [{cache_key}]")

    from inference_sdk import InferenceHTTPClient

    if cache_stats is not None:
        cache_stats["roboflow_api_calls"] = cache_stats.get("roboflow_api_calls", 0) + 1

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    raw_result = client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={"image": str(image_path.resolve())},
        use_cache=True,
    )

    regions = _extract_predictions_from_workflow_result(raw_result)
    for r in regions:
        r.setdefault("image_path", str(image_path))
        r.setdefault("page_type", page_type)

    visualization_files: list[str] = []
    if cache_file is not None:
        viz_dir = cache_file.parent / "visualizations"
        visualization_files = _save_workflow_visualization_pngs(
            raw_result, viz_dir, cache_file.stem
        )

    payload = {
        "input_hash": input_hash,
        "workspace_name": workspace_name,
        "workflow_id": workflow_id,
        "api_url": api_url,
        "logical_image_key": logical_image_key,
        "image_path": str(image_path.resolve()),
        "relative_path_hint": str(image_path),
        "page_type": page_type,
        "visualization_pngs": visualization_files,
        "raw_result": _drop_visualization(raw_result),
        "regions": regions,
    }
    if cache_file is not None:
        save_page_cache(cache_file, payload)

    return regions
