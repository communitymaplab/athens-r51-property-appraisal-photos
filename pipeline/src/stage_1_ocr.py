import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from config import PipelineConfig
from utils import get_ocr_json_path, load_ocr_json
from dotenv import load_dotenv
load_dotenv()


def _validate_crop_percents(
    ocr_crop_top_percent: float, ocr_crop_bottom_percent: float
) -> None:
    for name, value in (
        ("ocr_crop_top_percent", ocr_crop_top_percent),
        ("ocr_crop_bottom_percent", ocr_crop_bottom_percent),
    ):
        if value < 0 or value > 100:
            raise ValueError(f"{name} must be between 0 and 100")
    if ocr_crop_top_percent > 0 and ocr_crop_bottom_percent > 0:
        raise ValueError(
            "Only one of ocr_crop_top_percent or ocr_crop_bottom_percent can be > 0"
        )


def _prepare_ocr_image_bytes(
    image_path: Path, ocr_crop_top_percent: float, ocr_crop_bottom_percent: float
) -> bytes:
    _validate_crop_percents(ocr_crop_top_percent, ocr_crop_bottom_percent)

    if ocr_crop_top_percent <= 0 and ocr_crop_bottom_percent <= 0:
        return image_path.read_bytes()

    with Image.open(image_path) as im:
        w, h = im.size
        if h <= 1:
            return image_path.read_bytes()

        if ocr_crop_top_percent > 0:
            keep_h = max(1, int(round(h * (ocr_crop_top_percent / 100.0))))
            box = (0, 0, w, keep_h)
        else:
            keep_h = max(1, int(round(h * (ocr_crop_bottom_percent / 100.0))))
            box = (0, max(0, h - keep_h), w, h)

        cropped = im.crop(box)
        out = BytesIO()
        # PNG avoids extra jpeg-loss before OCR and is accepted by Azure DI.
        cropped.save(out, format="PNG")
        return out.getvalue()


def _run_azure_ocr(
    image_path: Path, ocr_crop_top_percent: float, ocr_crop_bottom_percent: float
) -> dict[str, Any]:
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise RuntimeError(
            "Azure OCR requested but Azure SDK is missing. Install "
            "'azure-ai-documentintelligence' and 'azure-core'."
        ) from exc

    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY are required for OCR cache misses."
        )

    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    image_bytes = _prepare_ocr_image_bytes(
        image_path, ocr_crop_top_percent, ocr_crop_bottom_percent
    )
    poller = client.begin_analyze_document(
        "prebuilt-read",
        body=image_bytes,
    )
    result = poller.result()
    return result.as_dict()


def get_cached_ocr(
    image_path: Path,
    config: PipelineConfig,
    *,
    ocr_crop_top_percent: float = 0.0,
    ocr_crop_bottom_percent: float = 0.0,
    force_no_ocr: bool = False,
) -> dict[str, Any] | None:
    """
    Stage 1: load cached OCR JSON for a page.
    On cache miss, call Azure Document Intelligence (`prebuilt-read`) and write
    the JSON into `ocr-api-output/`, unless ``force_no_ocr`` is True (then return
    None without calling Azure).
    """
    cached = load_ocr_json(image_path, config)
    if cached is not None:
        return cached
    if force_no_ocr:
        return None
    with Image.open(image_path) as img:
        width, height = img.size
    aspect_ratio = height / width
    file_size_kb = image_path.stat().st_size / 1024

    if abs(aspect_ratio - 1.00) <= 0.1 or file_size_kb < 650 or aspect_ratio < 1.0:
        #print(f"Skipping OCR for {image_path} because it is probably a loose photo")
        return None
    print(f"Running OCR for {image_path}")
    ocr_json = _run_azure_ocr(
        image_path,
        ocr_crop_top_percent=ocr_crop_top_percent,
        ocr_crop_bottom_percent=ocr_crop_bottom_percent,
    )
    json_path = get_ocr_json_path(image_path, config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(ocr_json, handle, ensure_ascii=False, indent=None, separators=(',', ':'))
    return ocr_json


def get_content_string(ocr_json: dict[str, Any] | None) -> str:
    if not ocr_json:
        return ""
    return str(ocr_json.get("content", ""))
