from dataclasses import dataclass
from pathlib import Path
from typing import Any


PageType = str


@dataclass
class PageArtifact:
    """Intermediate representation for a scanned page."""

    image_path: Path
    ocr_json: dict[str, Any] | None
    ocr_content: str
    classification: PageType
    region_text: str | None = None
    extracted_id: dict[str, Any] | None = None
    block_number: str | None = None  # Block number as given by directory containing the photo
    photo_regions: list[dict[str, Any]] | None = None  # Stage 4 detection output per page
    photo_crop_paths: list[str] | None = None  # Stage 5 paths relative to processed/
