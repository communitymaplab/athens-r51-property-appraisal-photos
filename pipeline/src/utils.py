import json
from pathlib import Path
from typing import Any, Iterable

from config import PipelineConfig


def iter_image_paths(config: PipelineConfig) -> Iterable[Path]:
    for path in sorted(config.dataset_dir.rglob("*")):
        if path.suffix.lower() in config.image_extensions and path.is_file():
            yield path


def get_ocr_json_path(image_path: Path, config: PipelineConfig) -> Path:
    return (
        image_path.parent
        / config.ocr_cache_dirname
        / image_path.with_suffix(".json").name
    )


def load_ocr_json(image_path: Path, config: PipelineConfig) -> dict[str, Any] | None:
    json_path = get_ocr_json_path(image_path, config)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
