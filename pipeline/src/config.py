from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the appraisal extractor pipeline."""

    dataset_dir: Path
    ocr_cache_dirname: str = "ocr-api-output"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    llm_model: str = "gpt-5-mini"


def default_config_from_path(dataset_dir: str | Path) -> PipelineConfig:
    return PipelineConfig(dataset_dir=Path(dataset_dir))
