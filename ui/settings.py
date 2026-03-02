from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True, frozen=True)
class UiSettings:
    enable_result_cache: bool
    cache_index_directory: Path
    cache_max_entries: int
    cache_retention_days: int
    cache_key_include_model_version: bool
    fal_model_id: str
    enable_openai_frame_judge: bool
    frame_judge_model_name: str
    frame_judge_minimum_score_threshold: float
    extracted_frames_per_second: int
    extracted_frame_image_format: str
    max_extracted_frames_per_video: int


def _to_bool(environment_variable_value: str, default: bool) -> bool:
    normalized_value = environment_variable_value.strip().lower()
    if not normalized_value:
        return default
    return normalized_value == "true"


def load_ui_settings() -> UiSettings:
    load_dotenv()

    cache_index_directory = (
        Path(os.getenv("CACHE_INDEX_DIRECTORY", "./outputs/cache_index")).expanduser().resolve()
    )
    cache_index_directory.mkdir(parents=True, exist_ok=True)

    return UiSettings(
        enable_result_cache=_to_bool(os.getenv("ENABLE_RESULT_CACHE", "true"), default=True),
        cache_index_directory=cache_index_directory,
        cache_max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
        cache_retention_days=int(os.getenv("CACHE_RETENTION_DAYS", "30")),
        cache_key_include_model_version=_to_bool(
            os.getenv("CACHE_KEY_INCLUDE_MODEL_VERSION", "true"), default=True
        ),
        fal_model_id=os.getenv(
            "FAL_MODEL_ID",
            "fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
        ).strip(),
        enable_openai_frame_judge=_to_bool(
            os.getenv("ENABLE_OPENAI_FRAME_JUDGE", "false"), default=False
        ),
        frame_judge_model_name=os.getenv("FRAME_JUDGE_MODEL_NAME", "o3").strip(),
        frame_judge_minimum_score_threshold=float(
            os.getenv("FRAME_JUDGE_MINIMUM_SCORE_THRESHOLD", "0.0")
        ),
        extracted_frames_per_second=int(os.getenv("EXTRACTED_FRAMES_PER_SECOND", "2")),
        extracted_frame_image_format=os.getenv("EXTRACTED_FRAME_IMAGE_FORMAT", "jpg").strip(),
        max_extracted_frames_per_video=int(os.getenv("MAX_EXTRACTED_FRAMES_PER_VIDEO", "24")),
    )
