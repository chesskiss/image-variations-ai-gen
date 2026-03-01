from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True, frozen=True)
class SingleImageTransformSettings:
    fal_api_key: str
    fal_model_id: str
    output_directory: Path


def load_single_image_transform_settings() -> SingleImageTransformSettings:
    load_dotenv()

    fal_api_key = os.getenv("FAL_API_KEY", "").strip()
    if not fal_api_key:
        raise ValueError("FAL_API_KEY is required.")

    fal_model_id = os.getenv(
        "FAL_MODEL_ID",
        "fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
    ).strip()
    if not fal_model_id:
        raise ValueError("FAL_MODEL_ID is required.")

    output_directory = Path(os.getenv("OUTPUT_DIRECTORY", "./outputs")).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    return SingleImageTransformSettings(
        fal_api_key=fal_api_key,
        fal_model_id=fal_model_id,
        output_directory=output_directory,
    )
