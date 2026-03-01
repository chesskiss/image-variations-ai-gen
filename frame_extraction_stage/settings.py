from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True, frozen=True)
class FrameExtractionStageSettings:
    storage_directory: Path
    ffmpeg_executable_path: str
    extracted_frames_per_second: int
    extracted_frame_image_format: str
    max_extracted_frames_per_video: int


def load_frame_extraction_stage_settings() -> FrameExtractionStageSettings:
    load_dotenv()

    storage_directory = Path(os.getenv("STORAGE_DIRECTORY", "./storage")).resolve()
    storage_directory.mkdir(parents=True, exist_ok=True)

    ffmpeg_executable_path = os.getenv("FFMPEG_EXECUTABLE_PATH", "ffmpeg").strip()
    extracted_frames_per_second = int(os.getenv("EXTRACTED_FRAMES_PER_SECOND", "2"))
    extracted_frame_image_format = os.getenv("EXTRACTED_FRAME_IMAGE_FORMAT", "jpg").strip().lower()
    max_extracted_frames_per_video = int(os.getenv("MAX_EXTRACTED_FRAMES_PER_VIDEO", "24"))

    return FrameExtractionStageSettings(
        storage_directory=storage_directory,
        ffmpeg_executable_path=ffmpeg_executable_path,
        extracted_frames_per_second=extracted_frames_per_second,
        extracted_frame_image_format=extracted_frame_image_format,
        max_extracted_frames_per_video=max_extracted_frames_per_video,
    )
