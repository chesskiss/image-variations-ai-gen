from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NewType

GenerationJobIdentifier = NewType("GenerationJobIdentifier", str)
PosePresetIdentifier = NewType("PosePresetIdentifier", str)


@dataclass(slots=True, frozen=True)
class GeneratedVideoAsset:
    generation_job_identifier: GenerationJobIdentifier
    pose_preset_identifier: PosePresetIdentifier
    local_video_file_path: Path
    video_duration_seconds: float | None
    video_file_format: str


@dataclass(slots=True, frozen=True)
class ExtractedFrameAsset:
    generation_job_identifier: GenerationJobIdentifier
    pose_preset_identifier: PosePresetIdentifier
    source_video_local_file_path: Path
    frame_sequence_number: int
    timestamp_seconds: float
    local_frame_image_file_path: Path
    image_file_format: str
    image_width_pixels: int | None
    image_height_pixels: int | None
    file_size_bytes: int
    basic_quality_metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
