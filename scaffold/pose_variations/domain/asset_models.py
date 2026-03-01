from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from pose_variations.domain.pose_preset_models import PosePresetIdentifier

AssetIdentifier = NewType("AssetIdentifier", str)
FalGenerationRequestIdentifier = NewType("FalGenerationRequestIdentifier", str)
GeneratedVideoUrl = NewType("GeneratedVideoUrl", str)


@dataclass(slots=True)
class UploadedImageAsset:
    asset_identifier: AssetIdentifier
    local_file_path: Path
    public_asset_url: str
    mime_type: str
    original_file_name: str


@dataclass(slots=True)
class GeneratedVideoAsset:
    asset_identifier: AssetIdentifier
    pose_preset_identifier: PosePresetIdentifier
    source_video_url: GeneratedVideoUrl
    local_file_path: Path
    public_asset_url: str


@dataclass(slots=True)
class ExtractedFrameAsset:
    asset_identifier: AssetIdentifier
    generated_video_asset_identifier: AssetIdentifier
    frame_timestamp_seconds: float
    local_file_path: Path
    public_asset_url: str
    sharpness_score: float | None = None
    perceptual_hash_value: str | None = None
    identity_similarity_score: float | None = None
