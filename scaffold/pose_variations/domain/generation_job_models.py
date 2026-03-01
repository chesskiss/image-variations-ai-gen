from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import NewType

from pose_variations.domain.asset_models import ExtractedFrameAsset, GeneratedVideoAsset, UploadedImageAsset
from pose_variations.domain.pose_preset_models import PosePresetIdentifier
from pose_variations.domain.scoring_models import FrameSelectionDecision

GenerationJobIdentifier = NewType("GenerationJobIdentifier", str)


class GenerationJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class GenerationJobProgressUpdate:
    stage_name: str
    status_message: str
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class GenerationJobRecord:
    generation_job_identifier: GenerationJobIdentifier
    generation_job_status: GenerationJobStatus
    selected_pose_preset_identifiers: list[PosePresetIdentifier]
    uploaded_image_asset: UploadedImageAsset
    generated_video_assets: list[GeneratedVideoAsset] = field(default_factory=list)
    extracted_frame_assets: list[ExtractedFrameAsset] = field(default_factory=list)
    frame_selection_decision: FrameSelectionDecision | None = None
    progress_updates: list[GenerationJobProgressUpdate] = field(default_factory=list)
    failure_reason: str | None = None
