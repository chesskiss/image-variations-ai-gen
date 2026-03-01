from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

GenerationJobIdentifier = NewType("GenerationJobIdentifier", str)
PosePresetIdentifier = NewType("PosePresetIdentifier", str)


@dataclass(slots=True, frozen=True)
class FrameCandidateForJudging:
    generation_job_identifier: GenerationJobIdentifier
    pose_preset_identifier: PosePresetIdentifier
    frame_sequence_number: int
    timestamp_seconds: float
    local_frame_image_file_path: Path
    image_width_pixels: int | None
    image_height_pixels: int | None
    file_size_bytes: int
    basic_quality_metrics: dict[str, float | int | str | bool | None]


@dataclass(slots=True, frozen=True)
class FrameJudgeScore:
    frame_sequence_number: int
    judge_score: float
    judge_confidence: float | None
    judge_reasoning_summary: str | None
    judge_model_name: str
    score_components: dict[str, float]


@dataclass(slots=True, frozen=True)
class FrameJudgeDecision:
    generation_job_identifier: GenerationJobIdentifier
    pose_preset_identifier: PosePresetIdentifier
    selected_frame_sequence_numbers: list[int]
    ranked_frame_scores: list[FrameJudgeScore]
    selection_policy_name: str
    selected_count: int
