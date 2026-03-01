from __future__ import annotations

from pathlib import Path

import pytest

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeScore,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_judge_stage.frame_selection_service import (
    FrameSelectionService,
    FrameSelectionServiceError,
)
from frame_judge_stage.rule_based_frame_judge import RuleBasedFrameJudge


def _build_frame_candidate_for_judging(
    frame_sequence_number: int,
    file_size_bytes: int,
    timestamp_seconds: float,
) -> FrameCandidateForJudging:
    return FrameCandidateForJudging(
        generation_job_identifier=GenerationJobIdentifier("job-1"),
        pose_preset_identifier=PosePresetIdentifier("rotate_right"),
        frame_sequence_number=frame_sequence_number,
        timestamp_seconds=timestamp_seconds,
        local_frame_image_file_path=Path(f"/tmp/frame_{frame_sequence_number:04d}.jpg"),
        image_width_pixels=512,
        image_height_pixels=512,
        file_size_bytes=file_size_bytes,
        basic_quality_metrics={},
    )


def test_frame_judge_contract_outputs_one_score_per_input_frame_candidate() -> None:
    frame_candidates_for_judging = [
        _build_frame_candidate_for_judging(
            frame_sequence_number=1, file_size_bytes=10_000, timestamp_seconds=1.0
        )
    ]

    rule_based_frame_judge = RuleBasedFrameJudge()
    frame_judge_scores = rule_based_frame_judge.judge_frame_candidates(frame_candidates_for_judging)

    assert len(frame_judge_scores) == len(frame_candidates_for_judging)
    assert frame_judge_scores[0].frame_sequence_number == 1
    assert 0.0 <= frame_judge_scores[0].judge_score <= 1.0


def test_frame_judge_contract_rejects_invalid_score_range() -> None:
    frame_candidates_for_judging = [
        _build_frame_candidate_for_judging(
            frame_sequence_number=1, file_size_bytes=1_000, timestamp_seconds=1.0
        )
    ]
    frame_judge_scores = [
        FrameJudgeScore(
            frame_sequence_number=1,
            judge_score=1.2,
            judge_confidence=0.9,
            judge_reasoning_summary="invalid",
            judge_model_name="bad-model",
            score_components={},
        )
    ]

    frame_selection_service = FrameSelectionService(selected_count=2, minimum_score_threshold=0.0)
    with pytest.raises(FrameSelectionServiceError):
        frame_selection_service.select_best_frames_from_judge_scores(
            generation_job_identifier=GenerationJobIdentifier("job-1"),
            pose_preset_identifier=PosePresetIdentifier("rotate_right"),
            frame_candidates_for_judging=frame_candidates_for_judging,
            frame_judge_scores=frame_judge_scores,
        )
