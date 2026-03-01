from __future__ import annotations

from pathlib import Path

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeScore,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_judge_stage.frame_selection_service import FrameSelectionService


def _candidate(frame_sequence_number: int, file_size_bytes: int) -> FrameCandidateForJudging:
    return FrameCandidateForJudging(
        generation_job_identifier=GenerationJobIdentifier("job-abc"),
        pose_preset_identifier=PosePresetIdentifier("rotate_left"),
        frame_sequence_number=frame_sequence_number,
        timestamp_seconds=frame_sequence_number / 2.0,
        local_frame_image_file_path=Path(f"/tmp/frame_{frame_sequence_number:04d}.jpg"),
        image_width_pixels=1024,
        image_height_pixels=1024,
        file_size_bytes=file_size_bytes,
        basic_quality_metrics={},
    )


def _score(frame_sequence_number: int, judge_score: float) -> FrameJudgeScore:
    return FrameJudgeScore(
        frame_sequence_number=frame_sequence_number,
        judge_score=judge_score,
        judge_confidence=0.8,
        judge_reasoning_summary=None,
        judge_model_name="rule-based-baseline",
        score_components={},
    )


def test_selection_ranking_is_deterministic_with_tie_breakers() -> None:
    frame_candidates_for_judging = [
        _candidate(frame_sequence_number=1, file_size_bytes=1000),
        _candidate(frame_sequence_number=2, file_size_bytes=1500),
        _candidate(frame_sequence_number=3, file_size_bytes=1500),
    ]
    frame_judge_scores = [
        _score(frame_sequence_number=1, judge_score=0.7),
        _score(frame_sequence_number=2, judge_score=0.9),
        _score(frame_sequence_number=3, judge_score=0.9),
    ]

    frame_selection_service = FrameSelectionService(selected_count=2, minimum_score_threshold=0.0)
    frame_judge_decision = frame_selection_service.select_best_frames_from_judge_scores(
        generation_job_identifier=GenerationJobIdentifier("job-abc"),
        pose_preset_identifier=PosePresetIdentifier("rotate_left"),
        frame_candidates_for_judging=frame_candidates_for_judging,
        frame_judge_scores=frame_judge_scores,
    )

    assert frame_judge_decision.selected_frame_sequence_numbers == [2, 3]
    assert [frame_judge_score.frame_sequence_number for frame_judge_score in frame_judge_decision.ranked_frame_scores] == [2, 3, 1]


def test_selection_returns_empty_decision_for_empty_input() -> None:
    frame_selection_service = FrameSelectionService(selected_count=2, minimum_score_threshold=0.0)
    frame_judge_decision = frame_selection_service.select_best_frames_from_judge_scores(
        generation_job_identifier=GenerationJobIdentifier("job-empty"),
        pose_preset_identifier=PosePresetIdentifier("rotate_right"),
        frame_candidates_for_judging=[],
        frame_judge_scores=[],
    )

    assert frame_judge_decision.selected_frame_sequence_numbers == []
    assert frame_judge_decision.ranked_frame_scores == []
    assert frame_judge_decision.selected_count == 0
