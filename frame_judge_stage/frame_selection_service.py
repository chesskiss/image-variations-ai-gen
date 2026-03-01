from __future__ import annotations

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeDecision,
    FrameJudgeScore,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)


class FrameSelectionServiceError(RuntimeError):
    """Raised when frame judge scores violate the selection contract."""


class FrameSelectionService:
    def __init__(
        self,
        selected_count: int = 2,
        minimum_score_threshold: float = 0.0,
    ) -> None:
        if selected_count < 0:
            raise FrameSelectionServiceError("selected_count must be zero or greater.")
        if minimum_score_threshold < 0.0 or minimum_score_threshold > 1.0:
            raise FrameSelectionServiceError("minimum_score_threshold must be in [0,1].")

        self._selected_count = selected_count
        self._minimum_score_threshold = minimum_score_threshold

    def select_best_frames_from_judge_scores(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        pose_preset_identifier: PosePresetIdentifier,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        frame_judge_scores: list[FrameJudgeScore],
    ) -> FrameJudgeDecision:
        if not frame_candidates_for_judging and not frame_judge_scores:
            return FrameJudgeDecision(
                generation_job_identifier=generation_job_identifier,
                pose_preset_identifier=pose_preset_identifier,
                selected_frame_sequence_numbers=[],
                ranked_frame_scores=[],
                selection_policy_name="score_desc_file_size_desc_sequence_asc",
                selected_count=0,
            )

        self._validate_contract(frame_candidates_for_judging, frame_judge_scores)

        file_size_by_frame_sequence_number = {
            frame_candidate.frame_sequence_number: frame_candidate.file_size_bytes
            for frame_candidate in frame_candidates_for_judging
        }

        ranked_frame_scores = sorted(
            frame_judge_scores,
            key=lambda frame_judge_score: (
                -frame_judge_score.judge_score,
                -file_size_by_frame_sequence_number[frame_judge_score.frame_sequence_number],
                frame_judge_score.frame_sequence_number,
            ),
        )

        threshold_passing_frame_sequence_numbers = [
            frame_judge_score.frame_sequence_number
            for frame_judge_score in ranked_frame_scores
            if frame_judge_score.judge_score >= self._minimum_score_threshold
        ]
        selected_frame_sequence_numbers = threshold_passing_frame_sequence_numbers[: self._selected_count]

        return FrameJudgeDecision(
            generation_job_identifier=generation_job_identifier,
            pose_preset_identifier=pose_preset_identifier,
            selected_frame_sequence_numbers=selected_frame_sequence_numbers,
            ranked_frame_scores=ranked_frame_scores,
            selection_policy_name="score_desc_file_size_desc_sequence_asc",
            selected_count=len(selected_frame_sequence_numbers),
        )

    def _validate_contract(
        self,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        frame_judge_scores: list[FrameJudgeScore],
    ) -> None:
        frame_sequence_numbers_from_candidates = {
            frame_candidate.frame_sequence_number
            for frame_candidate in frame_candidates_for_judging
        }
        frame_sequence_numbers_from_scores = {
            frame_judge_score.frame_sequence_number for frame_judge_score in frame_judge_scores
        }

        if len(frame_candidates_for_judging) != len(frame_judge_scores):
            raise FrameSelectionServiceError(
                "Frame judge output must include exactly one score per input frame candidate."
            )

        if frame_sequence_numbers_from_candidates != frame_sequence_numbers_from_scores:
            raise FrameSelectionServiceError(
                "Frame judge output frame sequence numbers do not match input frame candidates."
            )

        for frame_judge_score in frame_judge_scores:
            if frame_judge_score.judge_score < 0.0 or frame_judge_score.judge_score > 1.0:
                raise FrameSelectionServiceError(
                    "Frame judge score must be normalized to the [0,1] range."
                )
            if frame_judge_score.judge_confidence is not None and (
                frame_judge_score.judge_confidence < 0.0
                or frame_judge_score.judge_confidence > 1.0
            ):
                raise FrameSelectionServiceError(
                    "Frame judge confidence must be normalized to the [0,1] range."
                )
