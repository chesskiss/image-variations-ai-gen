from __future__ import annotations

from pathlib import Path

from frame_judge_stage.domain_models import FrameCandidateForJudging, FrameJudgeScore


class RuleBasedFrameJudge:
    def __init__(self, judge_model_name: str = "rule-based-baseline") -> None:
        self._judge_model_name = judge_model_name

    def judge_frame_candidates(
        self,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        original_image_file_path: Path | None = None,
    ) -> list[FrameJudgeScore]:
        if not frame_candidates_for_judging:
            return []

        maximum_file_size_bytes = max(
            frame_candidate_for_judging.file_size_bytes
            for frame_candidate_for_judging in frame_candidates_for_judging
        )
        maximum_file_size_bytes = max(maximum_file_size_bytes, 1)

        frame_judge_scores: list[FrameJudgeScore] = []
        for frame_candidate_for_judging in frame_candidates_for_judging:
            normalized_file_size_score = (
                frame_candidate_for_judging.file_size_bytes / maximum_file_size_bytes
            )
            center_timestamp_bonus = max(
                0.0,
                1.0 - abs(frame_candidate_for_judging.timestamp_seconds - 2.5) / 2.5,
            )
            combined_judge_score = min(
                1.0,
                max(0.0, normalized_file_size_score * 0.7 + center_timestamp_bonus * 0.3),
            )
            frame_judge_scores.append(
                FrameJudgeScore(
                    frame_sequence_number=frame_candidate_for_judging.frame_sequence_number,
                    judge_score=float(combined_judge_score),
                    judge_confidence=0.5,
                    judge_reasoning_summary="Rule-based baseline score from frame size and timestamp.",
                    judge_model_name=self._judge_model_name,
                    score_components={
                        "normalized_file_size_score": float(normalized_file_size_score),
                        "center_timestamp_bonus": float(center_timestamp_bonus),
                    },
                )
            )

        return frame_judge_scores
