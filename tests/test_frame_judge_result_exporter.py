from __future__ import annotations

import json
from pathlib import Path

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeDecision,
    FrameJudgeScore,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_judge_stage.result_exporter import FrameJudgeResultExporter


def test_frame_judge_result_exporter_writes_decision_and_selected_frames(tmp_path: Path) -> None:
    frame_one_file_path = tmp_path / "frame_0001.jpg"
    frame_two_file_path = tmp_path / "frame_0002.jpg"
    frame_one_file_path.write_bytes(b"frame-one")
    frame_two_file_path.write_bytes(b"frame-two")

    frame_candidates_for_judging = [
        FrameCandidateForJudging(
            generation_job_identifier=GenerationJobIdentifier("job-1"),
            pose_preset_identifier=PosePresetIdentifier("preset-1"),
            frame_sequence_number=1,
            timestamp_seconds=0.5,
            local_frame_image_file_path=frame_one_file_path,
            image_width_pixels=100,
            image_height_pixels=100,
            file_size_bytes=frame_one_file_path.stat().st_size,
            basic_quality_metrics={},
        ),
        FrameCandidateForJudging(
            generation_job_identifier=GenerationJobIdentifier("job-1"),
            pose_preset_identifier=PosePresetIdentifier("preset-1"),
            frame_sequence_number=2,
            timestamp_seconds=1.0,
            local_frame_image_file_path=frame_two_file_path,
            image_width_pixels=100,
            image_height_pixels=100,
            file_size_bytes=frame_two_file_path.stat().st_size,
            basic_quality_metrics={},
        ),
    ]

    frame_judge_decision = FrameJudgeDecision(
        generation_job_identifier=GenerationJobIdentifier("job-1"),
        pose_preset_identifier=PosePresetIdentifier("preset-1"),
        selected_frame_sequence_numbers=[2],
        ranked_frame_scores=[
            FrameJudgeScore(
                frame_sequence_number=2,
                judge_score=0.9,
                judge_confidence=0.8,
                judge_reasoning_summary=None,
                judge_model_name="rule-based-baseline",
                score_components={},
            ),
            FrameJudgeScore(
                frame_sequence_number=1,
                judge_score=0.7,
                judge_confidence=0.8,
                judge_reasoning_summary=None,
                judge_model_name="rule-based-baseline",
                score_components={},
            ),
        ],
        selection_policy_name="score_desc_file_size_desc_sequence_asc",
        selected_count=1,
    )

    frame_judge_result_exporter = FrameJudgeResultExporter()
    result_output_directory = frame_judge_result_exporter.export_frame_judge_results(
        output_base_directory=tmp_path / "outputs",
        frame_candidates_for_judging=frame_candidates_for_judging,
        frame_judge_decision=frame_judge_decision,
    )

    judge_decision_json_file_path = result_output_directory / "judge_decision.json"
    ranked_scores_json_file_path = result_output_directory / "ranked_scores.json"
    selected_frame_file_path = result_output_directory / "selected_frames" / "frame_0002.jpg"

    assert judge_decision_json_file_path.exists()
    assert ranked_scores_json_file_path.exists()
    assert selected_frame_file_path.exists()

    decision_payload = json.loads(judge_decision_json_file_path.read_text(encoding="utf-8"))
    assert decision_payload["selected_frame_sequence_numbers"] == [2]
