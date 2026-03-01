from __future__ import annotations

import os
from pathlib import Path

import pytest

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_judge_stage.openai_vision_frame_judge import OpenAiVisionFrameJudge


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_FRAME_JUDGE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Set RUN_OPENAI_FRAME_JUDGE_INTEGRATION_TESTS=true to run real OpenAI judge integration test.",
)
def test_openai_vision_frame_judge_with_fixture_frame() -> None:
    pass


"""
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for OpenAI frame judge integration test.")

    fixture_frame_image_file_path = Path("tests/fixtures/frame_fixture.jpg")
    if not fixture_frame_image_file_path.exists():
        pytest.skip("Add tests/fixtures/frame_fixture.jpg to run OpenAI frame judge integration test.")

    frame_candidate_for_judging = FrameCandidateForJudging(
        generation_job_identifier=GenerationJobIdentifier("job-int"),
        pose_preset_identifier=PosePresetIdentifier("rotate_right"),
        frame_sequence_number=1,
        timestamp_seconds=1.0,
        local_frame_image_file_path=fixture_frame_image_file_path,
        image_width_pixels=None,
        image_height_pixels=None,
        file_size_bytes=fixture_frame_image_file_path.stat().st_size,
        basic_quality_metrics={},
    )

    openai_vision_frame_judge = OpenAiVisionFrameJudge(
        openai_api_key=openai_api_key,
        judge_model_name=os.getenv("FRAME_JUDGE_MODEL_NAME", "o3"),
        timeout_seconds=int(os.getenv("FRAME_JUDGE_TIMEOUT_SECONDS", "45")),
    )
    frame_judge_scores = openai_vision_frame_judge.judge_frame_candidates(
        [frame_candidate_for_judging],
        original_image_file_path=fixture_frame_image_file_path,
    )

    assert len(frame_judge_scores) == 1
    assert 0.0 <= frame_judge_scores[0].judge_score <= 1.0
"""  # TODO - Add real OpenAI judge integration test with fixture frame.
