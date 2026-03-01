from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True, frozen=True)
class FrameJudgeStageSettings:
    enable_openai_frame_judge: bool
    frame_judge_model_name: str
    frame_judge_selected_count: int
    frame_judge_minimum_score_threshold: float
    frame_judge_timeout_seconds: int
    openai_api_key: str


def load_frame_judge_stage_settings() -> FrameJudgeStageSettings:
    load_dotenv()

    enable_openai_frame_judge = (
        os.getenv("ENABLE_OPENAI_FRAME_JUDGE", "false").strip().lower() == "true"
    )
    frame_judge_model_name = os.getenv("FRAME_JUDGE_MODEL_NAME", "o3").strip()
    frame_judge_selected_count = int(os.getenv("FRAME_JUDGE_SELECTED_COUNT", "2"))
    frame_judge_minimum_score_threshold = float(
        os.getenv("FRAME_JUDGE_MINIMUM_SCORE_THRESHOLD", "0.0")
    )
    frame_judge_timeout_seconds = int(os.getenv("FRAME_JUDGE_TIMEOUT_SECONDS", "45"))
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if frame_judge_selected_count < 0:
        raise ValueError("FRAME_JUDGE_SELECTED_COUNT must be zero or greater.")
    if frame_judge_minimum_score_threshold < 0.0 or frame_judge_minimum_score_threshold > 1.0:
        raise ValueError("FRAME_JUDGE_MINIMUM_SCORE_THRESHOLD must be between 0.0 and 1.0.")
    if frame_judge_timeout_seconds <= 0:
        raise ValueError("FRAME_JUDGE_TIMEOUT_SECONDS must be greater than zero.")

    return FrameJudgeStageSettings(
        enable_openai_frame_judge=enable_openai_frame_judge,
        frame_judge_model_name=frame_judge_model_name,
        frame_judge_selected_count=frame_judge_selected_count,
        frame_judge_minimum_score_threshold=frame_judge_minimum_score_threshold,
        frame_judge_timeout_seconds=frame_judge_timeout_seconds,
        openai_api_key=openai_api_key,
    )
