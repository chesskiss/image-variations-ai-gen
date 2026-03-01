from __future__ import annotations

from pathlib import Path
from typing import Protocol

from frame_judge_stage.domain_models import FrameCandidateForJudging, FrameJudgeScore


class FrameJudge(Protocol):
    def judge_frame_candidates(
        self,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        original_image_file_path: Path | None = None,
    ) -> list[FrameJudgeScore]:
        """Return exactly one normalized score for each input frame candidate."""
