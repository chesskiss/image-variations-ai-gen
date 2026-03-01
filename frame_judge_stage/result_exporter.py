from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    FrameJudgeDecision,
)


class FrameJudgeResultExporter:
    def export_frame_judge_results(
        self,
        output_base_directory: Path,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        frame_judge_decision: FrameJudgeDecision,
    ) -> Path:
        output_base_directory.mkdir(parents=True, exist_ok=True)
        timestamp_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_output_directory = output_base_directory / f"frame_judge_results_{timestamp_label}"
        result_output_directory.mkdir(parents=True, exist_ok=True)

        selected_frame_sequence_number_set = set(frame_judge_decision.selected_frame_sequence_numbers)
        selected_frames_output_directory = result_output_directory / "selected_frames"
        selected_frames_output_directory.mkdir(parents=True, exist_ok=True)

        frame_candidate_by_sequence_number = {
            frame_candidate.frame_sequence_number: frame_candidate
            for frame_candidate in frame_candidates_for_judging
        }
        for selected_frame_sequence_number in selected_frame_sequence_number_set:
            selected_frame_candidate = frame_candidate_by_sequence_number.get(selected_frame_sequence_number)
            if selected_frame_candidate is None:
                continue
            destination_file_path = (
                selected_frames_output_directory
                / f"frame_{selected_frame_sequence_number:04d}{selected_frame_candidate.local_frame_image_file_path.suffix}"
            )
            shutil.copy2(
                selected_frame_candidate.local_frame_image_file_path,
                destination_file_path,
            )

        judge_decision_json_file_path = result_output_directory / "judge_decision.json"
        ranked_scores_json_file_path = result_output_directory / "ranked_scores.json"

        decision_payload = {
            "generation_job_identifier": str(frame_judge_decision.generation_job_identifier),
            "pose_preset_identifier": str(frame_judge_decision.pose_preset_identifier),
            "selection_policy_name": frame_judge_decision.selection_policy_name,
            "selected_count": frame_judge_decision.selected_count,
            "selected_frame_sequence_numbers": frame_judge_decision.selected_frame_sequence_numbers,
        }
        judge_decision_json_file_path.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")

        ranked_scores_payload = [asdict(frame_judge_score) for frame_judge_score in frame_judge_decision.ranked_frame_scores]
        ranked_scores_json_file_path.write_text(json.dumps(ranked_scores_payload, indent=2), encoding="utf-8")

        return result_output_directory
