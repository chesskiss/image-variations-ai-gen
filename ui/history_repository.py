from __future__ import annotations

import json
from pathlib import Path


class HistoryRepository:
    def __init__(self, orchestrator_job_status_directory: Path) -> None:
        self._orchestrator_job_status_directory = orchestrator_job_status_directory

    def list_history_entries(
        self,
        limit: int,
        state_filter: str | None,
    ) -> list[dict[str, object]]:
        history_entries: list[dict[str, object]] = []

        for status_file_path in sorted(
            self._orchestrator_job_status_directory.glob("*/status.json"),
            reverse=True,
        ):
            try:
                status_payload = json.loads(status_file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            state = str(status_payload.get("state", ""))
            if state_filter is not None and state != state_filter:
                continue

            selected_frame_sequence_numbers: list[int] = []
            thumbnail_url: str | None = None
            pipeline_summary_json_file_path = status_payload.get("pipeline_summary_json_file_path")
            if isinstance(pipeline_summary_json_file_path, str) and pipeline_summary_json_file_path:
                summary_path = Path(pipeline_summary_json_file_path)
                if summary_path.exists():
                    try:
                        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        summary_payload = {}
                    frame_judge_decision = summary_payload.get("frame_judge_decision")
                    if isinstance(frame_judge_decision, dict):
                        raw_selected_sequence_numbers = frame_judge_decision.get(
                            "selected_frame_sequence_numbers", []
                        )
                        if isinstance(raw_selected_sequence_numbers, list):
                            selected_frame_sequence_numbers = [
                                int(sequence_number)
                                for sequence_number in raw_selected_sequence_numbers
                                if isinstance(sequence_number, int)
                            ]

                    judge_result_output_directory = summary_payload.get(
                        "judge_result_output_directory"
                    )
                    if isinstance(judge_result_output_directory, str):
                        selected_frames_directory = (
                            Path(judge_result_output_directory) / "selected_frames"
                        )
                        selected_frame_paths = sorted(selected_frames_directory.glob("*"))
                        if selected_frame_paths:
                            thumbnail_url = str(selected_frame_paths[0])

            history_entries.append(
                {
                    "job_id": status_payload.get("job_id"),
                    "pose_preset_identifier": status_payload.get("pose_preset_identifier"),
                    "state": state,
                    "started_at_utc": status_payload.get("started_at_utc"),
                    "finished_at_utc": status_payload.get("finished_at_utc"),
                    "was_cache_hit": bool(status_payload.get("was_cache_hit", False)),
                    "selected_frame_sequence_numbers": selected_frame_sequence_numbers,
                    "thumbnail_url": thumbnail_url,
                }
            )

        return history_entries[:limit]
