from __future__ import annotations

import json
from pathlib import Path

from ui.history_repository import HistoryRepository


def test_history_repository_filters_by_state_and_limits_entries(tmp_path: Path) -> None:
    status_directory = tmp_path / "jobs"
    status_directory.mkdir(parents=True, exist_ok=True)

    for job_identifier, state in [
        ("job-a", "completed"),
        ("job-b", "failed"),
        ("job-c", "completed"),
    ]:
        job_directory = status_directory / job_identifier
        job_directory.mkdir(parents=True, exist_ok=True)
        status_payload = {
            "job_id": job_identifier,
            "pose_preset_identifier": "rotate_left",
            "state": state,
            "started_at_utc": "2026-03-01T10:00:00+00:00",
            "finished_at_utc": "2026-03-01T10:01:00+00:00",
            "was_cache_hit": False,
            "pipeline_summary_json_file_path": None,
        }
        (job_directory / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")

    history_repository = HistoryRepository(status_directory)
    completed_entries = history_repository.list_history_entries(limit=1, state_filter="completed")

    assert len(completed_entries) == 1
    assert completed_entries[0]["state"] == "completed"
