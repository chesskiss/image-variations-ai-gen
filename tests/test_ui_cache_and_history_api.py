from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from ui import app as ui_application
from ui.cache_index_repository import CacheIndexRepository
from ui.cache_models import CacheIndexEntry


def test_cache_api_returns_hit_and_miss(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui_application, "PROJECT_ROOT_DIRECTORY", tmp_path)
    cache_index_repository = CacheIndexRepository(
        cache_index_directory=tmp_path / "outputs" / "cache_index",
        cache_max_entries=100,
        cache_retention_days=30,
    )
    monkeypatch.setattr(ui_application, "cache_index_repository", cache_index_repository)

    generated_video_file_path = tmp_path / "outputs" / "generated.mp4"
    generated_video_file_path.parent.mkdir(parents=True, exist_ok=True)
    generated_video_file_path.write_bytes(b"x")

    judge_result_output_directory = tmp_path / "outputs" / "judge"
    selected_frames_directory = judge_result_output_directory / "selected_frames"
    selected_frames_directory.mkdir(parents=True, exist_ok=True)
    selected_frame_path = selected_frames_directory / "frame_0001.jpg"
    selected_frame_path.write_bytes(b"x")

    summary_file_path = tmp_path / "outputs" / "summary.json"
    summary_file_path.write_text("{}", encoding="utf-8")

    cache_index_repository.upsert_cache_entry(
        CacheIndexEntry(
            cache_key="known",
            created_at_utc=datetime.now(UTC).isoformat(),
            last_accessed_at_utc=datetime.now(UTC).isoformat(),
            source_job_id="job-1",
            pose_preset_identifier="rotate_left",
            original_image_sha256="abc",
            pipeline_summary_json_file_path=str(summary_file_path),
            judge_result_output_directory=str(judge_result_output_directory),
            selected_frame_image_file_paths=[str(selected_frame_path)],
            generated_video_file_path=str(generated_video_file_path),
            storage_paths_exist=True,
        )
    )

    with TestClient(ui_application.application) as test_client:
        hit_response = test_client.get("/api/cache/known")
        miss_response = test_client.get("/api/cache/unknown")

    assert hit_response.status_code == 200
    assert hit_response.json()["cache_hit"] is True
    assert miss_response.status_code == 200
    assert miss_response.json()["cache_hit"] is False


def test_history_api_returns_entries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui_application, "PROJECT_ROOT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        ui_application,
        "ORCHESTRATOR_JOB_STATUS_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui" / "jobs",
    )

    status_directory = tmp_path / "outputs" / "orchestrator_ui" / "jobs" / "job-1"
    status_directory.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "job_id": "job-1",
        "pose_preset_identifier": "rotate_left",
        "state": "completed",
        "started_at_utc": "2026-03-01T10:00:00+00:00",
        "finished_at_utc": "2026-03-01T10:01:00+00:00",
        "was_cache_hit": True,
        "pipeline_summary_json_file_path": None,
    }
    (status_directory / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")

    with TestClient(ui_application.application) as test_client:
        response = test_client.get("/api/history?limit=50&state=completed")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["job_id"] == "job-1"
