from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ui import app as scaffold_application


def _write_status_payload(
    base_directory: Path,
    generation_job_identifier: str,
    payload: dict[str, object],
) -> None:
    status_file_path = (
        base_directory
        / "outputs"
        / "orchestrator_ui"
        / "jobs"
        / generation_job_identifier
        / "status.json"
    )
    status_file_path.parent.mkdir(parents=True, exist_ok=True)
    status_file_path.write_text(json.dumps(payload), encoding="utf-8")


def test_api_job_status_returns_completed_payload_with_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(scaffold_application, "PROJECT_ROOT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        scaffold_application,
        "ORCHESTRATOR_OUTPUT_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui",
    )
    monkeypatch.setattr(
        scaffold_application,
        "ORCHESTRATOR_JOB_STATUS_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui" / "jobs",
    )

    generation_job_identifier = "job-completed"
    judge_output_directory = tmp_path / "outputs" / "orchestrator_ui" / "judge_results"
    selected_frames_directory = judge_output_directory / "selected_frames"
    selected_frames_directory.mkdir(parents=True, exist_ok=True)
    (selected_frames_directory / "frame_0001.jpg").write_bytes(b"123")
    (judge_output_directory / "judge_decision.json").write_text("{}", encoding="utf-8")
    (judge_output_directory / "ranked_scores.json").write_text("[]", encoding="utf-8")

    summary_json_file_path = (
        tmp_path
        / "outputs"
        / "orchestrator_ui"
        / "pipeline_orchestrator"
        / f"pipeline_summary_{generation_job_identifier}.json"
    )
    summary_json_file_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "generation_job_identifier": generation_job_identifier,
        "pose_preset_identifier": "rotate_right",
        "judge_result_output_directory": str(judge_output_directory),
        "frame_judge_decision": {"selected_frame_sequence_numbers": [1, 2]},
    }
    summary_json_file_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    status_payload = {
        "job_id": generation_job_identifier,
        "state": "completed",
        "progress_percent": 100,
        "current_stage": "completed",
        "error_message": None,
        "started_at_utc": "2026-03-01T10:00:00+00:00",
        "finished_at_utc": "2026-03-01T10:01:00+00:00",
        "pipeline_summary_json_file_path": str(summary_json_file_path),
    }
    _write_status_payload(tmp_path, generation_job_identifier, status_payload)

    with TestClient(scaffold_application.application) as test_client:
        response = test_client.get(f"/api/jobs/{generation_job_identifier}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == generation_job_identifier
    assert payload["state"] == "completed"
    assert (
        payload["pipeline_summary_payload"]["generation_job_identifier"]
        == generation_job_identifier
    )
    assert payload["artifact_urls"]["summary_json_url"].startswith("/artifacts/")
    assert len(payload["artifact_urls"]["selected_frame_image_urls"]) == 1


def test_api_job_status_returns_failed_payload_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(scaffold_application, "PROJECT_ROOT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        scaffold_application,
        "ORCHESTRATOR_OUTPUT_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui",
    )
    monkeypatch.setattr(
        scaffold_application,
        "ORCHESTRATOR_JOB_STATUS_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui" / "jobs",
    )

    generation_job_identifier = "job-failed"
    status_payload = {
        "job_id": generation_job_identifier,
        "state": "failed",
        "progress_percent": 100,
        "current_stage": "failed",
        "error_message": "Orchestrator execution failed.",
        "started_at_utc": "2026-03-01T10:00:00+00:00",
        "finished_at_utc": "2026-03-01T10:01:00+00:00",
        "pipeline_summary_json_file_path": None,
    }
    _write_status_payload(tmp_path, generation_job_identifier, status_payload)

    with TestClient(scaffold_application.application) as test_client:
        response = test_client.get(f"/api/jobs/{generation_job_identifier}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "failed"
    assert payload["artifact_urls"] is None
    assert payload["pipeline_summary_payload"] is None
    assert payload["error_message"] == "Orchestrator execution failed."
