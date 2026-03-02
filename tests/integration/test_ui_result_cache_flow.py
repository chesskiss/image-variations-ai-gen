from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ui import app as ui_application
from ui.cache_index_repository import CacheIndexRepository
from ui.settings import UiSettings


@pytest.mark.integration
def test_second_request_uses_result_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ui_application, "PROJECT_ROOT_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        ui_application, "UPLOAD_STORAGE_DIRECTORY", tmp_path / "storage" / "ui_uploads"
    )
    monkeypatch.setattr(
        ui_application,
        "ORCHESTRATOR_OUTPUT_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui",
    )
    monkeypatch.setattr(
        ui_application,
        "ORCHESTRATOR_JOB_STATUS_DIRECTORY",
        tmp_path / "outputs" / "orchestrator_ui" / "jobs",
    )

    cache_index_repository = CacheIndexRepository(
        cache_index_directory=tmp_path / "outputs" / "cache_index",
        cache_max_entries=100,
        cache_retention_days=30,
    )
    monkeypatch.setattr(ui_application, "cache_index_repository", cache_index_repository)
    monkeypatch.setattr(
        ui_application,
        "ui_settings",
        UiSettings(
            enable_result_cache=True,
            cache_index_directory=tmp_path / "outputs" / "cache_index",
            cache_max_entries=100,
            cache_retention_days=30,
            cache_key_include_model_version=True,
            fal_model_id="fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
            enable_openai_frame_judge=True,
            frame_judge_model_name="o3",
            frame_judge_minimum_score_threshold=0.75,
            extracted_frames_per_second=2,
            extracted_frame_image_format="jpg",
            max_extracted_frames_per_video=24,
        ),
    )

    launch_counter = {"count": 0}

    def fake_launch_orchestrator_background_process(
        generation_job_identifier: str,
        orchestrator_command_arguments: list[str],
    ) -> None:
        del orchestrator_command_arguments
        launch_counter["count"] += 1

        pipeline_summary_json_file_path = (
            tmp_path
            / "outputs"
            / "orchestrator_ui"
            / "pipeline_orchestrator"
            / f"pipeline_summary_{generation_job_identifier}.json"
        )
        judge_result_output_directory = tmp_path / "outputs" / "orchestrator_ui" / "judge_results"
        selected_frames_directory = judge_result_output_directory / "selected_frames"
        selected_frames_directory.mkdir(parents=True, exist_ok=True)
        selected_frame_path = selected_frames_directory / "frame_0001.jpg"
        selected_frame_path.write_bytes(b"123")
        (judge_result_output_directory / "judge_decision.json").write_text("{}", encoding="utf-8")
        (judge_result_output_directory / "ranked_scores.json").write_text("[]", encoding="utf-8")

        summary_payload = {
            "generation_job_identifier": generation_job_identifier,
            "pose_preset_identifier": "rotate_left",
            "generated_video_file_path": str(tmp_path / "outputs" / "generated.mp4"),
            "judge_result_output_directory": str(judge_result_output_directory),
            "frame_judge_decision": {"selected_frame_sequence_numbers": [1]},
        }
        (tmp_path / "outputs" / "generated.mp4").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "outputs" / "generated.mp4").write_bytes(b"0")
        pipeline_summary_json_file_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_summary_json_file_path.write_text(json.dumps(summary_payload), encoding="utf-8")

        status_payload = ui_application._read_job_status_payload(generation_job_identifier)
        ui_application._update_cache_index_from_pipeline_summary(
            cache_key=str(status_payload["cache_key"]),
            original_image_sha256=str(status_payload["original_image_sha256"]),
            source_job_id=generation_job_identifier,
            pose_preset_identifier=str(status_payload["pose_preset_identifier"]),
            pipeline_summary_json_file_path=pipeline_summary_json_file_path,
        )

    monkeypatch.setattr(
        ui_application,
        "_launch_orchestrator_background_process",
        fake_launch_orchestrator_background_process,
    )

    image_buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(120, 50, 20)).save(image_buffer, format="JPEG")
    image_payload = image_buffer.getvalue()

    with TestClient(ui_application.application) as test_client:
        first_response = test_client.post(
            "/jobs",
            files={"uploaded_image_file": ("source.jpg", image_payload, "image/jpeg")},
            data={"selected_pose_preset_identifier": "rotate_left"},
            follow_redirects=False,
        )
        assert first_response.status_code == 303

        second_response = test_client.post(
            "/jobs",
            files={"uploaded_image_file": ("source.jpg", image_payload, "image/jpeg")},
            data={"selected_pose_preset_identifier": "rotate_left"},
            follow_redirects=False,
        )
        assert second_response.status_code == 303

        second_job_identifier = second_response.headers["location"].split("/")[-1]
        second_job_status_response = test_client.get(f"/api/jobs/{second_job_identifier}")
        second_job_payload = second_job_status_response.json()

    assert launch_counter["count"] == 1
    assert second_job_payload["was_cache_hit"] is True
    assert second_job_payload["state"] == "completed"
