from __future__ import annotations

from pathlib import Path

from ui import app as scaffold_application


def test_build_orchestrator_command_arguments_contains_expected_flags() -> None:
    uploaded_image_file_path = Path("/tmp/uploads/original.jpg")
    selected_pose_preset_identifier = "rotate_left"
    generation_job_identifier = "job-123"

    command_arguments = scaffold_application._build_orchestrator_command_arguments(
        uploaded_image_file_path=uploaded_image_file_path,
        selected_pose_preset_identifier=selected_pose_preset_identifier,
        generation_job_identifier=generation_job_identifier,
    )

    assert command_arguments[0:4] == ["uv", "run", "python", "run_pose_pipeline_orchestrator.py"]
    assert "--image" in command_arguments
    assert str(uploaded_image_file_path) in command_arguments
    assert "--preset" in command_arguments
    assert selected_pose_preset_identifier in command_arguments
    assert "--job-id" in command_arguments
    assert generation_job_identifier in command_arguments
    assert "--output-directory" in command_arguments
    assert str(scaffold_application.ORCHESTRATOR_OUTPUT_DIRECTORY) in command_arguments
