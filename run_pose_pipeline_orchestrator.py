from __future__ import annotations

import argparse
from pathlib import Path

from orchestration_stage.pipeline_orchestrator import PosePipelineOrchestrator
from single_image_transform.prompts import POSE_PRESET_PROMPTS_BY_NAME


def parse_command_line_arguments() -> argparse.Namespace:
    command_line_parser = argparse.ArgumentParser(
        description="Run generation, extraction, and frame judge stages in one orchestrated command.",
    )
    command_line_parser.add_argument("--image", required=True)
    command_line_parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(POSE_PRESET_PROMPTS_BY_NAME.keys()),
    )
    command_line_parser.add_argument("--job-id", default=None)
    command_line_parser.add_argument("--output-directory", default="./outputs")
    return command_line_parser.parse_args()


def main() -> None:
    command_line_arguments = parse_command_line_arguments()

    pose_pipeline_orchestrator = PosePipelineOrchestrator()
    pipeline_orchestrator_result = (
        pose_pipeline_orchestrator.run_generation_extraction_and_judge_pipeline(
            input_image_file_path=Path(command_line_arguments.image),
            pose_preset_identifier=command_line_arguments.preset,
            output_base_directory=Path(command_line_arguments.output_directory)
            .expanduser()
            .resolve(),
            generation_job_identifier=command_line_arguments.job_id,
        )
    )

    print(f"Generation job identifier: {pipeline_orchestrator_result.generation_job_identifier}")
    print(f"Generated video file path: {pipeline_orchestrator_result.generated_video_file_path}")
    print(f"Frames directory path: {pipeline_orchestrator_result.frames_directory_path}")
    print(f"Original image file path: {pipeline_orchestrator_result.original_image_file_path}")
    print(
        f"Selected frame sequence numbers: {pipeline_orchestrator_result.selected_frame_sequence_numbers}"
    )
    print(
        f"Judge result output directory: {pipeline_orchestrator_result.judge_result_output_directory}"
    )
    print(
        "Pipeline summary JSON file path: "
        f"{pipeline_orchestrator_result.orchestrator_summary_json_file_path}"
    )


if __name__ == "__main__":
    main()
