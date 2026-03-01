from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from frame_judge_stage.domain_models import (
    FrameCandidateForJudging,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_judge_stage.frame_selection_service import FrameSelectionService
from frame_judge_stage.openai_vision_frame_judge import OpenAiVisionFrameJudge
from frame_judge_stage.result_exporter import FrameJudgeResultExporter
from frame_judge_stage.rule_based_frame_judge import RuleBasedFrameJudge
from frame_judge_stage.settings import load_frame_judge_stage_settings


def parse_command_line_arguments() -> argparse.Namespace:
    command_line_parser = argparse.ArgumentParser(
        description="Run frame judge stage on extracted frames and export selected results.",
    )
    command_line_parser.add_argument("--frames-directory", required=True)
    command_line_parser.add_argument("--job-id", required=True)
    command_line_parser.add_argument("--preset-id", required=True)
    command_line_parser.add_argument("--output-directory", default="./outputs")
    command_line_parser.add_argument("--original-image-file-path", default=None)
    return command_line_parser.parse_args()


def load_frame_candidates_for_judging_from_directory(
    frames_directory: Path,
    generation_job_identifier: GenerationJobIdentifier,
    pose_preset_identifier: PosePresetIdentifier,
    original_image_file_path: Path | None,
) -> list[FrameCandidateForJudging]:
    frame_file_paths = sorted(
        [
            frame_file_path
            for frame_file_path in frames_directory.iterdir()
            if frame_file_path.is_file()
            and frame_file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (original_image_file_path is None or frame_file_path.resolve() != original_image_file_path.resolve())
        ]
    )

    frame_candidates_for_judging: list[FrameCandidateForJudging] = []
    for fallback_sequence_index, frame_file_path in enumerate(frame_file_paths, start=1):
        frame_sequence_number = parse_frame_sequence_number(frame_file_path.name, fallback_sequence_index)
        image_width_pixels, image_height_pixels = read_frame_dimensions(frame_file_path)
        frame_candidates_for_judging.append(
            FrameCandidateForJudging(
                generation_job_identifier=generation_job_identifier,
                pose_preset_identifier=pose_preset_identifier,
                frame_sequence_number=frame_sequence_number,
                timestamp_seconds=0.0,
                local_frame_image_file_path=frame_file_path,
                image_width_pixels=image_width_pixels,
                image_height_pixels=image_height_pixels,
                file_size_bytes=frame_file_path.stat().st_size,
                basic_quality_metrics={},
            )
        )

    frame_candidates_for_judging.sort(
        key=lambda frame_candidate_for_judging: frame_candidate_for_judging.frame_sequence_number
    )
    return frame_candidates_for_judging


def parse_frame_sequence_number(frame_file_name: str, fallback_sequence_index: int) -> int:
    frame_sequence_match = re.search(r"frame_(\d{4})", frame_file_name)
    if frame_sequence_match is not None:
        return int(frame_sequence_match.group(1))
    return fallback_sequence_index


def read_frame_dimensions(frame_file_path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(frame_file_path) as frame_image:
            return frame_image.size
    except OSError:
        return None, None


def resolve_original_image_file_path(
    frames_directory: Path,
    optional_original_image_file_path_argument: str | None,
) -> Path | None:
    if optional_original_image_file_path_argument is not None:
        resolved_original_image_file_path = (
            Path(optional_original_image_file_path_argument).expanduser().resolve()
        )
        if not resolved_original_image_file_path.exists() or not resolved_original_image_file_path.is_file():
            raise ValueError(
                f"Original image file does not exist: {resolved_original_image_file_path}"
            )
        return resolved_original_image_file_path

    candidate_original_file_names = [
        "original.jpg",
        "original.jpeg",
        "original.png",
        "source.jpg",
        "source.jpeg",
        "source.png",
    ]
    for candidate_original_file_name in candidate_original_file_names:
        candidate_original_image_file_path = frames_directory / candidate_original_file_name
        if candidate_original_image_file_path.exists() and candidate_original_image_file_path.is_file():
            return candidate_original_image_file_path.resolve()

    return None


def main() -> None:
    command_line_arguments = parse_command_line_arguments()
    frame_judge_stage_settings = load_frame_judge_stage_settings()

    frames_directory = Path(command_line_arguments.frames_directory).expanduser().resolve()
    if not frames_directory.exists() or not frames_directory.is_dir():
        raise ValueError(f"Frames directory does not exist: {frames_directory}")

    generation_job_identifier = GenerationJobIdentifier(command_line_arguments.job_id)
    pose_preset_identifier = PosePresetIdentifier(command_line_arguments.preset_id)
    original_image_file_path = resolve_original_image_file_path(
        frames_directory=frames_directory,
        optional_original_image_file_path_argument=command_line_arguments.original_image_file_path,
    )

    frame_candidates_for_judging = load_frame_candidates_for_judging_from_directory(
        frames_directory=frames_directory,
        generation_job_identifier=generation_job_identifier,
        pose_preset_identifier=pose_preset_identifier,
        original_image_file_path=original_image_file_path,
    )
    if not frame_candidates_for_judging:
        raise ValueError(f"No image frames found in directory: {frames_directory}")
    if frame_judge_stage_settings.enable_openai_frame_judge and original_image_file_path is None:
        raise ValueError(
            "OpenAI frame judge requires original image. Provide --original-image-file-path "
            "or place original.jpg/original.png in the frames directory."
        )

    if frame_judge_stage_settings.enable_openai_frame_judge:
        frame_judge = OpenAiVisionFrameJudge(
            openai_api_key=frame_judge_stage_settings.openai_api_key,
            judge_model_name=frame_judge_stage_settings.frame_judge_model_name,
            timeout_seconds=frame_judge_stage_settings.frame_judge_timeout_seconds,
        )
    else:
        frame_judge = RuleBasedFrameJudge(
            judge_model_name="rule-based-baseline",
        )

    frame_judge_scores = frame_judge.judge_frame_candidates(
        frame_candidates_for_judging,
        original_image_file_path=original_image_file_path,
    )
    frame_selection_service = FrameSelectionService(
        selected_count=frame_judge_stage_settings.frame_judge_selected_count,
        minimum_score_threshold=frame_judge_stage_settings.frame_judge_minimum_score_threshold,
    )
    frame_judge_decision = frame_selection_service.select_best_frames_from_judge_scores(
        generation_job_identifier=generation_job_identifier,
        pose_preset_identifier=pose_preset_identifier,
        frame_candidates_for_judging=frame_candidates_for_judging,
        frame_judge_scores=frame_judge_scores,
    )

    output_base_directory = Path(command_line_arguments.output_directory).expanduser().resolve()
    frame_judge_result_exporter = FrameJudgeResultExporter()
    result_output_directory = frame_judge_result_exporter.export_frame_judge_results(
        output_base_directory=output_base_directory,
        frame_candidates_for_judging=frame_candidates_for_judging,
        frame_judge_decision=frame_judge_decision,
    )

    print(f"Frame judge results written to: {result_output_directory}")
    print(
        "Selected frame sequence numbers: "
        f"{frame_judge_decision.selected_frame_sequence_numbers}"
    )


if __name__ == "__main__":
    main()
