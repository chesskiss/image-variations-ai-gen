from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from frame_extraction_stage.domain_models import ExtractedFrameAsset, GeneratedVideoAsset
from frame_extraction_stage.ffmpeg_command_runner import FfmpegCommandRunner
from frame_extraction_stage.storage_path_resolver import StoragePathResolver


class VideoFrameExtractionServiceError(RuntimeError):
    """Raised when frame extraction cannot complete safely."""


@dataclass(slots=True, frozen=True)
class VideoFrameExtractionConfiguration:
    ffmpeg_executable_path: str
    extracted_frames_per_second: int
    extracted_frame_image_format: str
    max_extracted_frames_per_video: int
    target_total_frames: int | None = None


class VideoFrameExtractionService:
    def __init__(
        self,
        video_frame_extraction_configuration: VideoFrameExtractionConfiguration,
        storage_path_resolver: StoragePathResolver,
        ffmpeg_command_runner: FfmpegCommandRunner,
    ) -> None:
        self._video_frame_extraction_configuration = video_frame_extraction_configuration
        self._storage_path_resolver = storage_path_resolver
        self._ffmpeg_command_runner = ffmpeg_command_runner

        self._validate_extraction_configuration()
        self._ffmpeg_command_runner.verify_ffmpeg_is_available()

    def extract_candidate_frames_from_generated_video(
        self,
        generated_video_asset: GeneratedVideoAsset,
    ) -> list[ExtractedFrameAsset]:
        video_file_path_for_generation_job = (
            self._storage_path_resolver.validate_local_video_file_path_within_generation_job_directory(
                generation_job_identifier=generated_video_asset.generation_job_identifier,
                local_video_file_path=generated_video_asset.local_video_file_path,
            )
        )

        output_directory_for_extracted_frames = self._storage_path_resolver.resolve_frames_output_directory(
            generation_job_identifier=generated_video_asset.generation_job_identifier,
            pose_preset_identifier=generated_video_asset.pose_preset_identifier,
        )
        deterministic_output_file_pattern = (
            output_directory_for_extracted_frames
            / f"frame_%04d.{self._video_frame_extraction_configuration.extracted_frame_image_format}"
        )

        ffmpeg_command_arguments = self.build_ffmpeg_command_arguments_for_uniform_fps_sampling(
            ffmpeg_executable_path=self._video_frame_extraction_configuration.ffmpeg_executable_path,
            video_file_path_for_generation_job=video_file_path_for_generation_job,
            deterministic_output_file_pattern=deterministic_output_file_pattern,
            extracted_frames_per_second=self._video_frame_extraction_configuration.extracted_frames_per_second,
            max_extracted_frames_per_video=self._video_frame_extraction_configuration.max_extracted_frames_per_video,
        )
        ffmpeg_command_execution_result = self._ffmpeg_command_runner.run_ffmpeg_command(
            command_arguments=ffmpeg_command_arguments,
            timeout_seconds=90,
        )
        self._ffmpeg_command_runner.raise_error_when_execution_failed(ffmpeg_command_execution_result)

        extracted_frame_assets = self._build_extracted_frame_assets_from_output_directory(
            generated_video_asset=generated_video_asset,
            output_directory_for_extracted_frames=output_directory_for_extracted_frames,
        )

        if len(extracted_frame_assets) > self._video_frame_extraction_configuration.max_extracted_frames_per_video:
            raise VideoFrameExtractionServiceError(
                "Extracted frame count exceeded configured maximum limit."
            )

        return extracted_frame_assets

    @staticmethod
    def build_ffmpeg_command_arguments_for_uniform_fps_sampling(
        ffmpeg_executable_path: str,
        video_file_path_for_generation_job: Path,
        deterministic_output_file_pattern: Path,
        extracted_frames_per_second: int,
        max_extracted_frames_per_video: int,
    ) -> list[str]:
        return [
            ffmpeg_executable_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_file_path_for_generation_job),
            "-vf",
            f"fps={extracted_frames_per_second}",
            "-vsync",
            "vfr",
            "-frames:v",
            str(max_extracted_frames_per_video),
            str(deterministic_output_file_pattern),
        ]

    def _build_extracted_frame_assets_from_output_directory(
        self,
        generated_video_asset: GeneratedVideoAsset,
        output_directory_for_extracted_frames: Path,
    ) -> list[ExtractedFrameAsset]:
        extracted_frame_file_paths = sorted(
            output_directory_for_extracted_frames.glob(
                f"frame_*.{self._video_frame_extraction_configuration.extracted_frame_image_format}"
            )
        )
        extracted_frame_assets: list[ExtractedFrameAsset] = []

        for extracted_frame_file_path in extracted_frame_file_paths:
            frame_sequence_number = self._parse_frame_sequence_number(extracted_frame_file_path.name)
            image_width_pixels, image_height_pixels = self._read_image_dimensions_if_possible(
                extracted_frame_file_path
            )
            file_size_bytes = extracted_frame_file_path.stat().st_size
            extracted_frame_timestamp_seconds = (
                frame_sequence_number
                / self._video_frame_extraction_configuration.extracted_frames_per_second
            )
            extracted_frame_assets.append(
                ExtractedFrameAsset(
                    generation_job_identifier=generated_video_asset.generation_job_identifier,
                    pose_preset_identifier=generated_video_asset.pose_preset_identifier,
                    source_video_local_file_path=generated_video_asset.local_video_file_path,
                    frame_sequence_number=frame_sequence_number,
                    timestamp_seconds=float(extracted_frame_timestamp_seconds),
                    local_frame_image_file_path=extracted_frame_file_path,
                    image_file_format=self._video_frame_extraction_configuration.extracted_frame_image_format,
                    image_width_pixels=image_width_pixels,
                    image_height_pixels=image_height_pixels,
                    file_size_bytes=file_size_bytes,
                    basic_quality_metrics={},
                )
            )

        extracted_frame_assets.sort(key=lambda extracted_frame_asset: extracted_frame_asset.frame_sequence_number)
        return extracted_frame_assets

    @staticmethod
    def _parse_frame_sequence_number(extracted_frame_file_name: str) -> int:
        frame_sequence_match = re.search(r"frame_(\d{4})", extracted_frame_file_name)
        if frame_sequence_match is None:
            raise VideoFrameExtractionServiceError(
                f"Unexpected frame file name format: {extracted_frame_file_name}"
            )
        return int(frame_sequence_match.group(1))

    @staticmethod
    def _read_image_dimensions_if_possible(extracted_frame_file_path: Path) -> tuple[int | None, int | None]:
        try:
            with Image.open(extracted_frame_file_path) as extracted_frame_image:
                return extracted_frame_image.size
        except OSError:
            return None, None

    def _validate_extraction_configuration(self) -> None:
        image_format = self._video_frame_extraction_configuration.extracted_frame_image_format
        if image_format not in {"jpg", "png"}:
            raise VideoFrameExtractionServiceError(
                "EXTRACTED_FRAME_IMAGE_FORMAT must be either 'jpg' or 'png'."
            )

        if self._video_frame_extraction_configuration.extracted_frames_per_second <= 0:
            raise VideoFrameExtractionServiceError(
                "EXTRACTED_FRAMES_PER_SECOND must be greater than zero."
            )

        if self._video_frame_extraction_configuration.max_extracted_frames_per_video <= 0:
            raise VideoFrameExtractionServiceError(
                "MAX_EXTRACTED_FRAMES_PER_VIDEO must be greater than zero."
            )
