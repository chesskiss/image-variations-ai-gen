from __future__ import annotations

import subprocess
from pathlib import Path

from pose_variations.domain.asset_models import ExtractedFrameAsset, GeneratedVideoAsset
from pose_variations.domain.generation_job_models import GenerationJobIdentifier
from pose_variations.infrastructure.settings import ApplicationSettings
from pose_variations.infrastructure.storage_repository import StorageRepository
from pose_variations.services.frame_quality_assessment_service import FrameQualityAssessmentService


class VideoFrameExtractionServiceError(RuntimeError):
    """Raised when video frame extraction fails."""


class VideoFrameExtractionService:
    def __init__(
        self,
        application_settings: ApplicationSettings,
        storage_repository: StorageRepository,
        frame_quality_assessment_service: FrameQualityAssessmentService,
    ) -> None:
        self._application_settings = application_settings
        self._storage_repository = storage_repository
        self._frame_quality_assessment_service = frame_quality_assessment_service

    async def extract_candidate_frames(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        generated_video_asset: GeneratedVideoAsset,
    ) -> list[ExtractedFrameAsset]:
        frame_output_directory = self._storage_repository.create_frame_output_directory(
            generation_job_identifier=generation_job_identifier,
            generated_video_asset_identifier=generated_video_asset.asset_identifier,
        )
        frame_file_pattern = frame_output_directory / "frame_%03d.jpg"

        extraction_frame_rate = 1.0 / self._application_settings.frame_extraction_interval_seconds
        extraction_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(generated_video_asset.local_file_path),
            "-vf",
            f"fps={extraction_frame_rate}",
            str(frame_file_pattern),
        ]

        extraction_result = subprocess.run(extraction_command, capture_output=True, text=True, check=False)
        if extraction_result.returncode != 0:
            raise VideoFrameExtractionServiceError(
                f"ffmpeg extraction failed: {extraction_result.stderr.strip()}"
            )

        extracted_frame_files = sorted(frame_output_directory.glob("frame_*.jpg"))
        if not extracted_frame_files:
            raise VideoFrameExtractionServiceError("No frames were extracted from generated video.")

        extracted_frame_assets: list[ExtractedFrameAsset] = []
        for frame_index, extracted_frame_file_path in enumerate(extracted_frame_files):
            frame_timestamp_seconds = frame_index * self._application_settings.frame_extraction_interval_seconds
            extracted_frame_asset = self._storage_repository.create_extracted_frame_asset(
                generation_job_identifier=generation_job_identifier,
                generated_video_asset_identifier=generated_video_asset.asset_identifier,
                frame_file_path=extracted_frame_file_path,
                frame_timestamp_seconds=frame_timestamp_seconds,
            )
            quality_assessment_result = self._frame_quality_assessment_service.assess_frame_quality(
                extracted_frame_file_path
            )
            extracted_frame_asset.sharpness_score = quality_assessment_result.sharpness_score
            extracted_frame_asset.perceptual_hash_value = quality_assessment_result.perceptual_hash_value
            extracted_frame_assets.append(extracted_frame_asset)

        deduplicated_frame_assets = self._deduplicate_and_filter_frames(extracted_frame_assets)
        return deduplicated_frame_assets

    def _deduplicate_and_filter_frames(
        self,
        extracted_frame_assets: list[ExtractedFrameAsset],
    ) -> list[ExtractedFrameAsset]:
        retained_frame_assets: list[ExtractedFrameAsset] = []

        for candidate_frame_asset in extracted_frame_assets:
            if not self._passes_face_detection_placeholder(candidate_frame_asset.local_file_path):
                continue
            if (candidate_frame_asset.sharpness_score or 0.0) < self._application_settings.minimum_sharpness_score:
                continue
            if candidate_frame_asset.perceptual_hash_value is None:
                continue

            is_near_duplicate = False
            for retained_frame_asset in retained_frame_assets:
                if retained_frame_asset.perceptual_hash_value is None:
                    continue
                hash_distance = self._frame_quality_assessment_service.calculate_perceptual_hash_distance(
                    candidate_frame_asset.perceptual_hash_value,
                    retained_frame_asset.perceptual_hash_value,
                )
                if hash_distance < self._application_settings.minimum_diversity_hash_distance:
                    is_near_duplicate = True
                    break

            if not is_near_duplicate:
                retained_frame_assets.append(candidate_frame_asset)

        return retained_frame_assets

    def _passes_face_detection_placeholder(self, _frame_file_path: Path) -> bool:
        # Architecture hook for optional future face detection step.
        return True
