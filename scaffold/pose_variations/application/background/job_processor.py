from __future__ import annotations

import asyncio
import logging
from datetime import timezone
from typing import Any

from fastapi import BackgroundTasks

from pose_variations.domain.asset_models import ExtractedFrameAsset
from pose_variations.domain.generation_job_models import (
    GenerationJobIdentifier,
    GenerationJobProgressUpdate,
    GenerationJobRecord,
    GenerationJobStatus,
)
from pose_variations.domain.pose_preset_models import PosePresetIdentifier
from pose_variations.infrastructure.storage_repository import StorageRepository
from pose_variations.services.fal_image_to_video_client import FalImageToVideoClient
from pose_variations.services.frame_selection_service import FrameSelectionService
from pose_variations.services.prompt_template_repository import PromptTemplateRepository
from pose_variations.services.video_downloader import VideoDownloader
from pose_variations.services.video_frame_extraction_service import VideoFrameExtractionService


class GenerationJobProcessor:
    def __init__(
        self,
        storage_repository: StorageRepository,
        prompt_template_repository: PromptTemplateRepository,
        fal_image_to_video_client: FalImageToVideoClient,
        video_downloader: VideoDownloader,
        video_frame_extraction_service: VideoFrameExtractionService,
        frame_selection_service: FrameSelectionService,
    ) -> None:
        self._storage_repository = storage_repository
        self._prompt_template_repository = prompt_template_repository
        self._fal_image_to_video_client = fal_image_to_video_client
        self._video_downloader = video_downloader
        self._video_frame_extraction_service = video_frame_extraction_service
        self._frame_selection_service = frame_selection_service

        self._generation_jobs_by_identifier: dict[GenerationJobIdentifier, GenerationJobRecord] = {}
        self._generation_jobs_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    def list_pose_presets_for_upload_form(self) -> list[dict[str, str]]:
        return [
            {
                "pose_preset_identifier": str(pose_preset.pose_preset_identifier),
                "pose_preset_display_name": pose_preset.pose_preset_display_name,
            }
            for pose_preset in self._prompt_template_repository.list_pose_presets_for_display()
        ]

    def create_generation_job(
        self,
        uploaded_image_content: bytes,
        uploaded_image_mime_type: str,
        uploaded_image_original_file_name: str,
        selected_pose_preset_identifiers: list[PosePresetIdentifier],
    ) -> GenerationJobIdentifier:
        generation_job_identifier = self._storage_repository.create_generation_job_identifier()
        uploaded_image_asset = self._storage_repository.save_uploaded_image_asset(
            generation_job_identifier=generation_job_identifier,
            uploaded_image_content=uploaded_image_content,
            mime_type=uploaded_image_mime_type,
            original_file_name=uploaded_image_original_file_name,
        )

        generation_job_record = GenerationJobRecord(
            generation_job_identifier=generation_job_identifier,
            generation_job_status=GenerationJobStatus.PENDING,
            selected_pose_preset_identifiers=selected_pose_preset_identifiers,
            uploaded_image_asset=uploaded_image_asset,
        )
        generation_job_record.progress_updates.append(
            GenerationJobProgressUpdate(stage_name="upload", status_message="Upload validated and stored.")
        )
        self._generation_jobs_by_identifier[generation_job_identifier] = generation_job_record
        self._logger.info("generation_job_created generation_job_id=%s", generation_job_identifier)
        return generation_job_identifier

    def enqueue_generation_job(
        self,
        background_tasks: BackgroundTasks,
        generation_job_identifier: GenerationJobIdentifier,
    ) -> None:
        background_tasks.add_task(self.process_generation_job, generation_job_identifier)

    async def process_generation_job(self, generation_job_identifier: GenerationJobIdentifier) -> None:
        async with self._generation_jobs_lock:
            generation_job_record = self._generation_jobs_by_identifier[generation_job_identifier]
            generation_job_record.generation_job_status = GenerationJobStatus.RUNNING
            self._append_progress(
                generation_job_record,
                stage_name="generation",
                status_message="Starting pose variation generation.",
            )

        try:
            for pose_preset_identifier in generation_job_record.selected_pose_preset_identifiers:
                prompt_template = self._prompt_template_repository.get_prompt_template_for_identifier(
                    pose_preset_identifier
                )
                self._append_progress(
                    generation_job_record,
                    stage_name="generation",
                    status_message=(
                        "Submitting generation request for preset "
                        f"{pose_preset_identifier}."
                    ),
                )

                fal_generation_request_identifier = (
                    await self._fal_image_to_video_client.submit_image_to_video_generation(
                        uploaded_image_asset=generation_job_record.uploaded_image_asset,
                        prompt_template=prompt_template,
                    )
                )
                generated_video_url = await self._fal_image_to_video_client.wait_for_generation_result(
                    fal_generation_request_identifier
                )
                downloaded_video_asset = await self._video_downloader.download_video_to_storage(
                    generation_job_identifier=generation_job_identifier,
                    pose_preset_identifier=pose_preset_identifier,
                    generated_video_url=generated_video_url,
                )
                generation_job_record.generated_video_assets.append(downloaded_video_asset)

            self._append_progress(
                generation_job_record,
                stage_name="frame_extraction",
                status_message="Extracting candidate frames from generated videos.",
            )
            for generated_video_asset in generation_job_record.generated_video_assets:
                extracted_frame_assets = (
                    await self._video_frame_extraction_service.extract_candidate_frames(
                        generation_job_identifier=generation_job_identifier,
                        generated_video_asset=generated_video_asset,
                    )
                )
                generation_job_record.extracted_frame_assets.extend(extracted_frame_assets)

            self._append_progress(
                generation_job_record,
                stage_name="selection",
                status_message="Selecting best distinct frames.",
            )
            generation_job_record.frame_selection_decision = (
                await self._frame_selection_service.select_distinct_best_frames(
                    uploaded_image_asset=generation_job_record.uploaded_image_asset,
                    candidate_frame_assets=generation_job_record.extracted_frame_assets,
                )
            )

            generation_job_record.generation_job_status = GenerationJobStatus.COMPLETED
            self._append_progress(
                generation_job_record,
                stage_name="completed",
                status_message="Generation job completed.",
            )
            self._logger.info("generation_job_completed generation_job_id=%s", generation_job_identifier)
        except Exception as generation_error:  # noqa: BLE001
            generation_job_record.generation_job_status = GenerationJobStatus.FAILED
            generation_job_record.failure_reason = str(generation_error)
            self._append_progress(
                generation_job_record,
                stage_name="failed",
                status_message="Generation job failed.",
            )
            self._logger.exception(
                "generation_job_failed generation_job_id=%s", generation_job_identifier
            )

    def get_generation_job_record(
        self,
        generation_job_identifier: GenerationJobIdentifier,
    ) -> GenerationJobRecord | None:
        return self._generation_jobs_by_identifier.get(generation_job_identifier)

    def serialize_generation_job(
        self,
        generation_job_identifier: GenerationJobIdentifier,
    ) -> dict[str, Any] | None:
        generation_job_record = self.get_generation_job_record(generation_job_identifier)
        if generation_job_record is None:
            return None

        selected_frame_assets: list[ExtractedFrameAsset] = []
        ranked_candidates: list[dict[str, Any]] = []
        if generation_job_record.frame_selection_decision is not None:
            selected_frame_assets = generation_job_record.frame_selection_decision.selected_frame_assets
            ranked_candidates = [
                {
                    "frame_asset_identifier": str(ranked_candidate.extracted_frame_asset.asset_identifier),
                    "public_asset_url": ranked_candidate.extracted_frame_asset.public_asset_url,
                    "frame_timestamp_seconds": ranked_candidate.extracted_frame_asset.frame_timestamp_seconds,
                    "sharpness_score": ranked_candidate.sharpness_score,
                    "identity_similarity_score": (
                        float(ranked_candidate.identity_similarity_score)
                        if ranked_candidate.identity_similarity_score is not None
                        else None
                    ),
                    "diversity_score_from_primary_frame": (
                        float(ranked_candidate.diversity_score_from_primary_frame)
                        if ranked_candidate.diversity_score_from_primary_frame is not None
                        else None
                    ),
                    "aggregate_selection_score": ranked_candidate.aggregate_selection_score,
                }
                for ranked_candidate in generation_job_record.frame_selection_decision.ranked_frame_candidates
            ]

        return {
            "generation_job_identifier": str(generation_job_record.generation_job_identifier),
            "generation_job_status": generation_job_record.generation_job_status.value,
            "failure_reason": generation_job_record.failure_reason,
            "selected_pose_preset_identifiers": [
                str(pose_preset_identifier)
                for pose_preset_identifier in generation_job_record.selected_pose_preset_identifiers
            ],
            "uploaded_image": {
                "public_asset_url": generation_job_record.uploaded_image_asset.public_asset_url,
                "mime_type": generation_job_record.uploaded_image_asset.mime_type,
                "original_file_name": generation_job_record.uploaded_image_asset.original_file_name,
            },
            "generated_videos": [
                {
                    "asset_identifier": str(video_asset.asset_identifier),
                    "pose_preset_identifier": str(video_asset.pose_preset_identifier),
                    "public_asset_url": video_asset.public_asset_url,
                }
                for video_asset in generation_job_record.generated_video_assets
            ],
            "selected_frames": [
                {
                    "asset_identifier": str(frame_asset.asset_identifier),
                    "public_asset_url": frame_asset.public_asset_url,
                    "frame_timestamp_seconds": frame_asset.frame_timestamp_seconds,
                    "identity_similarity_score": frame_asset.identity_similarity_score,
                }
                for frame_asset in selected_frame_assets
            ],
            "more_candidates": ranked_candidates,
            "progress_updates": [
                {
                    "stage_name": progress_update.stage_name,
                    "status_message": progress_update.status_message,
                    "created_at_utc": progress_update.created_at_utc.astimezone(
                        timezone.utc
                    ).isoformat(),
                }
                for progress_update in generation_job_record.progress_updates
            ],
        }

    def is_valid_pose_preset_identifier(self, pose_preset_identifier: str) -> bool:
        return self._prompt_template_repository.is_valid_pose_preset_identifier(pose_preset_identifier)

    def _append_progress(
        self,
        generation_job_record: GenerationJobRecord,
        stage_name: str,
        status_message: str,
    ) -> None:
        generation_job_record.progress_updates.append(
            GenerationJobProgressUpdate(stage_name=stage_name, status_message=status_message)
        )
