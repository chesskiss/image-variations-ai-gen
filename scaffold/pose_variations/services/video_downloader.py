from __future__ import annotations

from urllib.parse import urlparse

import httpx

from pose_variations.domain.asset_models import GeneratedVideoAsset, GeneratedVideoUrl
from pose_variations.domain.generation_job_models import GenerationJobIdentifier
from pose_variations.domain.pose_preset_models import PosePresetIdentifier
from pose_variations.infrastructure.storage_repository import StorageRepository


class VideoDownloaderError(RuntimeError):
    """Raised when generated video download fails."""


class VideoDownloader:
    def __init__(self, storage_repository: StorageRepository) -> None:
        self._storage_repository = storage_repository

    async def download_video_to_storage(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        pose_preset_identifier: PosePresetIdentifier,
        generated_video_url: GeneratedVideoUrl,
    ) -> GeneratedVideoAsset:
        parsed_video_url = urlparse(str(generated_video_url))
        if parsed_video_url.scheme not in {"http", "https"}:
            raise VideoDownloaderError(f"Invalid generated video URL scheme: {generated_video_url}")

        async with httpx.AsyncClient(timeout=120.0) as asynchronous_http_client:
            video_download_response = await asynchronous_http_client.get(str(generated_video_url))
            video_download_response.raise_for_status()
            video_binary_content = video_download_response.content

        if not video_binary_content:
            raise VideoDownloaderError("Downloaded generated video is empty.")

        return self._storage_repository.save_generated_video_asset(
            generation_job_identifier=generation_job_identifier,
            pose_preset_identifier=pose_preset_identifier,
            source_video_url=generated_video_url,
            video_binary_content=video_binary_content,
        )
