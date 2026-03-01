from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pose_variations.domain.asset_models import (
    AssetIdentifier,
    ExtractedFrameAsset,
    GeneratedVideoAsset,
    GeneratedVideoUrl,
    UploadedImageAsset,
)
from pose_variations.domain.generation_job_models import GenerationJobIdentifier
from pose_variations.domain.pose_preset_models import PosePresetIdentifier


class StorageRepository:
    def __init__(self, storage_directory: Path) -> None:
        self._storage_directory = storage_directory
        self._storage_directory.mkdir(parents=True, exist_ok=True)

    def create_generation_job_identifier(self) -> GenerationJobIdentifier:
        return GenerationJobIdentifier(uuid4().hex)

    def _generation_job_directory(self, generation_job_identifier: GenerationJobIdentifier) -> Path:
        generation_job_directory = self._storage_directory / str(generation_job_identifier)
        generation_job_directory.mkdir(parents=True, exist_ok=True)
        return generation_job_directory

    def _create_asset_identifier(self) -> AssetIdentifier:
        return AssetIdentifier(uuid4().hex)

    def _build_public_asset_url(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        relative_asset_path: Path,
    ) -> str:
        return f"/jobs/{generation_job_identifier}/assets/{relative_asset_path.as_posix()}"

    def save_uploaded_image_asset(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        uploaded_image_content: bytes,
        mime_type: str,
        original_file_name: str,
    ) -> UploadedImageAsset:
        extension_by_mime_type = {"image/jpeg": ".jpg", "image/png": ".png"}
        file_extension = extension_by_mime_type.get(mime_type, ".bin")
        asset_identifier = self._create_asset_identifier()
        generation_job_directory = self._generation_job_directory(generation_job_identifier)
        relative_asset_path = Path("uploads") / f"uploaded_{asset_identifier}{file_extension}"
        absolute_asset_path = generation_job_directory / relative_asset_path
        absolute_asset_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_asset_path.write_bytes(uploaded_image_content)
        return UploadedImageAsset(
            asset_identifier=asset_identifier,
            local_file_path=absolute_asset_path,
            public_asset_url=self._build_public_asset_url(generation_job_identifier, relative_asset_path),
            mime_type=mime_type,
            original_file_name=original_file_name,
        )

    def save_generated_video_asset(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        pose_preset_identifier: PosePresetIdentifier,
        source_video_url: GeneratedVideoUrl,
        video_binary_content: bytes,
        file_extension: str = ".mp4",
    ) -> GeneratedVideoAsset:
        asset_identifier = self._create_asset_identifier()
        generation_job_directory = self._generation_job_directory(generation_job_identifier)
        relative_asset_path = Path("videos") / f"video_{asset_identifier}{file_extension}"
        absolute_asset_path = generation_job_directory / relative_asset_path
        absolute_asset_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_asset_path.write_bytes(video_binary_content)
        return GeneratedVideoAsset(
            asset_identifier=asset_identifier,
            pose_preset_identifier=pose_preset_identifier,
            source_video_url=source_video_url,
            local_file_path=absolute_asset_path,
            public_asset_url=self._build_public_asset_url(generation_job_identifier, relative_asset_path),
        )

    def create_frame_output_directory(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        generated_video_asset_identifier: AssetIdentifier,
    ) -> Path:
        generation_job_directory = self._generation_job_directory(generation_job_identifier)
        frame_output_directory = generation_job_directory / "frames" / str(generated_video_asset_identifier)
        frame_output_directory.mkdir(parents=True, exist_ok=True)
        return frame_output_directory

    def create_extracted_frame_asset(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        generated_video_asset_identifier: AssetIdentifier,
        frame_file_path: Path,
        frame_timestamp_seconds: float,
    ) -> ExtractedFrameAsset:
        generation_job_directory = self._generation_job_directory(generation_job_identifier)
        relative_asset_path = frame_file_path.relative_to(generation_job_directory)
        return ExtractedFrameAsset(
            asset_identifier=self._create_asset_identifier(),
            generated_video_asset_identifier=generated_video_asset_identifier,
            frame_timestamp_seconds=frame_timestamp_seconds,
            local_file_path=frame_file_path,
            public_asset_url=self._build_public_asset_url(generation_job_identifier, relative_asset_path),
        )

    def resolve_safe_job_asset_path(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        requested_relative_asset_path: str,
    ) -> Path | None:
        generation_job_directory = self._generation_job_directory(generation_job_identifier)
        candidate_asset_path = (generation_job_directory / requested_relative_asset_path).resolve()
        try:
            candidate_asset_path.relative_to(generation_job_directory.resolve())
        except ValueError:
            return None
        if not candidate_asset_path.exists() or not candidate_asset_path.is_file():
            return None
        return candidate_asset_path
