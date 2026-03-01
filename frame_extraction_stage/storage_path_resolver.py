from __future__ import annotations

from pathlib import Path

from frame_extraction_stage.domain_models import GenerationJobIdentifier, PosePresetIdentifier


class StoragePathResolutionError(RuntimeError):
    """Raised when storage paths are invalid or unsafe."""


class StoragePathResolver:
    def __init__(self, storage_base_directory: Path) -> None:
        self._storage_base_directory = storage_base_directory.resolve()
        self._storage_base_directory.mkdir(parents=True, exist_ok=True)

    def resolve_generation_job_directory(
        self,
        generation_job_identifier: GenerationJobIdentifier,
    ) -> Path:
        generation_job_directory = self._storage_base_directory / str(generation_job_identifier)
        return self._resolve_and_ensure_path_within_storage(generation_job_directory, create_directory=True)

    def resolve_frames_output_directory(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        pose_preset_identifier: PosePresetIdentifier,
    ) -> Path:
        output_directory_for_extracted_frames = (
            self._storage_base_directory / str(generation_job_identifier) / str(pose_preset_identifier) / "frames"
        )
        return self._resolve_and_ensure_path_within_storage(
            output_directory_for_extracted_frames,
            create_directory=True,
        )

    def validate_local_video_file_path_within_generation_job_directory(
        self,
        generation_job_identifier: GenerationJobIdentifier,
        local_video_file_path: Path,
    ) -> Path:
        resolved_local_video_file_path = local_video_file_path.resolve()
        generation_job_directory = self.resolve_generation_job_directory(generation_job_identifier)

        if not resolved_local_video_file_path.exists() or not resolved_local_video_file_path.is_file():
            raise StoragePathResolutionError(
                f"Video file does not exist or is not a file: {resolved_local_video_file_path}"
            )

        try:
            resolved_local_video_file_path.relative_to(generation_job_directory)
        except ValueError as relative_path_error:
            raise StoragePathResolutionError(
                "Video file path must be inside the generation job storage directory."
            ) from relative_path_error

        return resolved_local_video_file_path

    def _resolve_and_ensure_path_within_storage(
        self,
        candidate_path: Path,
        create_directory: bool,
    ) -> Path:
        resolved_candidate_path = candidate_path.resolve()
        try:
            resolved_candidate_path.relative_to(self._storage_base_directory)
        except ValueError as relative_path_error:
            raise StoragePathResolutionError(
                "Resolved path escapes configured storage directory."
            ) from relative_path_error

        if create_directory:
            resolved_candidate_path.mkdir(parents=True, exist_ok=True)

        return resolved_candidate_path
