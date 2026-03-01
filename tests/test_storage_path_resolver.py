from __future__ import annotations

from pathlib import Path

import pytest

from frame_extraction_stage.domain_models import GenerationJobIdentifier, PosePresetIdentifier
from frame_extraction_stage.storage_path_resolver import (
    StoragePathResolutionError,
    StoragePathResolver,
)


def test_storage_path_resolution_prevents_escape_from_generation_job_directory(tmp_path: Path) -> None:
    storage_base_directory = tmp_path / "storage"
    storage_base_directory.mkdir(parents=True, exist_ok=True)

    storage_path_resolver = StoragePathResolver(storage_base_directory=storage_base_directory)

    generation_job_identifier = GenerationJobIdentifier("job-safe")
    storage_path_resolver.resolve_generation_job_directory(generation_job_identifier)

    outside_video_file_path = tmp_path / "outside_video.mp4"
    outside_video_file_path.write_bytes(b"video")

    with pytest.raises(StoragePathResolutionError):
        storage_path_resolver.validate_local_video_file_path_within_generation_job_directory(
            generation_job_identifier=generation_job_identifier,
            local_video_file_path=outside_video_file_path,
        )


def test_storage_path_resolution_prevents_escaped_output_directory(tmp_path: Path) -> None:
    storage_base_directory = tmp_path / "storage"
    storage_path_resolver = StoragePathResolver(storage_base_directory=storage_base_directory)

    with pytest.raises(StoragePathResolutionError):
        storage_path_resolver.resolve_frames_output_directory(
            generation_job_identifier=GenerationJobIdentifier("../escape"),
            pose_preset_identifier=PosePresetIdentifier("../escape-preset"),
        )
