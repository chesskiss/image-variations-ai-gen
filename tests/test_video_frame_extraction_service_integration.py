from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from frame_extraction_stage.domain_models import (
    GeneratedVideoAsset,
    GenerationJobIdentifier,
    PosePresetIdentifier,
)
from frame_extraction_stage.ffmpeg_command_runner import FfmpegCommandRunner
from frame_extraction_stage.storage_path_resolver import StoragePathResolver
from frame_extraction_stage.video_frame_extraction_service import (
    VideoFrameExtractionConfiguration,
    VideoFrameExtractionService,
)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for extraction test.")
def test_video_frame_extraction_from_fixture_mp4(tmp_path: Path) -> None:
    fixture_video_file_path = Path("tests/fixtures/tiny_sample.mp4")
    if not fixture_video_file_path.exists():
        pytest.skip(
            "Fixture tests/fixtures/tiny_sample.mp4 is missing. Add a tiny MP4 fixture to run this test."
        )

    storage_base_directory = tmp_path / "storage"
    generation_job_identifier = GenerationJobIdentifier("job-123")
    pose_preset_identifier = PosePresetIdentifier("rotate_right")

    local_video_file_path_for_generation_job = (
        storage_base_directory / str(generation_job_identifier) / str(pose_preset_identifier) / "video.mp4"
    )
    local_video_file_path_for_generation_job.parent.mkdir(parents=True, exist_ok=True)
    local_video_file_path_for_generation_job.write_bytes(fixture_video_file_path.read_bytes())

    generated_video_asset = GeneratedVideoAsset(
        generation_job_identifier=generation_job_identifier,
        pose_preset_identifier=pose_preset_identifier,
        local_video_file_path=local_video_file_path_for_generation_job,
        video_duration_seconds=None,
        video_file_format="mp4",
    )

    video_frame_extraction_configuration = VideoFrameExtractionConfiguration(
        ffmpeg_executable_path="ffmpeg",
        extracted_frames_per_second=2,
        extracted_frame_image_format="jpg",
        max_extracted_frames_per_video=24,
    )
    storage_path_resolver = StoragePathResolver(storage_base_directory=storage_base_directory)
    ffmpeg_command_runner = FfmpegCommandRunner(ffmpeg_executable_path="ffmpeg")
    video_frame_extraction_service = VideoFrameExtractionService(
        video_frame_extraction_configuration=video_frame_extraction_configuration,
        storage_path_resolver=storage_path_resolver,
        ffmpeg_command_runner=ffmpeg_command_runner,
    )

    extracted_frame_assets = video_frame_extraction_service.extract_candidate_frames_from_generated_video(
        generated_video_asset
    )

    assert extracted_frame_assets
    assert extracted_frame_assets == sorted(
        extracted_frame_assets,
        key=lambda extracted_frame_asset: extracted_frame_asset.frame_sequence_number,
    )

    first_extracted_frame_asset = extracted_frame_assets[0]
    assert first_extracted_frame_asset.local_frame_image_file_path.exists()
    assert first_extracted_frame_asset.file_size_bytes > 0
