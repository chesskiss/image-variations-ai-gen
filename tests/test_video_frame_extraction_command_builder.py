from __future__ import annotations

from pathlib import Path

from frame_extraction_stage.video_frame_extraction_service import VideoFrameExtractionService


def test_ffmpeg_command_builder_for_uniform_fps_sampling() -> None:
    command_arguments = VideoFrameExtractionService.build_ffmpeg_command_arguments_for_uniform_fps_sampling(
        ffmpeg_executable_path="ffmpeg",
        video_file_path_for_generation_job=Path("/storage/job-1/preset-a/video.mp4"),
        deterministic_output_file_pattern=Path("/storage/job-1/preset-a/frames/frame_%04d.jpg"),
        extracted_frames_per_second=2,
        max_extracted_frames_per_video=24,
    )

    assert command_arguments == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        "/storage/job-1/preset-a/video.mp4",
        "-vf",
        "fps=2",
        "-vsync",
        "vfr",
        "-frames:v",
        "24",
        "/storage/job-1/preset-a/frames/frame_%04d.jpg",
    ]
