from __future__ import annotations

from pathlib import Path

from ui.cache_key_builder import build_result_cache_key
from ui.settings import UiSettings


def _build_ui_settings(cache_key_include_model_version: bool) -> UiSettings:
    return UiSettings(
        enable_result_cache=True,
        cache_index_directory=Path("/tmp"),
        cache_max_entries=100,
        cache_retention_days=30,
        cache_key_include_model_version=cache_key_include_model_version,
        fal_model_id="fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
        enable_openai_frame_judge=True,
        frame_judge_model_name="o3",
        frame_judge_minimum_score_threshold=0.75,
        extracted_frames_per_second=2,
        extracted_frame_image_format="jpg",
        max_extracted_frames_per_video=24,
    )


def test_cache_key_is_deterministic_for_same_input() -> None:
    ui_settings = _build_ui_settings(cache_key_include_model_version=True)
    uploaded_image_bytes = b"same-image-bytes"

    first_cache_key, first_image_sha256 = build_result_cache_key(
        uploaded_image_bytes=uploaded_image_bytes,
        pose_preset_identifier="rotate_left",
        ui_settings=ui_settings,
    )
    second_cache_key, second_image_sha256 = build_result_cache_key(
        uploaded_image_bytes=uploaded_image_bytes,
        pose_preset_identifier="rotate_left",
        ui_settings=ui_settings,
    )

    assert first_cache_key == second_cache_key
    assert first_image_sha256 == second_image_sha256


def test_cache_key_changes_when_pose_preset_changes() -> None:
    ui_settings = _build_ui_settings(cache_key_include_model_version=True)
    uploaded_image_bytes = b"same-image-bytes"

    rotate_left_cache_key, _ = build_result_cache_key(
        uploaded_image_bytes=uploaded_image_bytes,
        pose_preset_identifier="rotate_left",
        ui_settings=ui_settings,
    )
    rotate_right_cache_key, _ = build_result_cache_key(
        uploaded_image_bytes=uploaded_image_bytes,
        pose_preset_identifier="rotate_right",
        ui_settings=ui_settings,
    )

    assert rotate_left_cache_key != rotate_right_cache_key
