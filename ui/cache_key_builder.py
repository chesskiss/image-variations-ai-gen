from __future__ import annotations

import hashlib
import json

from ui.settings import UiSettings


def calculate_sha256_hexdigest(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def build_result_cache_key(
    uploaded_image_bytes: bytes,
    pose_preset_identifier: str,
    ui_settings: UiSettings,
) -> tuple[str, str]:
    original_image_sha256 = calculate_sha256_hexdigest(uploaded_image_bytes)

    cache_key_payload: dict[str, str | int | float | bool] = {
        "original_image_sha256": original_image_sha256,
        "pose_preset_identifier": pose_preset_identifier,
        "judge_enabled": ui_settings.enable_openai_frame_judge,
        "judge_model": ui_settings.frame_judge_model_name,
        "judge_threshold": ui_settings.frame_judge_minimum_score_threshold,
        "extracted_frames_per_second": ui_settings.extracted_frames_per_second,
        "extracted_frame_image_format": ui_settings.extracted_frame_image_format,
        "max_extracted_frames_per_video": ui_settings.max_extracted_frames_per_video,
    }
    if ui_settings.cache_key_include_model_version:
        cache_key_payload["fal_model_id"] = ui_settings.fal_model_id

    canonical_cache_key_payload = json.dumps(
        cache_key_payload, sort_keys=True, separators=(",", ":")
    )
    cache_key = hashlib.sha256(canonical_cache_key_payload.encode("utf-8")).hexdigest()
    return cache_key, original_image_sha256
