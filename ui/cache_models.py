from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CacheIndexEntry:
    cache_key: str
    created_at_utc: str
    last_accessed_at_utc: str
    source_job_id: str
    pose_preset_identifier: str
    original_image_sha256: str
    pipeline_summary_json_file_path: str
    judge_result_output_directory: str
    selected_frame_image_file_paths: list[str]
    generated_video_file_path: str
    storage_paths_exist: bool


@dataclass(slots=True, frozen=True)
class CacheLookupResult:
    cache_hit: bool
    cache_index_entry: CacheIndexEntry | None
    reason: str
