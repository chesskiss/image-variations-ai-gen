from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ui.cache_index_repository import CacheIndexRepository
from ui.cache_models import CacheIndexEntry


def test_cache_repository_handles_corrupt_index_file(tmp_path: Path) -> None:
    cache_index_directory = tmp_path / "cache-index"
    cache_index_directory.mkdir(parents=True, exist_ok=True)
    (cache_index_directory / "cache_index.json").write_text("{invalid-json", encoding="utf-8")

    cache_index_repository = CacheIndexRepository(
        cache_index_directory=cache_index_directory,
        cache_max_entries=100,
        cache_retention_days=30,
    )

    cache_lookup_result = cache_index_repository.get_cache_entry("unknown")

    assert cache_lookup_result.cache_hit is False
    corrupt_files = list(cache_index_directory.glob("cache_index.corrupt.*.json"))
    assert len(corrupt_files) == 1


def test_cache_repository_marks_stale_entry_as_miss(tmp_path: Path) -> None:
    cache_index_repository = CacheIndexRepository(
        cache_index_directory=tmp_path / "cache-index",
        cache_max_entries=100,
        cache_retention_days=30,
    )

    stale_cache_index_entry = CacheIndexEntry(
        cache_key="cache-key-1",
        created_at_utc=datetime.now(UTC).isoformat(),
        last_accessed_at_utc=datetime.now(UTC).isoformat(),
        source_job_id="job-1",
        pose_preset_identifier="rotate_left",
        original_image_sha256="abc",
        pipeline_summary_json_file_path=str(tmp_path / "missing_summary.json"),
        judge_result_output_directory=str(tmp_path / "missing_judge_directory"),
        selected_frame_image_file_paths=[str(tmp_path / "missing_frame.jpg")],
        generated_video_file_path=str(tmp_path / "missing_generated.mp4"),
        storage_paths_exist=True,
    )
    cache_index_repository.upsert_cache_entry(stale_cache_index_entry)

    cache_lookup_result = cache_index_repository.get_cache_entry("cache-key-1")

    assert cache_lookup_result.cache_hit is False
    assert cache_lookup_result.reason == "stale"

    cache_index_file_path = tmp_path / "cache-index" / "cache_index.json"
    payload = json.loads(cache_index_file_path.read_text(encoding="utf-8"))
    assert payload["entries"][0]["storage_paths_exist"] is False
