from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

from ui.cache_models import CacheIndexEntry, CacheLookupResult


class CacheIndexRepository:
    def __init__(
        self,
        cache_index_directory: Path,
        cache_max_entries: int,
        cache_retention_days: int,
    ) -> None:
        self._cache_index_directory = cache_index_directory.resolve()
        self._cache_index_directory.mkdir(parents=True, exist_ok=True)
        self._cache_index_file_path = self._cache_index_directory / "cache_index.json"
        self._cache_max_entries = cache_max_entries
        self._cache_retention_days = cache_retention_days
        self._cache_index_lock = Lock()

    def get_cache_entry(self, cache_key: str) -> CacheLookupResult:
        cache_entries_by_cache_key = self._read_cache_entries_by_cache_key()
        cache_index_entry = cache_entries_by_cache_key.get(cache_key)
        if cache_index_entry is None:
            return CacheLookupResult(cache_hit=False, cache_index_entry=None, reason="miss")

        if not self._validate_cache_entry_paths_exist(cache_index_entry):
            cache_entries_by_cache_key[cache_key] = CacheIndexEntry(
                **{**asdict(cache_index_entry), "storage_paths_exist": False}
            )
            self._write_cache_entries_by_cache_key(cache_entries_by_cache_key)
            return CacheLookupResult(cache_hit=False, cache_index_entry=None, reason="stale")

        refreshed_cache_index_entry = CacheIndexEntry(
            **{**asdict(cache_index_entry), "last_accessed_at_utc": datetime.now(UTC).isoformat()}
        )
        cache_entries_by_cache_key[cache_key] = refreshed_cache_index_entry
        self._write_cache_entries_by_cache_key(cache_entries_by_cache_key)
        return CacheLookupResult(
            cache_hit=True,
            cache_index_entry=refreshed_cache_index_entry,
            reason="hit",
        )

    def upsert_cache_entry(self, cache_index_entry: CacheIndexEntry) -> None:
        cache_entries_by_cache_key = self._read_cache_entries_by_cache_key()
        cache_entries_by_cache_key[cache_index_entry.cache_key] = cache_index_entry
        self._write_cache_entries_by_cache_key(cache_entries_by_cache_key)
        self.prune_cache_entries()

    def prune_cache_entries(self) -> None:
        cache_entries_by_cache_key = self._read_cache_entries_by_cache_key()
        now_utc_datetime = datetime.now(UTC)
        retention_cutoff_datetime = now_utc_datetime - timedelta(days=self._cache_retention_days)

        retained_cache_entries: list[CacheIndexEntry] = []
        for cache_index_entry in cache_entries_by_cache_key.values():
            try:
                entry_last_accessed_datetime = datetime.fromisoformat(
                    cache_index_entry.last_accessed_at_utc
                )
            except ValueError:
                continue
            if entry_last_accessed_datetime >= retention_cutoff_datetime:
                retained_cache_entries.append(cache_index_entry)

        retained_cache_entries.sort(key=lambda entry: entry.last_accessed_at_utc, reverse=True)
        retained_cache_entries = retained_cache_entries[: self._cache_max_entries]

        self._write_cache_entries_by_cache_key(
            {
                cache_index_entry.cache_key: cache_index_entry
                for cache_index_entry in retained_cache_entries
            }
        )

    def list_cache_entries(self) -> list[CacheIndexEntry]:
        return sorted(
            self._read_cache_entries_by_cache_key().values(),
            key=lambda cache_index_entry: cache_index_entry.last_accessed_at_utc,
            reverse=True,
        )

    def _read_cache_entries_by_cache_key(self) -> dict[str, CacheIndexEntry]:
        with self._cache_index_lock:
            if not self._cache_index_file_path.exists():
                return {}

            try:
                cache_index_payload = json.loads(
                    self._cache_index_file_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                corrupt_file_path = (
                    self._cache_index_directory
                    / f"cache_index.corrupt.{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
                )
                self._cache_index_file_path.rename(corrupt_file_path)
                return {}

            raw_entries = cache_index_payload.get("entries", [])
            cache_entries_by_cache_key: dict[str, CacheIndexEntry] = {}
            for raw_entry in raw_entries:
                try:
                    cache_index_entry = CacheIndexEntry(**raw_entry)
                except TypeError:
                    continue
                cache_entries_by_cache_key[cache_index_entry.cache_key] = cache_index_entry
            return cache_entries_by_cache_key

    def _write_cache_entries_by_cache_key(
        self,
        cache_entries_by_cache_key: dict[str, CacheIndexEntry],
    ) -> None:
        with self._cache_index_lock:
            payload = {
                "entries": [
                    asdict(cache_index_entry)
                    for cache_index_entry in cache_entries_by_cache_key.values()
                ]
            }
            temporary_cache_index_file_path = self._cache_index_file_path.with_suffix(".tmp")
            temporary_cache_index_file_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            temporary_cache_index_file_path.replace(self._cache_index_file_path)

    @staticmethod
    def _validate_cache_entry_paths_exist(cache_index_entry: CacheIndexEntry) -> bool:
        required_paths = [
            Path(cache_index_entry.pipeline_summary_json_file_path),
            Path(cache_index_entry.judge_result_output_directory),
            Path(cache_index_entry.generated_video_file_path),
        ] + [
            Path(frame_image_file_path)
            for frame_image_file_path in cache_index_entry.selected_frame_image_file_paths
        ]

        return all(required_path.exists() for required_path in required_paths)
