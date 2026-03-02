from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

ui_jobs_created_total = Counter("ui_jobs_created_total", "Total UI jobs created")
ui_jobs_completed_total = Counter("ui_jobs_completed_total", "Total UI jobs completed")
ui_jobs_failed_total = Counter("ui_jobs_failed_total", "Total UI jobs failed")
ui_cache_hits_total = Counter("ui_cache_hits_total", "Total UI cache hits")
ui_cache_misses_total = Counter("ui_cache_misses_total", "Total UI cache misses")

ui_job_duration_seconds = Histogram("ui_job_duration_seconds", "Total UI job duration in seconds")
ui_orchestrator_subprocess_duration_seconds = Histogram(
    "ui_orchestrator_subprocess_duration_seconds",
    "Orchestrator subprocess duration in seconds",
)
ui_cache_lookup_duration_seconds = Histogram(
    "ui_cache_lookup_duration_seconds",
    "Result cache lookup duration in seconds",
)

ui_jobs_running = Gauge("ui_jobs_running", "Current running UI jobs")
cache_index_entries = Gauge("cache_index_entries", "Current cache index entries")


def export_prometheus_metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
