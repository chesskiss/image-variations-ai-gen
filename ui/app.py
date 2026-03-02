from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from single_image_transform.prompts import POSE_PRESET_PROMPTS_BY_NAME
from ui.cache_index_repository import CacheIndexRepository
from ui.cache_key_builder import build_result_cache_key
from ui.cache_models import CacheIndexEntry
from ui.history_repository import HistoryRepository
from ui.logging import configure_structured_logging
from ui.metrics import (
    cache_index_entries,
    export_prometheus_metrics_payload,
    ui_cache_hits_total,
    ui_cache_lookup_duration_seconds,
    ui_cache_misses_total,
    ui_job_duration_seconds,
    ui_jobs_completed_total,
    ui_jobs_created_total,
    ui_jobs_failed_total,
    ui_jobs_running,
    ui_orchestrator_subprocess_duration_seconds,
)
from ui.settings import UiSettings, load_ui_settings

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
UI_DIRECTORY = Path(__file__).resolve().parent
UPLOAD_STORAGE_DIRECTORY = PROJECT_ROOT_DIRECTORY / "storage" / "ui_uploads"
ORCHESTRATOR_OUTPUT_DIRECTORY = PROJECT_ROOT_DIRECTORY / "outputs" / "orchestrator_ui"
ORCHESTRATOR_JOB_STATUS_DIRECTORY = ORCHESTRATOR_OUTPUT_DIRECTORY / "jobs"

JOB_STATE_QUEUED = "queued"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"

ui_settings: UiSettings = load_ui_settings()
cache_index_repository = CacheIndexRepository(
    cache_index_directory=ui_settings.cache_index_directory,
    cache_max_entries=ui_settings.cache_max_entries,
    cache_retention_days=ui_settings.cache_retention_days,
)
application_logger: logging.Logger = configure_structured_logging()

UPLOADED_IMAGE_FILE_FORM_PARAMETER = File(...)
SELECTED_POSE_PRESET_IDENTIFIER_FORM_PARAMETER = Form(...)

_job_status_lock = Lock()
_background_process_by_job_identifier: dict[str, subprocess.Popen[str]] = {}


def _utc_now_isoformat() -> str:
    return datetime.now(UTC).isoformat()


def _list_pose_presets_for_display() -> list[dict[str, str]]:
    return [
        {
            "pose_preset_identifier": pose_preset_identifier,
            "pose_preset_display_name": pose_preset_identifier.replace("_", " ").title(),
        }
        for pose_preset_identifier in sorted(POSE_PRESET_PROMPTS_BY_NAME.keys())
    ]


def _sniff_image_extension(uploaded_image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(uploaded_image_bytes)) as uploaded_image:
            uploaded_image.verify()
            image_format = (uploaded_image.format or "").upper()
    except UnidentifiedImageError as image_error:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from image_error

    extension_by_format = {
        "JPEG": ".jpg",
        "PNG": ".png",
    }
    image_extension = extension_by_format.get(image_format)
    if image_extension is None:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG uploads are supported.")
    return image_extension


def _to_artifact_url(absolute_file_path: Path) -> str:
    relative_file_path = absolute_file_path.resolve().relative_to(PROJECT_ROOT_DIRECTORY)
    return f"/artifacts/{relative_file_path.as_posix()}"


def _sanitize_output_for_display(output_text: str, limit: int = 800) -> str:
    truncated_output_text = output_text.strip()[:limit]
    sanitized_output_text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-***", truncated_output_text)
    sanitized_output_text = re.sub(
        r"(key|token|secret)=[^\s]+", r"\1=***", sanitized_output_text, flags=re.IGNORECASE
    )
    return sanitized_output_text


def _status_file_path_for_job(generation_job_identifier: str) -> Path:
    return ORCHESTRATOR_JOB_STATUS_DIRECTORY / generation_job_identifier / "status.json"


def _write_job_status_payload(
    generation_job_identifier: str,
    job_status_payload: dict[str, object],
) -> None:
    status_file_path = _status_file_path_for_job(generation_job_identifier)
    status_file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_status_file_path = status_file_path.with_suffix(".tmp")
    with _job_status_lock:
        temporary_status_file_path.write_text(
            json.dumps(job_status_payload, indent=2),
            encoding="utf-8",
        )
        temporary_status_file_path.replace(status_file_path)


def _read_job_status_payload(generation_job_identifier: str) -> dict[str, object]:
    status_file_path = _status_file_path_for_job(generation_job_identifier)
    if not status_file_path.exists():
        raise HTTPException(status_code=404, detail="Job status was not found.")
    return json.loads(status_file_path.read_text(encoding="utf-8"))


def _build_orchestrator_command_arguments(
    uploaded_image_file_path: Path,
    selected_pose_preset_identifier: str,
    generation_job_identifier: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "run_pose_pipeline_orchestrator.py",
        "--image",
        str(uploaded_image_file_path),
        "--preset",
        selected_pose_preset_identifier,
        "--job-id",
        generation_job_identifier,
        "--output-directory",
        str(ORCHESTRATOR_OUTPUT_DIRECTORY),
    ]


def _collect_artifact_urls_from_pipeline_summary(
    pipeline_summary_payload: dict[str, object],
) -> dict[str, object]:
    judge_result_output_directory = Path(
        str(pipeline_summary_payload["judge_result_output_directory"])
    )
    selected_frames_directory = judge_result_output_directory / "selected_frames"
    selected_frame_image_urls = [
        _to_artifact_url(selected_frame_image_file_path)
        for selected_frame_image_file_path in sorted(selected_frames_directory.glob("*"))
        if selected_frame_image_file_path.is_file()
    ]

    summary_json_file_path = Path(str(pipeline_summary_payload["pipeline_summary_json_file_path"]))
    judge_decision_json_file_path = judge_result_output_directory / "judge_decision.json"
    ranked_scores_json_file_path = judge_result_output_directory / "ranked_scores.json"

    return {
        "summary_json_url": _to_artifact_url(summary_json_file_path),
        "judge_decision_json_url": _to_artifact_url(judge_decision_json_file_path),
        "ranked_scores_json_url": _to_artifact_url(ranked_scores_json_file_path),
        "selected_frame_image_urls": selected_frame_image_urls,
    }


def _update_cache_index_from_pipeline_summary(
    cache_key: str,
    original_image_sha256: str,
    source_job_id: str,
    pose_preset_identifier: str,
    pipeline_summary_json_file_path: Path,
) -> None:
    try:
        pipeline_summary_payload = json.loads(
            pipeline_summary_json_file_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return

    judge_result_output_directory = Path(
        str(pipeline_summary_payload.get("judge_result_output_directory", ""))
    )
    selected_frames_directory = judge_result_output_directory / "selected_frames"
    selected_frame_image_file_paths = [
        str(frame_path.resolve())
        for frame_path in sorted(selected_frames_directory.glob("*"))
        if frame_path.is_file()
    ]

    generated_video_file_path = str(
        Path(str(pipeline_summary_payload.get("generated_video_file_path", ""))).resolve()
    )
    now_timestamp = _utc_now_isoformat()
    cache_index_entry = CacheIndexEntry(
        cache_key=cache_key,
        created_at_utc=now_timestamp,
        last_accessed_at_utc=now_timestamp,
        source_job_id=source_job_id,
        pose_preset_identifier=pose_preset_identifier,
        original_image_sha256=original_image_sha256,
        pipeline_summary_json_file_path=str(pipeline_summary_json_file_path.resolve()),
        judge_result_output_directory=str(judge_result_output_directory.resolve()),
        selected_frame_image_file_paths=selected_frame_image_file_paths,
        generated_video_file_path=generated_video_file_path,
        storage_paths_exist=True,
    )
    cache_index_repository.upsert_cache_entry(cache_index_entry)
    cache_index_entries.set(len(cache_index_repository.list_cache_entries()))


def _watch_orchestrator_process_completion(
    generation_job_identifier: str,
    orchestrator_process: subprocess.Popen[str],
    orchestrator_stderr_log_file_path: Path,
) -> None:
    process_start_monotonic_time = time.perf_counter()
    timeout_seconds = 3600
    try:
        orchestrator_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        orchestrator_process.kill()

    ui_orchestrator_subprocess_duration_seconds.observe(
        time.perf_counter() - process_start_monotonic_time
    )
    _background_process_by_job_identifier.pop(generation_job_identifier, None)
    ui_jobs_running.dec()

    summary_json_file_path = (
        ORCHESTRATOR_OUTPUT_DIRECTORY
        / "pipeline_orchestrator"
        / f"pipeline_summary_{generation_job_identifier}.json"
    )

    existing_job_status_payload = _read_job_status_payload(generation_job_identifier)
    started_at_raw = existing_job_status_payload.get("started_at_utc")
    pipeline_duration_seconds: float | None = None
    if isinstance(started_at_raw, str):
        try:
            started_at_utc_datetime = datetime.fromisoformat(started_at_raw)
            pipeline_duration_seconds = (
                datetime.now(UTC) - started_at_utc_datetime
            ).total_seconds()
        except ValueError:
            pipeline_duration_seconds = None

    if orchestrator_process.returncode == 0 and summary_json_file_path.exists():
        existing_job_status_payload.update(
            {
                "state": JOB_STATE_COMPLETED,
                "progress_percent": 100,
                "current_stage": "completed",
                "finished_at_utc": _utc_now_isoformat(),
                "orchestrator_return_code": orchestrator_process.returncode,
                "pipeline_summary_json_file_path": str(summary_json_file_path),
                "pipeline_duration_seconds": pipeline_duration_seconds,
                "error_message": None,
            }
        )
        if existing_job_status_payload.get("cache_key") and existing_job_status_payload.get(
            "original_image_sha256"
        ):
            _update_cache_index_from_pipeline_summary(
                cache_key=str(existing_job_status_payload["cache_key"]),
                original_image_sha256=str(existing_job_status_payload["original_image_sha256"]),
                source_job_id=generation_job_identifier,
                pose_preset_identifier=str(existing_job_status_payload["pose_preset_identifier"]),
                pipeline_summary_json_file_path=summary_json_file_path,
            )
        ui_jobs_completed_total.inc()
        if pipeline_duration_seconds is not None:
            ui_job_duration_seconds.observe(pipeline_duration_seconds)
    else:
        orchestrator_stderr_text = ""
        if orchestrator_stderr_log_file_path.exists():
            orchestrator_stderr_text = orchestrator_stderr_log_file_path.read_text(encoding="utf-8")

        existing_job_status_payload.update(
            {
                "state": JOB_STATE_FAILED,
                "progress_percent": 100,
                "current_stage": "failed",
                "finished_at_utc": _utc_now_isoformat(),
                "orchestrator_return_code": orchestrator_process.returncode,
                "pipeline_duration_seconds": pipeline_duration_seconds,
                "error_message": _sanitize_output_for_display(
                    "Orchestrator execution failed. " + orchestrator_stderr_text,
                ),
            }
        )
        ui_jobs_failed_total.inc()

    _write_job_status_payload(generation_job_identifier, existing_job_status_payload)


def _launch_orchestrator_background_process(
    generation_job_identifier: str,
    orchestrator_command_arguments: list[str],
) -> None:
    job_output_directory = ORCHESTRATOR_JOB_STATUS_DIRECTORY / generation_job_identifier
    job_output_directory.mkdir(parents=True, exist_ok=True)
    orchestrator_stdout_log_file_path = job_output_directory / "orchestrator_stdout.log"
    orchestrator_stderr_log_file_path = job_output_directory / "orchestrator_stderr.log"

    stdout_log_file_handle = orchestrator_stdout_log_file_path.open("w", encoding="utf-8")
    stderr_log_file_handle = orchestrator_stderr_log_file_path.open("w", encoding="utf-8")

    try:
        orchestrator_process = subprocess.Popen(
            orchestrator_command_arguments,
            cwd=str(PROJECT_ROOT_DIRECTORY),
            stdout=stdout_log_file_handle,
            stderr=stderr_log_file_handle,
            text=True,
        )
    except OSError as process_error:
        stdout_log_file_handle.close()
        stderr_log_file_handle.close()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to launch orchestrator process: {process_error}",
        ) from process_error

    stdout_log_file_handle.close()
    stderr_log_file_handle.close()

    _background_process_by_job_identifier[generation_job_identifier] = orchestrator_process
    ui_jobs_running.inc()

    completion_watcher_thread = Thread(
        target=_watch_orchestrator_process_completion,
        kwargs={
            "generation_job_identifier": generation_job_identifier,
            "orchestrator_process": orchestrator_process,
            "orchestrator_stderr_log_file_path": orchestrator_stderr_log_file_path,
        },
        daemon=True,
    )
    completion_watcher_thread.start()


def _normalize_job_status_payload_for_api(
    job_status_payload: dict[str, object],
) -> dict[str, object]:
    pipeline_summary_payload: dict[str, object] | None = None
    artifact_urls: dict[str, object] | None = None

    pipeline_summary_json_file_path = job_status_payload.get("pipeline_summary_json_file_path")
    if isinstance(pipeline_summary_json_file_path, str) and pipeline_summary_json_file_path:
        summary_path = Path(pipeline_summary_json_file_path)
        if summary_path.exists():
            pipeline_summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            pipeline_summary_payload["pipeline_summary_json_file_path"] = str(summary_path)
            artifact_urls = _collect_artifact_urls_from_pipeline_summary(pipeline_summary_payload)

    return {
        "job_id": job_status_payload.get("job_id"),
        "state": job_status_payload.get("state"),
        "progress_percent": job_status_payload.get("progress_percent"),
        "current_stage": job_status_payload.get("current_stage"),
        "error_message": job_status_payload.get("error_message"),
        "started_at_utc": job_status_payload.get("started_at_utc"),
        "finished_at_utc": job_status_payload.get("finished_at_utc"),
        "was_cache_hit": bool(job_status_payload.get("was_cache_hit", False)),
        "cache_key": job_status_payload.get("cache_key"),
        "cache_source_job_id": job_status_payload.get("cache_source_job_id"),
        "pipeline_duration_seconds": job_status_payload.get("pipeline_duration_seconds"),
        "cache_lookup_duration_milliseconds": job_status_payload.get(
            "cache_lookup_duration_milliseconds"
        ),
        "pipeline_summary_payload": pipeline_summary_payload,
        "artifact_urls": artifact_urls,
    }


def mark_orphaned_running_jobs_as_failed() -> None:
    ORCHESTRATOR_JOB_STATUS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for status_file_path in ORCHESTRATOR_JOB_STATUS_DIRECTORY.glob("*/status.json"):
        status_payload = json.loads(status_file_path.read_text(encoding="utf-8"))
        state = status_payload.get("state")
        if state in {JOB_STATE_RUNNING, JOB_STATE_QUEUED}:
            status_payload.update(
                {
                    "state": JOB_STATE_FAILED,
                    "current_stage": "failed_after_restart",
                    "progress_percent": 100,
                    "finished_at_utc": _utc_now_isoformat(),
                    "error_message": "Server restarted before background job completion.",
                }
            )
            _write_job_status_payload(str(status_payload["job_id"]), status_payload)


@asynccontextmanager
async def application_lifespan(application_instance: FastAPI):
    del application_instance
    mark_orphaned_running_jobs_as_failed()
    cache_index_repository.prune_cache_entries()
    cache_index_entries.set(len(cache_index_repository.list_cache_entries()))
    yield


application = FastAPI(title="AI Image Variations UI Bridge", lifespan=application_lifespan)
application.mount("/static", StaticFiles(directory=str(UI_DIRECTORY / "static")), name="static")
template_renderer = Jinja2Templates(directory=str(UI_DIRECTORY / "templates"))


@application.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_identifier = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_identifier = request_identifier
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_identifier
    return response


@application.get("/", response_class=HTMLResponse)
async def render_home_page(request: Request) -> HTMLResponse:
    return template_renderer.TemplateResponse(
        "home.html",
        {
            "request": request,
            "available_pose_presets": _list_pose_presets_for_display(),
            "error_message": None,
        },
    )


@application.get("/history", response_class=HTMLResponse)
async def render_history_page(request: Request) -> HTMLResponse:
    return template_renderer.TemplateResponse(
        "history.html",
        {
            "request": request,
        },
    )


@application.post("/jobs")
async def create_orchestrator_job(
    request: Request,
    uploaded_image_file: UploadFile = UPLOADED_IMAGE_FILE_FORM_PARAMETER,
    selected_pose_preset_identifier: str = SELECTED_POSE_PRESET_IDENTIFIER_FORM_PARAMETER,
) -> RedirectResponse:
    request_identifier = str(request.state.request_identifier)

    if selected_pose_preset_identifier not in POSE_PRESET_PROMPTS_BY_NAME:
        raise HTTPException(status_code=400, detail="Invalid pose preset identifier.")

    uploaded_image_bytes = await uploaded_image_file.read()
    if len(uploaded_image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    image_extension = _sniff_image_extension(uploaded_image_bytes)

    generation_job_identifier = uuid4().hex
    upload_job_directory = UPLOAD_STORAGE_DIRECTORY / generation_job_identifier
    upload_job_directory.mkdir(parents=True, exist_ok=True)
    uploaded_image_file_path = upload_job_directory / f"original{image_extension}"
    uploaded_image_file_path.write_bytes(uploaded_image_bytes)

    cache_lookup_start_monotonic_time = time.perf_counter()
    cache_key, original_image_sha256 = build_result_cache_key(
        uploaded_image_bytes=uploaded_image_bytes,
        pose_preset_identifier=selected_pose_preset_identifier,
        ui_settings=ui_settings,
    )
    cache_lookup_result = cache_index_repository.get_cache_entry(cache_key)
    cache_lookup_duration_seconds = time.perf_counter() - cache_lookup_start_monotonic_time
    ui_cache_lookup_duration_seconds.observe(cache_lookup_duration_seconds)
    cache_lookup_duration_milliseconds = int(cache_lookup_duration_seconds * 1000)

    ui_jobs_created_total.inc()

    if (
        ui_settings.enable_result_cache
        and cache_lookup_result.cache_hit
        and cache_lookup_result.cache_index_entry
    ):
        cache_index_entry = cache_lookup_result.cache_index_entry
        cache_job_status_payload: dict[str, object] = {
            "job_id": generation_job_identifier,
            "request_id": request_identifier,
            "state": JOB_STATE_COMPLETED,
            "progress_percent": 100,
            "current_stage": "completed_from_cache",
            "pose_preset_identifier": selected_pose_preset_identifier,
            "uploaded_image_file_path": str(uploaded_image_file_path),
            "orchestrator_command_arguments": [],
            "started_at_utc": _utc_now_isoformat(),
            "finished_at_utc": _utc_now_isoformat(),
            "orchestrator_return_code": 0,
            "orchestrator_stdout_log_file_path": None,
            "orchestrator_stderr_log_file_path": None,
            "pipeline_summary_json_file_path": cache_index_entry.pipeline_summary_json_file_path,
            "error_message": None,
            "cache_key": cache_key,
            "original_image_sha256": original_image_sha256,
            "was_cache_hit": True,
            "cache_source_job_id": cache_index_entry.source_job_id,
            "pipeline_duration_seconds": 0.0,
            "cache_lookup_duration_milliseconds": cache_lookup_duration_milliseconds,
        }
        _write_job_status_payload(generation_job_identifier, cache_job_status_payload)
        ui_cache_hits_total.inc()
        ui_jobs_completed_total.inc()
        cache_index_entries.set(len(cache_index_repository.list_cache_entries()))
        application_logger.info(
            "cache_hit",
            extra={
                "component": "ui",
                "request_id": request_identifier,
                "job_id": generation_job_identifier,
                "cache_key": cache_key,
            },
        )
        return RedirectResponse(url=f"/jobs/{generation_job_identifier}", status_code=303)

    ui_cache_misses_total.inc()
    orchestrator_command_arguments = _build_orchestrator_command_arguments(
        uploaded_image_file_path=uploaded_image_file_path,
        selected_pose_preset_identifier=selected_pose_preset_identifier,
        generation_job_identifier=generation_job_identifier,
    )

    base_job_status_payload: dict[str, object] = {
        "job_id": generation_job_identifier,
        "request_id": request_identifier,
        "state": JOB_STATE_QUEUED,
        "progress_percent": 10,
        "current_stage": "uploaded",
        "pose_preset_identifier": selected_pose_preset_identifier,
        "uploaded_image_file_path": str(uploaded_image_file_path),
        "orchestrator_command_arguments": orchestrator_command_arguments,
        "started_at_utc": _utc_now_isoformat(),
        "finished_at_utc": None,
        "orchestrator_return_code": None,
        "orchestrator_stdout_log_file_path": str(
            ORCHESTRATOR_JOB_STATUS_DIRECTORY
            / generation_job_identifier
            / "orchestrator_stdout.log"
        ),
        "orchestrator_stderr_log_file_path": str(
            ORCHESTRATOR_JOB_STATUS_DIRECTORY
            / generation_job_identifier
            / "orchestrator_stderr.log"
        ),
        "pipeline_summary_json_file_path": None,
        "error_message": None,
        "cache_key": cache_key,
        "original_image_sha256": original_image_sha256,
        "was_cache_hit": False,
        "cache_source_job_id": None,
        "pipeline_duration_seconds": None,
        "cache_lookup_duration_milliseconds": cache_lookup_duration_milliseconds,
    }
    _write_job_status_payload(generation_job_identifier, base_job_status_payload)

    _launch_orchestrator_background_process(
        generation_job_identifier, orchestrator_command_arguments
    )

    running_job_status_payload = dict(base_job_status_payload)
    running_job_status_payload.update(
        {
            "state": JOB_STATE_RUNNING,
            "progress_percent": 20,
            "current_stage": "running_orchestrator",
        }
    )
    _write_job_status_payload(generation_job_identifier, running_job_status_payload)

    application_logger.info(
        "job_created",
        extra={
            "component": "ui",
            "request_id": request_identifier,
            "job_id": generation_job_identifier,
            "cache_key": cache_key,
        },
    )

    return RedirectResponse(url=f"/jobs/{generation_job_identifier}", status_code=303)


@application.get("/jobs/{generation_job_identifier}", response_class=HTMLResponse)
async def render_job_page(request: Request, generation_job_identifier: str) -> HTMLResponse:
    job_status_payload = _read_job_status_payload(generation_job_identifier)
    return template_renderer.TemplateResponse(
        "job.html",
        {
            "request": request,
            "generation_job_identifier": generation_job_identifier,
            "pose_preset_identifier": job_status_payload.get("pose_preset_identifier"),
        },
    )


@application.get("/api/jobs/{generation_job_identifier}")
async def get_job_status(generation_job_identifier: str) -> JSONResponse:
    job_status_payload = _read_job_status_payload(generation_job_identifier)
    normalized_job_status_payload = _normalize_job_status_payload_for_api(job_status_payload)
    return JSONResponse(normalized_job_status_payload)


@application.get("/api/cache/{cache_key}")
async def get_cache_entry(cache_key: str) -> JSONResponse:
    cache_lookup_result = cache_index_repository.get_cache_entry(cache_key)
    if not cache_lookup_result.cache_hit or cache_lookup_result.cache_index_entry is None:
        return JSONResponse(
            {"cache_key": cache_key, "cache_hit": False, "reason": cache_lookup_result.reason}
        )

    return JSONResponse(
        {
            "cache_key": cache_key,
            "cache_hit": True,
            "reason": cache_lookup_result.reason,
            "entry": asdict(cache_lookup_result.cache_index_entry),
        }
    )


@application.get("/api/history")
async def get_job_history(
    limit: int = Query(default=50, ge=1, le=500),
    state: str | None = Query(default=None),
) -> JSONResponse:
    history_repository = HistoryRepository(ORCHESTRATOR_JOB_STATUS_DIRECTORY)
    history_entries = history_repository.list_history_entries(limit=limit, state_filter=state)
    for history_entry in history_entries:
        thumbnail_url = history_entry.get("thumbnail_url")
        if isinstance(thumbnail_url, str):
            history_entry["thumbnail_url"] = _to_artifact_url(Path(thumbnail_url))
    return JSONResponse({"entries": history_entries})


@application.get("/metrics")
async def get_prometheus_metrics() -> Response:
    metrics_payload, content_type = export_prometheus_metrics_payload()
    return Response(content=metrics_payload, media_type=content_type)


@application.get("/artifacts/{artifact_relative_path:path}")
async def serve_artifact_file(artifact_relative_path: str) -> FileResponse:
    resolved_artifact_file_path = (PROJECT_ROOT_DIRECTORY / artifact_relative_path).resolve()
    if not resolved_artifact_file_path.exists() or not resolved_artifact_file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    allowed_artifact_roots = [
        (PROJECT_ROOT_DIRECTORY / "storage").resolve(),
        (PROJECT_ROOT_DIRECTORY / "outputs").resolve(),
    ]
    if not any(
        str(resolved_artifact_file_path).startswith(str(allowed_artifact_root))
        for allowed_artifact_root in allowed_artifact_roots
    ):
        raise HTTPException(status_code=403, detail="Forbidden artifact path.")

    return FileResponse(path=resolved_artifact_file_path)
