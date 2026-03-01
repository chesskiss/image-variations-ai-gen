from __future__ import annotations

import json
import subprocess
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from single_image_transform.prompts import POSE_PRESET_PROMPTS_BY_NAME

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
SCAFFOLD_DIRECTORY = Path(__file__).resolve().parent
UPLOAD_STORAGE_DIRECTORY = PROJECT_ROOT_DIRECTORY / "storage" / "ui_uploads"
ORCHESTRATOR_OUTPUT_DIRECTORY = PROJECT_ROOT_DIRECTORY / "outputs" / "orchestrator_ui"

application = FastAPI(title="PoseVariations UI Bridge")
application.mount("/static", StaticFiles(directory=str(SCAFFOLD_DIRECTORY / "static")), name="static")
template_renderer = Jinja2Templates(directory=str(SCAFFOLD_DIRECTORY / "templates"))


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
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from image_error

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


@application.post("/run", response_class=HTMLResponse)
async def run_orchestrator_job(
    request: Request,
    uploaded_image_file: UploadFile = File(...),
    selected_pose_preset_identifier: str = Form(...),
) -> HTMLResponse:
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

    orchestrator_command_arguments = [
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

    orchestrator_process_result = subprocess.run(
        orchestrator_command_arguments,
        cwd=str(PROJECT_ROOT_DIRECTORY),
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )

    if orchestrator_process_result.returncode != 0:
        return template_renderer.TemplateResponse(
            "home.html",
            {
                "request": request,
                "available_pose_presets": _list_pose_presets_for_display(),
                "error_message": (
                    "Orchestrator execution failed. "
                    f"{orchestrator_process_result.stderr.strip()[:800]}"
                ),
            },
            status_code=500,
        )

    summary_json_file_path = (
        ORCHESTRATOR_OUTPUT_DIRECTORY
        / "pipeline_orchestrator"
        / f"pipeline_summary_{generation_job_identifier}.json"
    )
    if not summary_json_file_path.exists():
        return template_renderer.TemplateResponse(
            "home.html",
            {
                "request": request,
                "available_pose_presets": _list_pose_presets_for_display(),
                "error_message": "Orchestrator completed but summary JSON was not found.",
            },
            status_code=500,
        )

    pipeline_summary_payload = json.loads(summary_json_file_path.read_text(encoding="utf-8"))

    judge_result_output_directory = Path(pipeline_summary_payload["judge_result_output_directory"]).resolve()
    selected_frames_directory = judge_result_output_directory / "selected_frames"
    selected_frame_image_urls = [
        _to_artifact_url(selected_frame_image_file_path)
        for selected_frame_image_file_path in sorted(selected_frames_directory.glob("*"))
        if selected_frame_image_file_path.is_file()
    ]

    return template_renderer.TemplateResponse(
        "job.html",
        {
            "request": request,
            "pipeline_summary_payload": pipeline_summary_payload,
            "summary_json_url": _to_artifact_url(summary_json_file_path),
            "judge_decision_json_url": _to_artifact_url(judge_result_output_directory / "judge_decision.json"),
            "ranked_scores_json_url": _to_artifact_url(judge_result_output_directory / "ranked_scores.json"),
            "selected_frame_image_urls": selected_frame_image_urls,
            "orchestrator_standard_output": orchestrator_process_result.stdout.strip(),
        },
    )


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
