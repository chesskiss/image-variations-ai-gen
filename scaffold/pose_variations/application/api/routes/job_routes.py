from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError

from pose_variations.domain.generation_job_models import GenerationJobIdentifier
from pose_variations.domain.pose_preset_models import PosePresetIdentifier
from pose_variations.infrastructure.settings import ApplicationSettings

router = APIRouter(tags=["jobs"])


@router.post("/jobs")
async def create_generation_job(
    request: Request,
    background_tasks: BackgroundTasks,
    uploaded_image_file: UploadFile = File(...),
    selected_pose_preset_identifiers: list[str] = Form(...),
) -> RedirectResponse:
    generation_job_processor = request.app.state.generation_job_processor
    application_settings: ApplicationSettings = request.app.state.application_settings

    if len(selected_pose_preset_identifiers) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly two pose presets must be selected.",
        )

    if not all(
        generation_job_processor.is_valid_pose_preset_identifier(selected_pose_preset_identifier)
        for selected_pose_preset_identifier in selected_pose_preset_identifiers
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more selected pose presets are invalid.",
        )

    uploaded_image_content = await uploaded_image_file.read()
    sniffed_mime_type = validate_uploaded_image_payload(
        uploaded_image_file=uploaded_image_file,
        uploaded_image_content=uploaded_image_content,
        application_settings=application_settings,
    )

    generation_job_identifier = generation_job_processor.create_generation_job(
        uploaded_image_content=uploaded_image_content,
        uploaded_image_mime_type=sniffed_mime_type,
        uploaded_image_original_file_name=uploaded_image_file.filename or "uploaded-image",
        selected_pose_preset_identifiers=[
            PosePresetIdentifier(pose_preset_identifier)
            for pose_preset_identifier in selected_pose_preset_identifiers
        ],
    )
    generation_job_processor.enqueue_generation_job(background_tasks, generation_job_identifier)

    return RedirectResponse(
        url=f"/jobs/{generation_job_identifier}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/jobs/{generation_job_identifier}", response_class=HTMLResponse)
async def render_generation_job_page(
    request: Request,
    generation_job_identifier: str,
) -> HTMLResponse:
    generation_job_processor = request.app.state.generation_job_processor
    template_renderer = request.app.state.template_renderer

    serialized_job = generation_job_processor.serialize_generation_job(
        GenerationJobIdentifier(generation_job_identifier)
    )
    if serialized_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation job not found.")

    return template_renderer.TemplateResponse(
        "job.html",
        {
            "request": request,
            "generation_job": serialized_job,
        },
    )


@router.get("/jobs/{generation_job_identifier}/assets/{relative_asset_path:path}")
async def serve_job_asset(
    request: Request,
    generation_job_identifier: str,
    relative_asset_path: str,
) -> FileResponse:
    storage_repository = request.app.state.storage_repository
    safe_asset_path = storage_repository.resolve_safe_job_asset_path(
        generation_job_identifier=GenerationJobIdentifier(generation_job_identifier),
        requested_relative_asset_path=relative_asset_path,
    )
    if safe_asset_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    return FileResponse(path=safe_asset_path)


def validate_uploaded_image_payload(
    uploaded_image_file: UploadFile,
    uploaded_image_content: bytes,
    application_settings: ApplicationSettings,
) -> str:
    if len(uploaded_image_content) > application_settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds configured maximum upload size.",
        )

    provided_mime_type = (uploaded_image_file.content_type or "").lower()
    if provided_mime_type not in application_settings.allowed_upload_mime_type_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file MIME type is not allowed.",
        )

    sniffed_mime_type = sniff_uploaded_image_mime_type(uploaded_image_content)
    if sniffed_mime_type not in application_settings.allowed_upload_mime_type_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file content does not match an allowed image type.",
        )

    return sniffed_mime_type


def sniff_uploaded_image_mime_type(uploaded_image_content: bytes) -> str:
    try:
        with Image.open(BytesIO(uploaded_image_content)) as loaded_image:
            loaded_image.verify()
            image_format = (loaded_image.format or "").upper()
    except UnidentifiedImageError as image_identification_error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from image_identification_error

    image_mime_type_by_format: dict[str, str] = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
    }
    detected_mime_type = image_mime_type_by_format.get(image_format)
    if detected_mime_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG image uploads are supported.",
        )
    return detected_mime_type


def generation_job_to_template_payload(serialized_job: dict[str, Any]) -> dict[str, Any]:
    return serialized_job
