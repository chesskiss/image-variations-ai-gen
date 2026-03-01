from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from pose_variations.domain.generation_job_models import GenerationJobIdentifier

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/jobs/{generation_job_identifier}")
async def get_generation_job_status(
    request: Request,
    generation_job_identifier: str,
) -> JSONResponse:
    generation_job_processor = request.app.state.generation_job_processor
    serialized_job = generation_job_processor.serialize_generation_job(
        GenerationJobIdentifier(generation_job_identifier)
    )
    if serialized_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation job not found.")

    response = JSONResponse(content=serialized_job)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/jobs/{generation_job_identifier}/events")
async def get_generation_job_events_placeholder() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)
