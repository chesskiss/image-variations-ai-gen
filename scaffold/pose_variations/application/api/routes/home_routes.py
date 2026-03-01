from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["home"])


@router.get("/", response_class=HTMLResponse)
async def render_home_page(request: Request) -> HTMLResponse:
    generation_job_processor = request.app.state.generation_job_processor
    template_renderer = request.app.state.template_renderer
    return template_renderer.TemplateResponse(
        "home.html",
        {
            "request": request,
            "available_pose_presets": generation_job_processor.list_pose_presets_for_upload_form(),
        },
    )
