from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pose_variations.application.api.routes import api_job_routes, home_routes, job_routes
from pose_variations.application.background.job_processor import GenerationJobProcessor
from pose_variations.infrastructure.logging_configuration import configure_application_logging
from pose_variations.infrastructure.settings import ApplicationSettings, get_application_settings
from pose_variations.infrastructure.storage_repository import StorageRepository
from pose_variations.services.fal_image_to_video_client import FalImageToVideoClient
from pose_variations.services.frame_quality_assessment_service import FrameQualityAssessmentService
from pose_variations.services.frame_selection_service import FrameSelectionService
from pose_variations.services.openai_identity_similarity_judge import OpenAiIdentitySimilarityJudge
from pose_variations.services.prompt_template_repository import PromptTemplateRepository
from pose_variations.services.video_downloader import VideoDownloader
from pose_variations.services.video_frame_extraction_service import VideoFrameExtractionService


def create_fastapi_application(
    application_settings: ApplicationSettings | None = None,
) -> FastAPI:
    configure_application_logging()
    configured_settings = application_settings or get_application_settings()

    application = FastAPI(title="PoseVariations")

    storage_repository = StorageRepository(configured_settings.storage_directory)
    prompt_template_repository = PromptTemplateRepository()
    fal_image_to_video_client = FalImageToVideoClient(configured_settings)
    video_downloader = VideoDownloader(storage_repository)
    frame_quality_assessment_service = FrameQualityAssessmentService()
    video_frame_extraction_service = VideoFrameExtractionService(
        configured_settings,
        storage_repository,
        frame_quality_assessment_service,
    )
    openai_identity_similarity_judge = (
        OpenAiIdentitySimilarityJudge(configured_settings)
        if configured_settings.enable_openai_similarity_judge
        else None
    )
    frame_selection_service = FrameSelectionService(
        configured_settings,
        frame_quality_assessment_service,
        openai_identity_similarity_judge,
    )

    generation_job_processor = GenerationJobProcessor(
        storage_repository=storage_repository,
        prompt_template_repository=prompt_template_repository,
        fal_image_to_video_client=fal_image_to_video_client,
        video_downloader=video_downloader,
        video_frame_extraction_service=video_frame_extraction_service,
        frame_selection_service=frame_selection_service,
    )

    application.state.application_settings = configured_settings
    application.state.storage_repository = storage_repository
    application.state.generation_job_processor = generation_job_processor
    application.state.template_renderer = Jinja2Templates(directory=str(Path("templates")))

    application.mount("/static", StaticFiles(directory="static"), name="static")

    application.include_router(home_routes.router)
    application.include_router(job_routes.router)
    application.include_router(api_job_routes.router)

    return application


application = create_fastapi_application()
