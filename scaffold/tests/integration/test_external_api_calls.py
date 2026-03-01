from __future__ import annotations

import os
from pathlib import Path

import pytest

from pose_variations.domain.asset_models import AssetIdentifier, UploadedImageAsset
from pose_variations.infrastructure.settings import ApplicationSettings
from pose_variations.services.fal_image_to_video_client import FalImageToVideoClient
from pose_variations.services.openai_identity_similarity_judge import OpenAiIdentitySimilarityJudge


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_FAL_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Set RUN_FAL_INTEGRATION_TESTS=true to run real fal.ai integration tests.",
)
@pytest.mark.asyncio
async def test_real_fal_image_to_video_call() -> None:
    application_settings = ApplicationSettings()
    if not application_settings.fal_api_key:
        pytest.skip("FAL_API_KEY is required for real integration test.")

    uploaded_image_asset = UploadedImageAsset(
        asset_identifier=AssetIdentifier("integration-uploaded"),
        local_file_path=Path("tests/assets/integration_placeholder.jpg"),
        public_asset_url="",
        mime_type="image/jpeg",
        original_file_name="integration_placeholder.jpg",
    )
    if not uploaded_image_asset.local_file_path.exists():
        pytest.skip("Provide tests/assets/integration_placeholder.jpg for fal integration test.")

    fal_image_to_video_client = FalImageToVideoClient(application_settings)
    fal_request_identifier = await fal_image_to_video_client.submit_image_to_video_generation(
        uploaded_image_asset=uploaded_image_asset,
        prompt_template="Rotate right while preserving identity.",
    )
    generated_video_url = await fal_image_to_video_client.wait_for_generation_result(
        fal_request_identifier
    )

    assert str(generated_video_url).startswith("http")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Set RUN_OPENAI_INTEGRATION_TESTS=true to run real OpenAI integration tests.",
)
@pytest.mark.asyncio
async def test_real_openai_similarity_call() -> None:
    application_settings = ApplicationSettings(enable_openai_similarity_judge=True)
    if not application_settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for real integration test.")

    fixture_image_path = Path("tests/assets/integration_placeholder.jpg")
    if not fixture_image_path.exists():
        pytest.skip("Provide tests/assets/integration_placeholder.jpg for OpenAI integration test.")

    openai_identity_similarity_judge = OpenAiIdentitySimilarityJudge(application_settings)
    identity_similarity_score = await openai_identity_similarity_judge.score_identity_similarity(
        original_image_file_path=fixture_image_path,
        candidate_image_file_path=fixture_image_path,
    )
    assert 0.0 <= float(identity_similarity_score) <= 1.0
