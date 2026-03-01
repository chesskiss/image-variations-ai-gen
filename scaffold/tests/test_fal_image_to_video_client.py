from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from pose_variations.domain.asset_models import AssetIdentifier, UploadedImageAsset
from pose_variations.infrastructure.settings import ApplicationSettings
from pose_variations.services.fal_image_to_video_client import FalImageToVideoClient


@pytest.mark.asyncio
@respx.mock
async def test_fal_image_to_video_client_submission_and_polling(tmp_path: Path) -> None:
    application_settings = ApplicationSettings(
        fal_api_key="test-fal-key",
        fal_model_id="fal-ai/test-model",
        fal_poll_interval_seconds=0.01,
        fal_max_poll_attempts=2,
    )
    fal_image_to_video_client = FalImageToVideoClient(application_settings)

    uploaded_file_path = tmp_path / "uploaded.jpg"
    uploaded_file_path.write_bytes(b"dummy-image")

    uploaded_image_asset = UploadedImageAsset(
        asset_identifier=AssetIdentifier("uploaded-asset"),
        local_file_path=uploaded_file_path,
        public_asset_url="/jobs/example/assets/uploads/uploaded.jpg",
        mime_type="image/jpeg",
        original_file_name="uploaded.jpg",
    )

    respx.post("https://api.fal.ai/v1/serverless/files/file/local/uploads%2Fuploaded.jpg").mock(
        return_value=Response(200, json={"url": "https://files.example.com/uploaded.jpg"})
    )
    respx.post("https://queue.fal.run/fal-ai/test-model").mock(
        return_value=Response(200, json={"request_id": "request-123"})
    )
    respx.get("https://queue.fal.run/fal-ai/test-model/requests/request-123/status").mock(
        return_value=Response(200, json={"status": "COMPLETED"})
    )
    respx.get("https://queue.fal.run/fal-ai/test-model/requests/request-123").mock(
        return_value=Response(200, json={"video": {"url": "https://files.example.com/video.mp4"}})
    )

    fal_request_identifier = await fal_image_to_video_client.submit_image_to_video_generation(
        uploaded_image_asset=uploaded_image_asset,
        prompt_template="test prompt",
    )
    generated_video_url = await fal_image_to_video_client.wait_for_generation_result(
        fal_request_identifier
    )

    assert str(fal_request_identifier) == "request-123"
    assert str(generated_video_url) == "https://files.example.com/video.mp4"
