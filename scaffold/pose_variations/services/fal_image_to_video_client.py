from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from pose_variations.domain.asset_models import (
    FalGenerationRequestIdentifier,
    GeneratedVideoUrl,
    UploadedImageAsset,
)
from pose_variations.infrastructure.settings import ApplicationSettings


class FalImageToVideoClientError(RuntimeError):
    """Raised when fal.ai integration fails."""


class FalImageToVideoClient:
    def __init__(self, application_settings: ApplicationSettings) -> None:
        self._application_settings = application_settings
        self._fal_queue_base_url = "https://queue.fal.run"
        self._fal_file_upload_base_url = "https://api.fal.ai/v1/serverless/files/file/local"

    def _authorization_headers(self) -> dict[str, str]:
        if not self._application_settings.fal_api_key:
            raise FalImageToVideoClientError("FAL_API_KEY must be configured.")
        return {"Authorization": f"Key {self._application_settings.fal_api_key}"}

    async def submit_image_to_video_generation(
        self,
        uploaded_image_asset: UploadedImageAsset,
        prompt_template: str,
    ) -> FalGenerationRequestIdentifier:
        uploaded_image_url = await self._upload_image_file_to_fal_storage(uploaded_image_asset.local_file_path)

        model_submission_url = f"{self._fal_queue_base_url}/{self._application_settings.fal_model_id}"
        request_payload = {
            "input": {
                "prompt": prompt_template,
                "image_url": uploaded_image_url,
                "duration": 5,
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as asynchronous_http_client:
            submission_response = await asynchronous_http_client.post(
                model_submission_url,
                headers=self._authorization_headers(),
                json=request_payload,
            )
            submission_response.raise_for_status()
            submission_payload = submission_response.json()

        request_identifier = submission_payload.get("request_id") or submission_payload.get("id")
        if not isinstance(request_identifier, str) or not request_identifier:
            raise FalImageToVideoClientError(
                f"fal.ai submission response did not include request identifier: {submission_payload}"
            )

        return FalGenerationRequestIdentifier(request_identifier)

    async def wait_for_generation_result(
        self,
        fal_generation_request_identifier: FalGenerationRequestIdentifier,
    ) -> GeneratedVideoUrl:
        status_endpoint_url = (
            f"{self._fal_queue_base_url}/{self._application_settings.fal_model_id}/requests/"
            f"{fal_generation_request_identifier}/status"
        )
        result_endpoint_url = (
            f"{self._fal_queue_base_url}/{self._application_settings.fal_model_id}/requests/"
            f"{fal_generation_request_identifier}"
        )

        current_poll_delay_seconds = self._application_settings.fal_poll_interval_seconds

        async with httpx.AsyncClient(timeout=60.0) as asynchronous_http_client:
            for poll_attempt_number in range(self._application_settings.fal_max_poll_attempts):
                status_response = await asynchronous_http_client.get(
                    status_endpoint_url,
                    headers=self._authorization_headers(),
                )
                status_response.raise_for_status()
                status_payload = status_response.json()

                request_status = str(status_payload.get("status", "")).upper()
                if request_status in {"COMPLETED", "SUCCESS", "FINISHED"}:
                    break
                if request_status in {"FAILED", "ERROR", "CANCELLED"}:
                    raise FalImageToVideoClientError(
                        f"fal.ai request failed for {fal_generation_request_identifier}: {status_payload}"
                    )

                if poll_attempt_number == self._application_settings.fal_max_poll_attempts - 1:
                    raise FalImageToVideoClientError(
                        f"fal.ai request timed out for {fal_generation_request_identifier}."
                    )

                await asyncio.sleep(current_poll_delay_seconds)
                current_poll_delay_seconds = min(current_poll_delay_seconds * 1.5, 10.0)

            result_response = await asynchronous_http_client.get(
                result_endpoint_url,
                headers=self._authorization_headers(),
            )
            result_response.raise_for_status()
            result_payload = result_response.json()

        extracted_video_url = self._extract_video_url_from_payload(result_payload)
        return GeneratedVideoUrl(extracted_video_url)

    async def _upload_image_file_to_fal_storage(self, image_file_path: Path) -> str:
        target_path = f"uploads/{image_file_path.name}"
        encoded_target_path = quote(target_path, safe="")
        upload_url = f"{self._fal_file_upload_base_url}/{encoded_target_path}"
        with image_file_path.open("rb") as image_binary_stream:
            files = {"file_upload": (image_file_path.name, image_binary_stream)}
            async with httpx.AsyncClient(timeout=60.0) as asynchronous_http_client:
                upload_response = await asynchronous_http_client.post(
                    upload_url,
                    headers=self._authorization_headers(),
                    files=files,
                )
                upload_response.raise_for_status()
                upload_payload = upload_response.json()

        uploaded_image_url = upload_payload.get("url") or upload_payload.get("access_url")
        if not isinstance(uploaded_image_url, str) or not uploaded_image_url:
            raise FalImageToVideoClientError(
                f"fal.ai storage upload did not return an image URL: {upload_payload}"
            )
        return uploaded_image_url

    def _extract_video_url_from_payload(self, result_payload: dict[str, Any]) -> str:
        for candidate_url in self._iter_candidate_urls(result_payload):
            parsed_candidate_url = urlparse(candidate_url)
            if parsed_candidate_url.scheme in {"https", "http"}:
                return candidate_url
        raise FalImageToVideoClientError(
            f"Unable to locate generated video URL in fal.ai payload: {result_payload}"
        )

    def _iter_candidate_urls(self, payload_value: object) -> Iterator[str]:
        if isinstance(payload_value, dict):
            for dictionary_key, dictionary_value in payload_value.items():
                if dictionary_key == "url" and isinstance(dictionary_value, str):
                    yield dictionary_value
                yield from self._iter_candidate_urls(dictionary_value)
        elif isinstance(payload_value, list):
            for list_item in payload_value:
                yield from self._iter_candidate_urls(list_item)
