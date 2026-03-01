from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import fal_client
import httpx

from single_image_transform.config import SingleImageTransformSettings


class SingleImageTransformError(RuntimeError):
    """Raised when single-image transform flow fails."""


class FalImageTransformClient:
    def __init__(self, settings: SingleImageTransformSettings) -> None:
        self._settings = settings
        os.environ["FAL_KEY"] = self._settings.fal_api_key

    def transform_local_image(
        self,
        input_image_file_path: Path,
        prompt_template: str,
    ) -> Path:
        if not input_image_file_path.exists() or not input_image_file_path.is_file():
            raise SingleImageTransformError(
                f"Input image file was not found: {input_image_file_path}"
            )

        uploaded_image_url = fal_client.upload_file(str(input_image_file_path))
        generation_result_payload = fal_client.subscribe(
            self._settings.fal_model_id,
            arguments={
                "prompt": prompt_template,
                "image_url": uploaded_image_url,
            },
            with_logs=True,
        )

        generated_asset_url = self._extract_generated_asset_url(generation_result_payload)
        output_extension = self._derive_output_extension_from_url(generated_asset_url)
        output_file_path = (
            self._settings.output_directory / f"generated_{uuid4().hex}{output_extension}"
        )
        self._download_generated_asset(generated_asset_url, output_file_path)
        return output_file_path

    def _extract_generated_asset_url(self, generation_result_payload: dict[str, Any]) -> str:
        for candidate_url in self._iter_candidate_urls(generation_result_payload):
            parsed_candidate_url = urlparse(candidate_url)
            if parsed_candidate_url.scheme in {"http", "https"}:
                return candidate_url

        raise SingleImageTransformError(
            f"No downloadable generated asset URL found in model response: {generation_result_payload}"
        )

    def _iter_candidate_urls(self, candidate_payload: object) -> list[str]:
        discovered_urls: list[str] = []
        if isinstance(candidate_payload, dict):
            for dictionary_key, dictionary_value in candidate_payload.items():
                if dictionary_key == "url" and isinstance(dictionary_value, str):
                    discovered_urls.append(dictionary_value)
                discovered_urls.extend(self._iter_candidate_urls(dictionary_value))
        elif isinstance(candidate_payload, list):
            for list_item in candidate_payload:
                discovered_urls.extend(self._iter_candidate_urls(list_item))
        return discovered_urls

    def _download_generated_asset(self, generated_asset_url: str, output_file_path: Path) -> None:
        with httpx.Client(timeout=120.0) as synchronous_http_client:
            generated_asset_response = synchronous_http_client.get(generated_asset_url)
            generated_asset_response.raise_for_status()
            output_file_path.write_bytes(generated_asset_response.content)

    def _derive_output_extension_from_url(self, generated_asset_url: str) -> str:
        file_suffix = Path(urlparse(generated_asset_url).path).suffix.lower()
        if file_suffix in {".png", ".jpg", ".jpeg", ".webp", ".mp4"}:
            return file_suffix
        return ".bin"
