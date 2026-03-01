from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path

import httpx

from pose_variations.domain.scoring_models import IdentitySimilarityScore
from pose_variations.infrastructure.settings import ApplicationSettings


class OpenAiIdentitySimilarityJudgeError(RuntimeError):
    """Raised when OpenAI similarity scoring fails."""


class OpenAiIdentitySimilarityJudge:
    _SYSTEM_PROMPT = (
        "You are a strict identity similarity evaluator. Compare the original image and candidate image "
        "for the same person and outfit continuity. Return only one float between 0.0 and 1.0 with no text."
    )

    def __init__(self, application_settings: ApplicationSettings) -> None:
        self._application_settings = application_settings
        self._openai_base_url = "https://api.openai.com/v1/responses"

    @staticmethod
    def parse_identity_similarity_output(raw_model_output_text: str) -> IdentitySimilarityScore:
        float_match = re.search(r"([01](?:\.\d+)?)", raw_model_output_text)
        if float_match is None:
            raise OpenAiIdentitySimilarityJudgeError(
                f"OpenAI similarity judge did not return a parseable float: {raw_model_output_text!r}"
            )

        parsed_score = float(float_match.group(1))
        if parsed_score < 0.0 or parsed_score > 1.0:
            raise OpenAiIdentitySimilarityJudgeError(
                f"OpenAI similarity score is outside the expected range: {parsed_score}"
            )
        return IdentitySimilarityScore(parsed_score)

    async def score_identity_similarity(
        self,
        original_image_file_path: Path,
        candidate_image_file_path: Path,
    ) -> IdentitySimilarityScore:
        if not self._application_settings.openai_api_key:
            raise OpenAiIdentitySimilarityJudgeError(
                "OPENAI_API_KEY must be configured when similarity judge is enabled."
            )

        original_image_data_url = self._build_data_url_for_image_file(original_image_file_path)
        candidate_image_data_url = self._build_data_url_for_image_file(candidate_image_file_path)

        request_payload = {
            "model": self._application_settings.openai_similarity_model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self._SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Return only a strict float score from 0.0 to 1.0.",
                        },
                        {"type": "input_image", "image_url": original_image_data_url},
                        {"type": "input_image", "image_url": candidate_image_data_url},
                    ],
                },
            ],
            "temperature": 0,
        }

        backoff_seconds = 1.0
        for attempt_number in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=60.0) as asynchronous_http_client:
                    openai_response = await asynchronous_http_client.post(
                        self._openai_base_url,
                        headers={
                            "Authorization": f"Bearer {self._application_settings.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_payload,
                    )
                    openai_response.raise_for_status()
                    openai_response_payload = openai_response.json()
                    response_text_output = self._extract_text_output(openai_response_payload)
                    return self.parse_identity_similarity_output(response_text_output)
            except (httpx.HTTPError, KeyError, ValueError) as openai_request_error:
                if attempt_number == 3:
                    raise OpenAiIdentitySimilarityJudgeError(
                        "OpenAI similarity judge request failed after retries."
                    ) from openai_request_error
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2.0

        raise OpenAiIdentitySimilarityJudgeError("Unexpected OpenAI retry exhaustion.")

    def _extract_text_output(self, openai_response_payload: dict[str, object]) -> str:
        output_entries = openai_response_payload.get("output")
        if not isinstance(output_entries, list):
            raise OpenAiIdentitySimilarityJudgeError("OpenAI response missing output list.")

        for output_entry in output_entries:
            if not isinstance(output_entry, dict):
                continue
            content_entries = output_entry.get("content")
            if not isinstance(content_entries, list):
                continue
            for content_entry in content_entries:
                if not isinstance(content_entry, dict):
                    continue
                text_value = content_entry.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()

        raise OpenAiIdentitySimilarityJudgeError("OpenAI response did not include text output.")

    def _build_data_url_for_image_file(self, image_file_path: Path) -> str:
        image_binary_content = image_file_path.read_bytes()
        encoded_image_payload = base64.b64encode(image_binary_content).decode("utf-8")
        mime_type = "image/png" if image_file_path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime_type};base64,{encoded_image_payload}"
