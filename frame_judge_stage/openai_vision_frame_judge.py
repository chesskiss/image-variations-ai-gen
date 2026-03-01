from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from frame_judge_stage.domain_models import FrameCandidateForJudging, FrameJudgeScore


class OpenAiVisionFrameJudgeError(RuntimeError):
    """Raised when OpenAI vision-based frame judging fails."""


class OpenAiVisionFrameJudge:
    def __init__(
        self,
        openai_api_key: str,
        judge_model_name: str = "o3",
        timeout_seconds: int = 45,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._judge_model_name = judge_model_name
        self._timeout_seconds = timeout_seconds

    def judge_frame_candidates(
        self,
        frame_candidates_for_judging: list[FrameCandidateForJudging],
        original_image_file_path: Path | None = None,
    ) -> list[FrameJudgeScore]:
        if not frame_candidates_for_judging:
            return []
        if not self._openai_api_key:
            raise OpenAiVisionFrameJudgeError("OPENAI_API_KEY is required when OpenAI frame judge is enabled.")
        if original_image_file_path is None:
            raise OpenAiVisionFrameJudgeError(
                "original_image_file_path is required for OpenAI similarity-based frame judging."
            )
        if not original_image_file_path.exists() or not original_image_file_path.is_file():
            raise OpenAiVisionFrameJudgeError(
                f"original_image_file_path does not exist: {original_image_file_path}"
            )

        frame_judge_scores: list[FrameJudgeScore] = []
        for frame_candidate_for_judging in frame_candidates_for_judging:
            raw_model_output_text = self._request_single_frame_score(
                original_image_file_path=original_image_file_path,
                frame_candidate_for_judging=frame_candidate_for_judging,
            )
            parsed_judge_score = self._parse_score_value(raw_model_output_text)
            frame_judge_scores.append(
                FrameJudgeScore(
                    frame_sequence_number=frame_candidate_for_judging.frame_sequence_number,
                    judge_score=parsed_judge_score,
                    judge_confidence=None,
                    judge_reasoning_summary=None,
                    judge_model_name=self._judge_model_name,
                    score_components={"openai_score": parsed_judge_score},
                )
            )

        return frame_judge_scores

    def _request_single_frame_score(
        self,
        original_image_file_path: Path,
        frame_candidate_for_judging: FrameCandidateForJudging,
    ) -> str:
        original_image_data_url = self._build_image_data_url(original_image_file_path)
        variation_image_data_url = self._build_image_data_url(
            frame_candidate_for_judging.local_frame_image_file_path
        )

        request_payload = {
            "model": self._judge_model_name,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a judge that decides how much a pose variation of an image "
                                "retains the original identity of the subject and his outfit. You need "
                                "to give a score between 0 and 1, where 0 is no identity retained at "
                                "all and 1 is perfect identity preservation. You must be very strict as "
                                "humans have very keen eye for facial details. The first image is the "
                                "original image and the second is the variation. Return only a JSON "
                                "object with key score and float value in [0,1]."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Score strict identity and outfit preservation. First image is original, "
                                "second image is variation."
                            ),
                        },
                        {"type": "input_image", "image_url": original_image_data_url},
                        {"type": "input_image", "image_url": variation_image_data_url},
                    ],
                },
            ],
            "temperature": 0,
        }

        with httpx.Client(timeout=self._timeout_seconds) as synchronous_http_client:
            openai_response = synchronous_http_client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self._openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
            openai_response.raise_for_status()
            openai_response_payload = openai_response.json()

        output_text = self._extract_output_text(openai_response_payload)
        return output_text

    @staticmethod
    def _extract_output_text(openai_response_payload: dict[str, object]) -> str:
        output_entries = openai_response_payload.get("output")
        if not isinstance(output_entries, list):
            raise OpenAiVisionFrameJudgeError("OpenAI response did not contain output list.")

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

        raise OpenAiVisionFrameJudgeError("OpenAI response did not include text output.")

    @staticmethod
    def _parse_score_value(raw_model_output_text: str) -> float:
        try:
            parsed_json = json.loads(raw_model_output_text)
            if isinstance(parsed_json, dict) and "score" in parsed_json:
                parsed_score = float(parsed_json["score"])
                if 0.0 <= parsed_score <= 1.0:
                    return parsed_score
        except (ValueError, TypeError):
            pass

        score_match = re.search(r"([01](?:\.\d+)?)", raw_model_output_text)
        if score_match is None:
            raise OpenAiVisionFrameJudgeError(
                f"Unable to parse a normalized score from model output: {raw_model_output_text}"
            )

        parsed_score = float(score_match.group(1))
        if parsed_score < 0.0 or parsed_score > 1.0:
            raise OpenAiVisionFrameJudgeError(
                f"Parsed score is outside [0,1]: {parsed_score}"
            )
        return parsed_score

    @staticmethod
    def _build_image_data_url(local_frame_image_file_path: Path) -> str:
        image_binary_content = local_frame_image_file_path.read_bytes()
        encoded_image_content = base64.b64encode(image_binary_content).decode("utf-8")
        suffix = local_frame_image_file_path.suffix.lower()
        image_mime_type = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{image_mime_type};base64,{encoded_image_content}"
