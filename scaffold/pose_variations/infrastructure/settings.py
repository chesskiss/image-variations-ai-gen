from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    fal_api_key: str = ""
    openai_api_key: str = ""
    enable_openai_similarity_judge: bool = False

    max_upload_size_bytes: int = 5_242_880
    allowed_upload_mime_types: str = "image/jpeg,image/png"
    storage_directory: Path = Path("./storage")

    fal_model_id: str = "fal-ai/luma-dream-machine/ray-2-flash/image-to-video"
    fal_max_poll_attempts: int = 90
    fal_poll_interval_seconds: float = 2.0

    openai_similarity_model: str = "gpt-4.1-mini"
    openai_identity_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    frame_extraction_interval_seconds: float = 0.5
    minimum_sharpness_score: float = 80.0
    minimum_diversity_hash_distance: int = 8

    @property
    def allowed_upload_mime_type_set(self) -> set[str]:
        return {
            mime_type.strip().lower()
            for mime_type in self.allowed_upload_mime_types.split(",")
            if mime_type.strip()
        }


@lru_cache(maxsize=1)
def get_application_settings() -> ApplicationSettings:
    return ApplicationSettings()
