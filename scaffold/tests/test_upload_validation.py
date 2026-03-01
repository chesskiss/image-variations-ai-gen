from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from pose_variations.application.api.fastapi_application import create_fastapi_application
from pose_variations.infrastructure.settings import ApplicationSettings


def _create_png_image_bytes() -> bytes:
    image_buffer = BytesIO()
    with Image.new("RGB", (16, 16), color=(120, 80, 60)) as generated_image:
        generated_image.save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def test_upload_rejects_disallowed_mime_type(tmp_path: Path) -> None:
    application_settings = ApplicationSettings(
        storage_directory=tmp_path / "storage",
        max_upload_size_bytes=5_000_000,
    )
    test_application = create_fastapi_application(application_settings)
    test_client = TestClient(test_application)

    response = test_client.post(
        "/jobs",
        files={"uploaded_image_file": ("face.png", _create_png_image_bytes(), "text/plain")},
        data={
            "selected_pose_preset_identifiers": ["rotate_left", "rotate_right"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file MIME type is not allowed."


def test_upload_rejects_file_larger_than_limit(tmp_path: Path) -> None:
    application_settings = ApplicationSettings(
        storage_directory=tmp_path / "storage",
        max_upload_size_bytes=20,
    )
    test_application = create_fastapi_application(application_settings)
    test_client = TestClient(test_application)

    response = test_client.post(
        "/jobs",
        files={"uploaded_image_file": ("face.png", _create_png_image_bytes(), "image/png")},
        data={
            "selected_pose_preset_identifiers": ["rotate_left", "rotate_right"],
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded file exceeds configured maximum upload size."
