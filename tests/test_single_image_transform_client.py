from __future__ import annotations

from pathlib import Path

import pytest

from single_image_transform.config import SingleImageTransformSettings
from single_image_transform.fal_image_transform_client import FalImageTransformClient


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        return None

    def get(self, _url: str) -> _FakeResponse:
        return _FakeResponse(self._content)


def test_transform_local_image_writes_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_image_file_path = tmp_path / "source.jpg"
    input_image_file_path.write_bytes(b"source-bytes")

    settings = SingleImageTransformSettings(
        fal_api_key="test-key",
        fal_model_id="fal-ai/example-model",
        output_directory=tmp_path / "outputs",
    )
    settings.output_directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "single_image_transform.fal_image_transform_client.fal_client.upload_file",
        lambda _input_path: "https://cdn.fal.ai/uploaded.jpg",
    )
    monkeypatch.setattr(
        "single_image_transform.fal_image_transform_client.fal_client.subscribe",
        lambda _model_identifier, arguments, with_logs: {
            "video": {"url": "https://cdn.fal.ai/generated.mp4"},
            "echo": arguments,
            "logs_enabled": with_logs,
        },
    )
    monkeypatch.setattr(
        "single_image_transform.fal_image_transform_client.httpx.Client",
        lambda timeout: _FakeHttpClient(content=b"generated-binary"),
    )

    fal_image_transform_client = FalImageTransformClient(settings)
    output_file_path = fal_image_transform_client.transform_local_image(
        input_image_file_path=input_image_file_path,
        prompt_template="make variation",
    )

    assert output_file_path.exists()
    assert output_file_path.read_bytes() == b"generated-binary"
    assert output_file_path.suffix == ".mp4"
