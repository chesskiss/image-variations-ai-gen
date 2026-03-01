from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from ui import app as scaffold_application


def test_write_and_read_job_status_payload_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        scaffold_application, "ORCHESTRATOR_JOB_STATUS_DIRECTORY", tmp_path / "jobs"
    )

    generation_job_identifier = "job-roundtrip"
    expected_job_status_payload = {
        "job_id": generation_job_identifier,
        "state": "queued",
        "progress_percent": 10,
        "current_stage": "uploaded",
    }

    scaffold_application._write_job_status_payload(
        generation_job_identifier=generation_job_identifier,
        job_status_payload=expected_job_status_payload,
    )

    actual_job_status_payload = scaffold_application._read_job_status_payload(
        generation_job_identifier
    )
    assert actual_job_status_payload == expected_job_status_payload


def test_read_job_status_payload_raises_not_found_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        scaffold_application, "ORCHESTRATOR_JOB_STATUS_DIRECTORY", tmp_path / "jobs"
    )

    with pytest.raises(HTTPException) as http_exception:
        scaffold_application._read_job_status_payload("missing-job")

    assert http_exception.value.status_code == 404
