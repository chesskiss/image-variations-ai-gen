from __future__ import annotations

from fastapi.testclient import TestClient

from ui import app as ui_application


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    with TestClient(ui_application.application) as test_client:
        response = test_client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "ui_jobs_created_total" in response.text
    assert "ui_cache_hits_total" in response.text
