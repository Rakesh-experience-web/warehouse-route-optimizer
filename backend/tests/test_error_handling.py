"""tests/test_error_handling.py — API error handler integration tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestMapNotFound:
    def test_missing_map_returns_404(self):
        resp = client.get("/api/v1/maps/nonexistent-map-id")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert body["error"] == "NOT_FOUND"
        assert "request_id" in body

    def test_404_body_has_message(self):
        resp = client.get("/api/v1/maps/does-not-exist")
        body = resp.json()
        assert "message" in body
        assert len(body["message"]) > 0


class TestValidationErrors:
    def test_malformed_optimize_body_returns_400(self):
        resp = client.post("/api/v1/optimize", json={"invalid": True})
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("error") == "VALIDATION_ERROR"

    def test_malformed_train_body_returns_400(self):
        resp = client.post("/api/v1/ml/train", json={"not_samples": []})
        assert resp.status_code == 400


class TestCorrelationId:
    def test_response_includes_request_id_for_errors(self):
        resp = client.get("/api/v1/maps/missing")
        body = resp.json()
        assert "request_id" in body
        assert isinstance(body["request_id"], str)
        assert len(body["request_id"]) > 0

    def test_custom_request_id_echoed_back(self):
        custom_id = "test-correlation-abc-123"
        resp = client.get(
            "/api/v1/maps/missing",
            headers={"X-Request-ID": custom_id},
        )
        body = resp.json()
        assert body.get("request_id") == custom_id


class TestServerErrors:
    """Verify that 500 errors do not leak stack traces."""

    def test_internal_error_hides_traceback(self, monkeypatch):
        from app.services import layout_store as ls_module

        def _boom(*args, **kwargs):
            raise RuntimeError("internal crash with secret DB password")

        monkeypatch.setattr(ls_module.LayoutStore, "list_maps", _boom)
        resp = client.get("/api/v1/maps")
        assert resp.status_code == 500
        body = resp.json()
        assert "secret DB password" not in resp.text
        assert "traceback" not in resp.text.lower()
        assert body.get("error") == "INTERNAL_ERROR"
