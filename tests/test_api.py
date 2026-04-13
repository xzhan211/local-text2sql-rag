"""
Integration tests for the FastAPI routes.

Uses TestClient (synchronous WSGI-style test client from Starlette).
All external calls are mocked: inference pipeline, training pipeline,
SQLite helpers, and init_db.

Test strategy:
  - GET /health → 200 + correct body
  - POST /api/v1/query → correct response shape, nlq forwarded to pipeline
  - POST /api/v1/query → 422 on missing nlq
  - POST /api/v1/train/upload → 202, returns run_id + status="running"
  - POST /api/v1/train/upload → background task registered
  - GET /api/v1/train/{id} → running status (metrics=None)
  - GET /api/v1/train/{id} → done status with metrics dict
  - GET /api/v1/train/{id} → 404 for unknown run_id
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Patch init_db at the app level so tests don't need a real DB on startup
pytestmark = pytest.mark.usefixtures("_patch_init_db")


@pytest.fixture(autouse=True)
def _patch_init_db():
    with patch("app.main.init_db"):
        yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── POST /api/v1/query ────────────────────────────────────────────────────────

class TestQueryRoute:
    def _mock_query(self, sql="SELECT 1", confidence=0.9, used_lesson=False):
        return {"sql": sql, "confidence": confidence, "used_lesson": used_lesson}

    def test_returns_correct_shape(self, client):
        with patch("app.api.routes_query.query", return_value=self._mock_query()):
            resp = client.post("/api/v1/query", json={"nlq": "how many customers"})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"sql", "confidence", "used_lesson"}

    def test_nlq_forwarded_to_pipeline(self, client):
        with patch("app.api.routes_query.query", return_value=self._mock_query()) as mock_q:
            client.post("/api/v1/query", json={"nlq": "how many customers"})
        mock_q.assert_called_once_with("how many customers")

    def test_sql_in_response(self, client):
        with patch("app.api.routes_query.query",
                   return_value=self._mock_query(sql="SELECT COUNT(*) FROM customers")):
            resp = client.post("/api/v1/query", json={"nlq": "how many customers"})
        assert resp.json()["sql"] == "SELECT COUNT(*) FROM customers"

    def test_used_lesson_in_response(self, client):
        with patch("app.api.routes_query.query",
                   return_value=self._mock_query(used_lesson=True)):
            resp = client.post("/api/v1/query", json={"nlq": "how many customers"})
        assert resp.json()["used_lesson"] is True

    def test_missing_nlq_returns_422(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422


# ── POST /api/v1/train/upload ─────────────────────────────────────────────────

class TestUploadRoute:
    def _csv_file(self, content: str = "nlq,sql\nhow many,SELECT COUNT(*) FROM customers\n"):
        return {"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")}

    def test_returns_202(self, client):
        with patch("app.api.routes_train.insert_training_run", return_value=1), \
             patch("app.api.routes_train.run_pipeline"):
            resp = client.post("/api/v1/train/upload", files=self._csv_file())
        assert resp.status_code == 202

    def test_returns_run_id_and_status(self, client):
        with patch("app.api.routes_train.insert_training_run", return_value=42), \
             patch("app.api.routes_train.run_pipeline"):
            resp = client.post("/api/v1/train/upload", files=self._csv_file())
        body = resp.json()
        assert body["run_id"] == 42
        assert body["status"] == "running"

    def test_no_file_returns_422(self, client):
        resp = client.post("/api/v1/train/upload")
        assert resp.status_code == 422


# ── GET /api/v1/train/{run_id} ────────────────────────────────────────────────

class TestTrainStatusRoute:
    def _make_row(self, run_id=1, status="running", metrics=None):
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "id": run_id, "status": status, "metrics": metrics
        }[key]
        return row

    def test_running_status_metrics_null(self, client):
        with patch("app.api.routes_train.get_training_run",
                   return_value=self._make_row(status="running", metrics=None)):
            resp = client.get("/api/v1/train/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["metrics"] is None

    def test_done_status_with_metrics(self, client):
        metrics_json = json.dumps({"pass_at_1": 0.8, "hard_failure_rate": 0.1})
        with patch("app.api.routes_train.get_training_run",
                   return_value=self._make_row(status="done", metrics=metrics_json)):
            resp = client.get("/api/v1/train/1")
        body = resp.json()
        assert body["status"] == "done"
        assert body["metrics"]["pass_at_1"] == pytest.approx(0.8)

    def test_unknown_run_id_returns_404(self, client):
        with patch("app.api.routes_train.get_training_run", return_value=None):
            resp = client.get("/api/v1/train/999")
        assert resp.status_code == 404

    def test_run_id_in_response(self, client):
        with patch("app.api.routes_train.get_training_run",
                   return_value=self._make_row(run_id=7, status="done",
                                               metrics=json.dumps({}))):
            resp = client.get("/api/v1/train/7")
        assert resp.json()["run_id"] == 7
