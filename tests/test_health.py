"""Tests for GET /health endpoint."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.main import app


def test_health_check_success(client: TestClient):
    """Ensure GET /health returns 200 OK and healthy status with database connected."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "SATYA AI Test"
    assert data["database"] == "connected"
    assert "version" in data
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


def test_api_health_check_alias(client: TestClient):
    """Ensure GET /api/health is also reachable and consistent."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_health_check_database_error():
    """Ensure health check safely reports 'degraded' when the database connection fails."""
    mock_db = MagicMock(spec=Session)
    mock_db.execute.side_effect = Exception("Simulated DB connection failure")

    def mock_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = mock_get_db
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"] == "disconnected"

    app.dependency_overrides.clear()
