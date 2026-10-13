"""Tests for request validation on POST /api/verify."""

import pytest
from fastapi.testclient import TestClient


def test_verify_valid_request_with_language(client: TestClient):
    """Test valid text and language payload returns 200 and completed report."""
    payload = {
        "text": "Drinking boiled lemon water cures viral infections completely.",
        "language": "en",
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert "report" in data
    report = data["report"]
    assert report["input_text"] == payload["text"]
    assert report["language"] == "en"
    assert "id" in report
    assert "claims" in report
    assert len(report["claims"]) >= 1
    assert "X-Request-ID" in response.headers


def test_verify_valid_request_without_language(client: TestClient):
    """Test valid payload without language defaults language to None and succeeds."""
    payload = {
        "text": "The reserve bank has announced a 500-rupee note phase out from next week.",
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["language"] is None


def test_verify_empty_text_rejected(client: TestClient):
    """Test empty string payload is rejected with 422 Unprocessable Entity."""
    response = client.post("/api/verify", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert "detail" in data


def test_verify_whitespace_text_rejected(client: TestClient):
    """Test whitespace-only payload is rejected with 422 Unprocessable Entity."""
    response = client.post("/api/verify", json={"text": "    \n\t   "})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_verify_missing_text_rejected(client: TestClient):
    """Test payload missing the 'text' key is rejected with 422."""
    response = client.post("/api/verify", json={"language": "en"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_verify_oversized_text_rejected(client: TestClient):
    """Test payload exceeding 5000 characters is rejected with 422."""
    large_text = "Claim text sentence. " * 300  # > 6000 chars
    response = client.post("/api/verify", json={"text": large_text})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
