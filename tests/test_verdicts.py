"""Tests for verdict statuses, schemas, and evidence-first cautious behavior."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas.claim import ClaimBase
from app.schemas.common import EvidenceStance, VerdictStatus
from app.schemas.report import ReportBase


def test_supported_verdicts_enumeration():
    """Ensure all required verdicts are explicitly supported with expected string values."""
    expected_verdicts = {
        "verified",
        "false",
        "outdated",
        "partly_supported",
        "cannot_confirm",
    }
    actual_verdicts = {v.value for v in VerdictStatus}
    assert actual_verdicts == expected_verdicts


def test_all_verdicts_accepted_in_schemas():
    """Ensure schemas accept each of the 5 supported verdicts without error."""
    for verdict in VerdictStatus:
        claim = ClaimBase(
            claim_text="Test statement under assessment",
            verdict=verdict,
            confidence_score=0.85 if verdict != VerdictStatus.CANNOT_CONFIRM else 0.0,
            explanation=f"Testing verdict: {verdict.value}",
        )
        assert claim.verdict == verdict

        report = ReportBase(
            input_text="Test statement under assessment",
            overall_verdict=verdict,
            summary=f"Report with verdict: {verdict.value}",
        )
        assert report.overall_verdict == verdict


def test_invalid_verdict_rejected():
    """Ensure invalid / unsupported verdict strings are rejected by Pydantic."""
    invalid_verdicts = ["true", "unverified", "fake_news", "debunked", "unknown"]
    for bad_verdict in invalid_verdicts:
        with pytest.raises(ValidationError):
            ClaimBase(
                claim_text="Sample text",
                verdict=bad_verdict,  # type: ignore
            )


def test_evidence_stance_values():
    """Ensure EvidenceStance supports standard relational stances."""
    expected_stances = {"supports", "refutes", "neutral", "inconclusive"}
    actual_stances = {s.value for s in EvidenceStance}
    assert actual_stances == expected_stances


def test_phase1_critical_rule_cannot_confirm_returned(client: TestClient):
    """
    CRITICAL RULE TEST:
    Until evidence retrieval and analysis are implemented, the verification endpoint
    must return cannot_confirm rather than pretend a claim was verified.
    Must not fabricate source URLs, quotations, evidence or fact-check results.
    """
    response = client.post(
        "/api/verify",
        json={"text": "Government is giving free laptops to all university students who apply."},
    )
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    # 1. Must return cannot_confirm
    assert report["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value

    # 2. All claims must be marked cannot_confirm
    assert len(report["claims"]) >= 1
    for claim in report["claims"]:
        assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
        # 3. Evidence list must be empty (NO fabricated evidence or source URLs)
        assert claim["claim_evidence"] == []

    # 4. Explanation must state cautious reason without fabricating facts
    for claim in report["claims"]:
        assert "cannot_confirm" in claim["explanation"]
