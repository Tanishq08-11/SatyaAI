"""Tests for database persistence and relationship models."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Claim, ClaimEvidence, Report, Source
from app.schemas.common import EvidenceStance, VerdictStatus


def test_database_persistence_hierarchy(db_session: Session):
    """Test full relational persistence: Report -> Claims -> ClaimEvidence -> Source."""
    # 1. Create Report
    report = Report(
        input_text="A viral message claiming a new national tax is effective immediately.",
        language="en",
        overall_verdict=VerdictStatus.FALSE.value,
        summary="Official Ministry of Finance releases refute this viral message.",
    )
    db_session.add(report)
    db_session.flush()

    assert report.id is not None
    assert report.created_at is not None

    # 2. Create Claim
    claim = Claim(
        report_id=report.id,
        claim_text="New national tax is effective immediately.",
        verdict=VerdictStatus.FALSE.value,
        confidence_score=0.98,
        explanation="Debunked by press release from official revenue department.",
    )
    db_session.add(claim)
    db_session.flush()

    assert claim.id is not None

    # 3. Create Source
    source = Source(
        url="https://pib.gov.in/factcheck/tax-notice-clarification",
        domain="pib.gov.in",
        title="Official Clarification: No New Emergency Tax Introduced",
        publisher="Press Information Bureau",
        reliability_score=0.99,
    )
    db_session.add(source)
    db_session.flush()

    assert source.id is not None

    # 4. Create ClaimEvidence junction
    evidence = ClaimEvidence(
        claim_id=claim.id,
        source_id=source.id,
        snippet="The Ministry of Finance has clarified that no such tax decree has been issued.",
        stance=EvidenceStance.REFUTES.value,
        relevance_score=0.95,
    )
    db_session.add(evidence)
    db_session.commit()

    # 5. Query back and verify relationship traversal
    queried_report = db_session.query(Report).filter(Report.id == report.id).first()
    assert queried_report is not None
    assert len(queried_report.claims) == 1

    queried_claim = queried_report.claims[0]
    assert queried_claim.claim_text == "New national tax is effective immediately."
    assert len(queried_claim.claim_evidence) == 1

    queried_evidence = queried_claim.claim_evidence[0]
    assert queried_evidence.stance == EvidenceStance.REFUTES.value
    assert queried_evidence.source.domain == "pib.gov.in"
    assert queried_evidence.source.publisher == "Press Information Bureau"


def test_sqlite_foreign_key_constraint_enforcement(db_session: Session):
    """Test that SQLite PRAGMA foreign_keys=ON raises IntegrityError when foreign key is invalid."""
    invalid_claim = Claim(
        report_id="non-existent-report-id-12345",
        claim_text="Orphan claim without valid parent report.",
        verdict=VerdictStatus.CANNOT_CONFIRM.value,
    )
    db_session.add(invalid_claim)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_cascade_delete_report_removes_claims_and_evidence(db_session: Session):
    """Test that deleting a Report cascades to its Claims and ClaimEvidence, leaving Source intact."""
    report = Report(input_text="Sample claim for cascade test.")
    db_session.add(report)
    db_session.flush()

    claim = Claim(report_id=report.id, claim_text="Cascade test claim.")
    db_session.add(claim)
    db_session.flush()

    source = Source(url="https://example.com/cascade-source")
    db_session.add(source)
    db_session.flush()

    evidence = ClaimEvidence(claim_id=claim.id, source_id=source.id)
    db_session.add(evidence)
    db_session.commit()

    report_id = report.id
    claim_id = claim.id
    evidence_id = evidence.id
    source_id = source.id

    # Delete Report
    db_session.delete(report)
    db_session.commit()

    # Verify Report, Claim, and Evidence are deleted
    assert db_session.query(Report).filter(Report.id == report_id).first() is None
    assert db_session.query(Claim).filter(Claim.id == claim_id).first() is None
    assert db_session.query(ClaimEvidence).filter(ClaimEvidence.id == evidence_id).first() is None

    # Source should remain untouched
    assert db_session.query(Source).filter(Source.id == source_id).first() is not None
