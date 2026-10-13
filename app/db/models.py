"""SQLAlchemy ORM models for reports, claims, sources, and claim_evidence."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.schemas.common import EvidenceStance, VerdictStatus


def generate_uuid() -> str:
    """Generate a unique UUID4 string identifier."""
    return str(uuid.uuid4())


class Report(Base):
    """Database model representing a verification report generated for a user message."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(36), index=True, nullable=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        String(255), index=True, nullable=True
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    overall_verdict: Mapped[str] = mapped_column(
        String(50),
        default=VerdictStatus.CANNOT_CONFIRM.value,
        nullable=False,
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    claims: Mapped[List["Claim"]] = relationship(
        "Claim",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="Claim.created_at",
    )

    def __repr__(self) -> str:
        return f"<Report(id={self.id}, verdict={self.overall_verdict})>"


class Claim(Base):
    """Database model representing an individual factual claim extracted from an input."""

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(
        String(50),
        default=VerdictStatus.CANNOT_CONFIRM.value,
        nullable=False,
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    report: Mapped["Report"] = relationship("Report", back_populates="claims")
    claim_evidence: Mapped[List["ClaimEvidence"]] = relationship(
        "ClaimEvidence",
        back_populates="claim",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Claim(id={self.id}, verdict={self.verdict})>"


class Source(Base):
    """Database model representing an authoritative evidence source / publisher."""

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    url: Mapped[str] = mapped_column(
        String(2048), unique=True, index=True, nullable=False
    )
    domain: Mapped[Optional[str]] = mapped_column(
        String(255), index=True, nullable=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    publisher: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_date: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    retrieval_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reliability_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    claim_evidence: Mapped[List["ClaimEvidence"]] = relationship(
        "ClaimEvidence",
        back_populates="source",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Source(id={self.id}, domain={self.domain})>"


class ClaimEvidence(Base):
    """Database model representing the junction relationship between a Claim and an evidence Source."""

    __tablename__ = "claim_evidence"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    claim_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stance: Mapped[str] = mapped_column(
        String(50),
        default=EvidenceStance.NEUTRAL.value,
        nullable=False,
    )
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    claim: Mapped["Claim"] = relationship(
        "Claim", back_populates="claim_evidence"
    )
    source: Mapped["Source"] = relationship(
        "Source", back_populates="claim_evidence"
    )

    def __repr__(self) -> str:
        return f"<ClaimEvidence(claim_id={self.claim_id}, source_id={self.source_id}, stance={self.stance})>"


class ProcessedWhatsAppMessage(Base):
    """Database model to track processed WhatsApp message IDs for idempotency and duplicate protection."""

    __tablename__ = "whatsapp_processed_messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    message_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    sender_id: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(36), index=True, nullable=True
    )
    message_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ProcessedWhatsAppMessage(message_id={self.message_id}, sender={self.sender_id})>"

