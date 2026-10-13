"""Schemas for sources and claim evidence."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from app.schemas.common import EvidenceStance


class SourceBase(BaseModel):
    """Base schema for an evidence source."""

    url: str = Field(..., max_length=2048, description="Canonical URL of the source")
    domain: Optional[str] = Field(None, max_length=255, description="Domain name")
    title: Optional[str] = Field(None, max_length=512, description="Article or document title")
    publisher: Optional[str] = Field(None, max_length=255, description="Publishing organization")
    published_date: Optional[str] = Field(None, max_length=100, description="Publication date string if available")
    source_type: Optional[str] = Field(None, max_length=50, description="Type of source: fact_check, news, primary, web")
    retrieval_method: Optional[str] = Field(None, max_length=50, description="Retrieval method used")
    reliability_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Reputation/reliability score between 0.0 and 1.0"
    )


class SourceCreate(SourceBase):
    """Schema for creating a new evidence source."""

    pass


class SourceResponse(SourceBase):
    """Schema for returning source details."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class ClaimEvidenceBase(BaseModel):
    """Base schema for evidence linked to a claim."""

    snippet: Optional[str] = Field(None, description="Relevant excerpt or quotation from source")
    stance: EvidenceStance = Field(
        default=EvidenceStance.NEUTRAL,
        description="Relationship of evidence to claim: supports, refutes, neutral, inconclusive",
    )
    explanation: Optional[str] = Field(
        None, description="Short explanation of how this evidence relates to the claim"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Relevance similarity score between 0.0 and 1.0"
    )


class ClaimEvidenceCreate(ClaimEvidenceBase):
    """Schema for creating claim-evidence junction."""

    claim_id: str
    source_id: str


class ClaimEvidenceResponse(ClaimEvidenceBase):
    """Schema for returning claim-evidence junction details with embedded source."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    claim_id: str
    source_id: str
    source: Optional[SourceResponse] = None
    created_at: datetime
