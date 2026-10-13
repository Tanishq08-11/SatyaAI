"""Schemas for claims extracted from messages."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import VerdictStatus
from app.schemas.evidence import ClaimEvidenceResponse


class ClaimBase(BaseModel):
    """Base schema for an individual claim."""

    claim_text: str = Field(..., min_length=1, description="Extracted claim text statement")
    verdict: VerdictStatus = Field(
        default=VerdictStatus.CANNOT_CONFIRM,
        description="Factual verification verdict status",
    )
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score of verdict between 0.0 and 1.0"
    )
    explanation: Optional[str] = Field(
        None, description="Cautious reasoning explaining the verdict decision"
    )


class ClaimCreate(ClaimBase):
    """Schema for creating a claim under a report."""

    report_id: str


class ClaimResponse(ClaimBase):
    """Schema for returning claim details with attached evidence."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    report_id: str
    created_at: datetime
    claim_evidence: List[ClaimEvidenceResponse] = Field(
        default_factory=list,
        description="List of evidence references supporting/refuting the claim",
    )
