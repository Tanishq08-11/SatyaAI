"""Schemas for fact-checking reports."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.claim import ClaimResponse
from app.schemas.common import VerdictStatus


class ReportBase(BaseModel):
    """Base schema for a fact-check report."""

    input_text: str = Field(..., min_length=1, description="Original input text analyzed")
    language: Optional[str] = Field(None, max_length=10, description="Language code")
    overall_verdict: VerdictStatus = Field(
        default=VerdictStatus.CANNOT_CONFIRM,
        description="Aggregate verdict for the entire input",
    )
    summary: Optional[str] = Field(
        None, description="Executive summary of the verification findings"
    )


class ReportCreate(ReportBase):
    """Schema for creating a report record."""

    pass


class ReportResponse(ReportBase):
    """Schema for returning full report details with all extracted claims."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    claims: List[ClaimResponse] = Field(
        default_factory=list,
        description="List of factual claims evaluated in this report",
    )
