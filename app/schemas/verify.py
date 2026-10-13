"""Schemas for the /api/verify endpoint."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.report import ReportResponse


class VerifyRequest(BaseModel):
    """Request payload for verifying text claims."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The message or text containing claims to be verified",
        examples=["Breaking: The central bank has announced an immediate currency recall."],
    )
    language: Optional[str] = Field(
        None,
        max_length=10,
        description="Optional ISO language code or hint (e.g. 'en', 'hi', 'te')",
        examples=["en"],
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Unique request correlation ID for tracking verification runs",
    )
    message_id: Optional[str] = Field(
        None,
        description="External source message ID (e.g. WhatsApp wamid)",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text cannot be empty or contain only whitespace.")
        return stripped

    @field_validator("language")
    @classmethod
    def clean_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = v.strip().lower()
            return cleaned if cleaned else None
        return None


class VerifyResponse(BaseModel):
    """Response payload returned from /api/verify."""

    status: str = Field(
        default="completed",
        description="Processing status of the verification request",
    )
    message: str = Field(
        ...,
        description="Summary status message regarding evidence and verification outcome",
    )
    report: ReportResponse = Field(
        ...,
        description="Detailed verification report containing extracted claims and verdict",
    )
