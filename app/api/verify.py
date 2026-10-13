"""Fact verification API endpoint."""

import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.verify import VerifyRequest, VerifyResponse
from app.services.verification import verify_text_claims

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Verification"])


@router.post(
    "/verify",
    response_model=VerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify claim text",
    description=(
        "Accepts text and an optional language code. Extracts candidate claims and returns "
        "a structured fact-checking report. In Phase 1, unverified claims return 'cannot_confirm' "
        "in compliance with the evidence-first policy (no fabricated evidence)."
    ),
)
async def verify_claim(
    request: VerifyRequest,
    db: Session = Depends(get_db),
) -> VerifyResponse:
    """Intake and evaluate text message claims against evidence policy."""
    logger.info("Received claim verification request")
    report = await verify_text_claims(db=db, request=request)

    return VerifyResponse(
        status="completed",
        message=(
            f"TruthGraph verification completed with verdict: {report.overall_verdict}."
        ),
        report=report,  # FastAPI/Pydantic will serialize this via from_attributes
    )
