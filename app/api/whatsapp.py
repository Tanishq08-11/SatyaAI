"""WhatsApp Cloud API Webhook API endpoints supporting multiple route aliases."""

import hmac
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.config import get_settings, reload_settings
from app.db.session import get_db
from app.services.whatsapp_service import WhatsAppService

logger = logging.getLogger(__name__)

# Router without hardcoded prefix so it can be mounted with aliases
router = APIRouter(tags=["WhatsApp"])
default_service = WhatsAppService()


@router.get("/webhook/whatsapp", response_class=PlainTextResponse, summary="Meta Webhook Verification (Direct)")
@router.get("/whatsapp/webhook", response_class=PlainTextResponse, summary="Meta Webhook Verification (Alias)")
@router.get("/webhook", response_class=PlainTextResponse, summary="Meta Webhook Verification (Root)")
async def verify_webhook(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """Handle GET verification handshake from Meta Webhooks setup."""
    # Support both dot notation (Meta standard) and underscore notation (reverse proxies / tools)
    mode = hub_mode or request.query_params.get("hub.mode") or request.query_params.get("hub_mode")
    verify_token = (
        hub_verify_token
        or request.query_params.get("hub.verify_token")
        or request.query_params.get("hub_verify_token")
    )
    challenge = (
        hub_challenge
        or request.query_params.get("hub.challenge")
        or request.query_params.get("hub_challenge")
    )

    if not mode or not verify_token:
        logger.warning("WhatsApp webhook verification missing required hub query parameters.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required query parameters: hub.mode, hub.verify_token",
        )

    settings = get_settings()
    configured_token = settings.WHATSAPP_VERIFY_TOKEN

    token_candidate = (verify_token or "").strip().strip("'\"")
    expected_token = (configured_token or "").strip().strip("'\"")

    # If mismatch, reload settings dynamically from disk (.env) in case .env was updated
    if not (expected_token and hmac.compare_digest(token_candidate, expected_token)):
        fresh_settings = reload_settings()
        configured_token = fresh_settings.WHATSAPP_VERIFY_TOKEN
        expected_token = (configured_token or "").strip().strip("'\"")

    if not configured_token:
        logger.error("WHATSAPP_VERIFY_TOKEN is not configured in server environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server WHATSAPP_VERIFY_TOKEN configuration is missing.",
        )

    if mode == "subscribe" and hmac.compare_digest(token_candidate, expected_token):
        logger.info("WhatsApp webhook challenge verification succeeded.")
        return PlainTextResponse(content=challenge or "", status_code=status.HTTP_200_OK)

    logger.warning(
        "WhatsApp webhook verification failed: mode='%s', token mismatch.",
        mode,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Verification token mismatch or invalid hub.mode.",
    )


@router.post("/webhook/whatsapp", summary="Meta Incoming Webhook Events (Direct)")
@router.post("/whatsapp/webhook", summary="Meta Incoming Webhook Events (Alias)")
@router.post("/webhook", summary="Meta Incoming Webhook Events (Root)")
async def handle_webhook_event(
    request: Request,
    db: Session = Depends(get_db),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
):
    """
    Handle POST webhook events from Meta WhatsApp Cloud API:
    1. Validate HMAC-SHA256 signature if app secret is configured.
    2. Parse message payload.
    3. Filter duplicate messages.
    4. Verify claims via TruthGraph verification service and send outbound replies.
    5. Always return HTTP 200 to acknowledge receipt promptly.
    """
    body_bytes = await request.body()

    # 1. Validate signature
    if not default_service.verify_signature(body_bytes, x_hub_signature_256):
        logger.warning("Rejected webhook POST with invalid X-Hub-Signature-256 header.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid HMAC-SHA256 signature.",
        )

    # 2. Parse payload safely
    try:
        payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as exc:
        logger.warning("Failed to decode webhook JSON body: %s", exc)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "invalid_json_received"})

    # 3. Process events
    try:
        results = await default_service.process_webhook_payload(payload, db)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "received", "outcomes": results},
        )
    except Exception as exc:
        # Crucial: Log error and still return HTTP 200 so Meta does not retry indefinitely
        logger.error("Error processing WhatsApp webhook payload: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "error_handled", "error": str(exc)},
        )
