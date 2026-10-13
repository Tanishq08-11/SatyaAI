"""Meta WhatsApp Cloud API client for sending outbound text messages."""

import logging
from typing import Any, Dict, Optional
import httpx

from app.config import Settings, get_settings, reload_settings

logger = logging.getLogger(__name__)


class WhatsAppClient:
    """Client for dispatching outbound messages via Meta WhatsApp Cloud API."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings:
            return self._settings
        return reload_settings()

    def is_configured(self) -> bool:
        """Check whether the required Meta WhatsApp Cloud API credentials are set."""
        return bool(
            self.settings.WHATSAPP_ACCESS_TOKEN
            and self.settings.WHATSAPP_PHONE_NUMBER_ID
        )

    async def send_text_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
        context_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a text message to a WhatsApp user using Meta WhatsApp Cloud API.
        
        Args:
            to: Destination WhatsApp phone number (E.164 formatted, e.g. '1234567890').
            text: Text body of the message (max 4096 characters).
            preview_url: Whether to generate a link preview for the first URL in text.
            context_message_id: Optional incoming WhatsApp message ID (wamid) to thread/quote as context.
            
        Returns:
            Dict containing API response details.
        """
        clean_recipient = to.strip().lstrip("+")
        if not clean_recipient:
            logger.warning("WhatsApp send aborted: recipient number is empty.")
            return {"status": "failed", "error": "Recipient phone number is empty."}

        # Privacy-conscious phone number masking for logs
        masked_recipient = (
            f"{clean_recipient[:4]}***{clean_recipient[-4:]}"
            if len(clean_recipient) > 7
            else "***"
        )

        # Truncate to WhatsApp character limit if needed
        trimmed_body = text[:4000] if len(text) > 4000 else text

        if not self.is_configured():
            logger.info(
                "WhatsApp Cloud API credentials not configured; simulated send to %s (body len=%d, context=%s)",
                masked_recipient,
                len(trimmed_body),
                context_message_id,
            )
            return {
                "status": "simulated",
                "recipient": clean_recipient,
                "message_length": len(trimmed_body),
                "context_message_id": context_message_id,
                "mock": True,
            }

        graph_version = self.settings.WHATSAPP_GRAPH_API_VERSION or "v21.0"
        phone_number_id = self.settings.WHATSAPP_PHONE_NUMBER_ID
        url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"

        headers = {
            "Authorization": f"Bearer {self.settings.WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_recipient,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": trimmed_body,
            },
        }
        if context_message_id:
            payload["context"] = {"message_id": context_message_id}

        logger.info(
            "Dispatching outbound WhatsApp message to %s (length %d chars, context=%s)",
            masked_recipient,
            len(trimmed_body),
            context_message_id,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    # If Meta fails due to context message (e.g. expired or invalid wamid), retry without context
                    if context_message_id and response.status_code == 400:
                        logger.warning(
                            "WhatsApp send failed with context message %s; retrying without context header.",
                            context_message_id,
                        )
                        payload.pop("context", None)
                        response = await client.post(url, headers=headers, json=payload)

                    if response.status_code == 401:
                        logger.error(
                            "Meta WhatsApp Cloud API Authentication Error (HTTP 401, OAuthException Code 190): "
                            "The WhatsApp Access Token in .env (WHATSAPP_ACCESS_TOKEN) has expired or is invalid. "
                            "To fix: Open Meta Developer Dashboard -> WhatsApp -> API Setup, copy a new Temporary Access Token (or System User token), paste it into .env, and save."
                        )
                    elif response.status_code >= 400:
                        logger.error(
                            "WhatsApp Cloud API returned error status %d: %s",
                            response.status_code,
                            response.text,
                        )
                    response.raise_for_status()

                data = response.json()
                logger.info(
                    "WhatsApp message dispatched successfully to %s. Response message ID: %s",
                    masked_recipient,
                    data.get("messages", [{}])[0].get("id", "unknown"),
                )
                return data
        except httpx.TimeoutException:
            logger.error("Timeout sending message to WhatsApp Cloud API for recipient %s", clean_recipient)
            return {"status": "error", "error": "WhatsApp API timeout"}
        except httpx.HTTPStatusError as exc:
            error_msg = "WhatsApp API error"
            try:
                err_data = exc.response.json()
                error_msg = err_data.get("error", {}).get("message", error_msg)
            except Exception:
                pass
            logger.error(
                "WhatsApp Cloud API HTTP error %d: %s",
                exc.response.status_code,
                error_msg,
            )
            return {
                "status": "error",
                "status_code": exc.response.status_code,
                "error": error_msg,
            }
        except Exception as exc:
            logger.error("Unexpected error sending WhatsApp message to %s: %s", clean_recipient, exc)
            return {"status": "error", "error": str(exc)}
