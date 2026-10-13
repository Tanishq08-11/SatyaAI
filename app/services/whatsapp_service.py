"""WhatsApp service handling payload parsing, duplicate detection, signature validation, report formatting, and dispatch."""

import hashlib
import hmac
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import ProcessedWhatsAppMessage, Report
from app.integrations.whatsapp_client import WhatsAppClient
from app.schemas.common import EvidenceStance, VerdictStatus
from app.schemas.verify import VerifyRequest
from app.services.source_evaluator import evaluate_source_quality, format_source_tier_badge
from app.services.verification import verify_text_claims

logger = logging.getLogger(__name__)


VERDICT_EMOJIS = {
    VerdictStatus.VERIFIED.value: "✅ Verified",
    VerdictStatus.FALSE.value: "❌ Contradicted / False",
    VerdictStatus.PARTLY_SUPPORTED.value: "⚠️ Partially True / Misleading",
    VerdictStatus.OUTDATED.value: "⏳ Outdated",
    VerdictStatus.CANNOT_CONFIRM.value: "❓ Cannot Confirm",
}

GREETING_WORDS = {
    "hi", "hello", "hey", "help", "start", "menu", "namaste", "info", "about",
    "good morning", "good evening", "good afternoon", "who are you"
}


def mask_phone_number(phone: str) -> str:
    """Mask sensitive phone digits for privacy-conscious logging."""
    clean = (phone or "").strip().lstrip("+")
    if len(clean) <= 6:
        return "***"
    return f"{clean[:4]}***{clean[-4:]}"


class WhatsAppService:
    """Service for processing incoming Meta WhatsApp webhooks and generating evidence-backed replies."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[WhatsAppClient] = None,
    ):
        self._settings = settings
        self._client = client
        # Fast in-memory cache for recent message IDs (bounded to 2000 entries)
        self._processed_cache: Set[str] = set()
        # Active in-flight message IDs being processed concurrently to prevent duplicate retry executions
        self._in_flight_ids: Set[str] = set()

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def client(self) -> WhatsAppClient:
        return self._client or WhatsAppClient(settings=self.settings)

    def verify_signature(self, payload_bytes: bytes, signature_header: Optional[str]) -> bool:
        """
        Validate incoming Meta webhook signature using HMAC-SHA256 and effective_meta_app_secret.
        Header format: 'sha256=<hex_digest>'
        """
        app_secret = self.settings.effective_meta_app_secret
        if not app_secret:
            # If secret is not configured in local development, allow and log warning
            logger.debug("Meta App Secret is not configured; skipping HMAC signature verification.")
            return True

        if not signature_header or not signature_header.startswith("sha256="):
            logger.warning("Missing or malformed X-Hub-Signature-256 header.")
            return False

        expected_hash = hmac.new(
            key=app_secret.encode("utf-8"),
            msg=payload_bytes,
            digestmod=hashlib.sha256,
        ).hexdigest()

        expected_signature = f"sha256={expected_hash}"
        is_valid = hmac.compare_digest(expected_signature, signature_header)
        if not is_valid:
            logger.warning("Invalid WhatsApp webhook HMAC-SHA256 signature.")
        return is_valid

    def is_duplicate(self, message_id: str, db: Session) -> bool:
        """Check if message_id was previously received and processed or is currently in-flight."""
        if not message_id:
            return False
        if message_id in self._in_flight_ids:
            return True
        if message_id in self._processed_cache:
            return True
        existing = (
            db.query(ProcessedWhatsAppMessage)
            .filter(ProcessedWhatsAppMessage.message_id == message_id)
            .first()
        )
        if existing:
            self._processed_cache.add(message_id)
            return True
        return False

    def mark_processed(
        self,
        message_id: str,
        sender_id: str,
        message_type: str,
        db: Session,
        correlation_id: Optional[str] = None,
    ):
        """Record message_id in cache and persistent database for duplicate protection."""
        if not message_id:
            return
        # Add to memory cache
        if len(self._processed_cache) > 2000:
            self._processed_cache.clear()
        self._processed_cache.add(message_id)

        try:
            record = ProcessedWhatsAppMessage(
                message_id=message_id,
                sender_id=sender_id,
                correlation_id=correlation_id,
                message_type=message_type,
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("Failed to persist processed WhatsApp message ID %s: %s", message_id, exc)

    def is_greeting(self, text: str) -> bool:
        """Check if message is a simple conversational greeting or help command."""
        clean = re.sub(r"[^\w\s]", "", text.strip().lower())
        return clean in GREETING_WORDS or clean.startswith("hi ") or clean.startswith("hello ")

    def format_greeting_reply(self) -> str:
        """Generate friendly greeting and instructions on how to use SATYA AI."""
        return (
            "👋 *Welcome to SATYA AI — Fact-Checking Bot* 🔍\n\n"
            "I verify viral news, claims, and forwards using real-time authoritative research "
            "across Tavily, Serper, and institutional archives.\n\n"
            "*How to use:*\n"
            "Send any claim or message you want fact-checked. For example:\n"
            "• _\"The World Health Organization confirmed global eradication of smallpox in 1980.\"_\n"
            "• _\"Central bank announced a mandatory shutdown of ATM networks starting tonight.\"_\n\n"
            "_Evidence-first verification powered by TruthGraph._"
        )

    def format_verification_reply(self, report: Report) -> str:
        """
        Format a Report object into a readable, concise WhatsApp message.
        Follows mobile typography guidelines, using bold headings, bullet points, and clean links.
        """
        verdict_label = VERDICT_EMOJIS.get(
            report.overall_verdict, f"❓ {report.overall_verdict.replace('_', ' ').title()}"
        )

        lines: List[str] = [
            "*SATYA AI — Fact Check* 🔍",
            "",
            f"*Overall Verdict:* {verdict_label}",
        ]

        if report.summary:
            lines.extend(["", "*Summary:*", report.summary])

        # Individual claims
        claims = report.claims or []
        if len(claims) == 1:
            claim = claims[0]
            claim_verdict = VERDICT_EMOJIS.get(claim.verdict, claim.verdict.replace('_', ' ').title())
            lines.extend([
                "",
                "*Claim Evaluated:*",
                f"\"{claim.claim_text}\"",
                f"• *Verdict:* {claim_verdict}",
            ])
            if claim.explanation:
                lines.append(f"• *Details:* {claim.explanation}")

            # Supporting or contradicting snippet
            supporting_snippets = [
                ce.snippet for ce in (claim.claim_evidence or [])
                if ce.stance == EvidenceStance.SUPPORTS.value and ce.snippet
            ]
            contradicting_snippets = [
                ce.snippet for ce in (claim.claim_evidence or [])
                if ce.stance == EvidenceStance.REFUTES.value and ce.snippet
            ]
            if supporting_snippets:
                lines.extend(["", "*Supporting Evidence:*", f"\"{supporting_snippets[0][:200]}\""])
            elif contradicting_snippets:
                lines.extend(["", "*Contradicting Evidence:*", f"\"{contradicting_snippets[0][:200]}\""])

        elif len(claims) > 1:
            lines.extend(["", "*Claims Evaluated:*"])
            for idx, claim in enumerate(claims, start=1):
                claim_verdict = VERDICT_EMOJIS.get(claim.verdict, claim.verdict.replace('_', ' ').title())
                lines.append(f"{idx}. \"{claim.claim_text}\"")
                lines.append(f"   • *Verdict:* {claim_verdict}")
                if claim.explanation:
                    lines.append(f"   • *Details:* {claim.explanation}")

        # Sources & Evidence URLs with Authority Badges
        unique_sources: Dict[str, tuple] = {}
        for claim in claims:
            for ce in claim.claim_evidence or []:
                if ce.source and ce.source.url:
                    pub = ce.source.publisher or ce.source.domain or "Source"
                    sq = evaluate_source_quality(ce.source.url, pub)
                    badge = format_source_tier_badge(sq.tier, sq.source_type)
                    unique_sources[ce.source.url] = (pub, badge)

        lines.extend(["", "*Sources & Evidence:*"])
        if unique_sources:
            for idx, (url, (pub, badge)) in enumerate(unique_sources.items(), start=1):
                lines.append(f"{idx}. *{pub}* [{badge}]: {url}")
        else:
            if report.overall_verdict == VerdictStatus.CANNOT_CONFIRM.value:
                lines.append("No authoritative public evidence found to substantiate this statement.")
            else:
                lines.append("No specific external source links available for this inquiry.")

        # Source quality assessment note if weak counterclaims were evaluated
        has_counterclaims_dismissed = any(
            claim.explanation and (
                "dismissed in favor of authoritative consensus" in claim.explanation
                or "dismissed in accordance with satya ai's evidence hierarchy" in claim.explanation.lower()
                or "low-authority or user-generated counterclaim" in claim.explanation.lower()
            )
            for claim in claims
        )
        if has_counterclaims_dismissed:
            lines.extend([
                "",
                "ℹ️ *Source Quality Assessment:*",
                "Dissenting low-authority or user-generated counterclaims were evaluated and dismissed in accordance with SATYA AI's evidence hierarchy, which prioritizes peer-reviewed scientific institutions over unverified claims.",
            ])

        # Optional confidence assessment only when meaningfully supported
        if report.overall_verdict == VerdictStatus.VERIFIED.value:
            lines.extend(["", "*Confidence:* High (Authoritatively Corroborated)"])

        lines.extend([
            "",
            "_Evidence-first verification powered by TruthGraph._",
        ])

        return "\n".join(lines)

    def format_unsupported_reply(self, message_type: str) -> str:
        """Generate guidance reply when receiving unsupported message formats (audio, image, video)."""
        type_str = message_type.capitalize() if message_type else "Media"
        return (
            f"👋 *SATYA AI Fact-Checking Bot*\n\n"
            f"Thank you for contacting SATYA AI! Currently, our verification engine specializes in "
            f"evaluating *text messages and factual claims*.\n\n"
            f"{type_str} verification is planned for an upcoming release.\n\n"
            f"Please copy and paste the text of the claim you would like us to verify."
        )

    async def process_webhook_payload(
        self, payload: Dict[str, Any], db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Process incoming Meta webhook JSON payload:
        1. Extract sender ID, message ID, and message contents.
        2. Filter duplicates.
        3. Handle greetings, empty text, or unsupported media formats.
        4. For factual text: run verification pipeline via Tavily/Serper/TruthGraph and send reply.
        """
        session_managed = False
        if db is None:
            from app.db.session import SessionLocal
            db = SessionLocal()
            session_managed = True

        results: List[Dict[str, Any]] = []

        try:
            entries = payload.get("entry", [])
            if not entries:
                return [{"status": "no_entries"}]

            for entry in entries:
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    # Ignore status updates (sent, delivered, read)
                    if not messages:
                        if value.get("statuses"):
                            logger.debug("Received WhatsApp delivery status update; acknowledging.")
                            results.append({"status": "acknowledged_status_update"})
                        continue

                    for msg in messages:
                        message_id = msg.get("id", "")
                        sender_id = msg.get("from", "")
                        msg_type = msg.get("type", "unknown")

                        if not sender_id or not message_id:
                            logger.warning("Malformed WhatsApp message item missing sender or ID.")
                            continue

                        # Duplicate check (checks in-flight, memory cache, and persistent DB)
                        if self.is_duplicate(message_id, db):
                            logger.info("Skipping duplicate or in-flight WhatsApp message ID: %s", message_id)
                            results.append({
                                "status": "duplicate_skipped",
                                "message_id": message_id,
                                "sender": sender_id,
                            })
                            continue

                        # Generate unique correlation ID for this verification run
                        correlation_id = str(uuid.uuid4())
                        masked_sender = mask_phone_number(sender_id)

                        # Lock message in-flight to prevent concurrent Meta retries from duplicating execution
                        self._in_flight_ids.add(message_id)
                        try:
                            logger.info(
                                "WhatsApp Inbound Run: correlation_id=%s, msg_id=%s, sender=%s, type=%s",
                                correlation_id,
                                message_id,
                                masked_sender,
                                msg_type,
                            )

                            # Handle message types
                            if msg_type == "text":
                                text_body = msg.get("text", {}).get("body", "").strip()
                                if not text_body:
                                    logger.info(
                                        "Empty text body in WhatsApp message %s (correlation_id=%s); sending prompt.",
                                        message_id,
                                        correlation_id,
                                    )
                                    guidance = "Please send a text message containing the factual claim you would like verified."
                                    send_outcome = await self.client.send_text_message(
                                        to=sender_id,
                                        text=guidance,
                                        context_message_id=message_id,
                                    )
                                    self.mark_processed(
                                        message_id=message_id,
                                        sender_id=sender_id,
                                        message_type=msg_type,
                                        db=db,
                                        correlation_id=correlation_id,
                                    )
                                    results.append({
                                        "status": "empty_text_handled",
                                        "message_id": message_id,
                                        "correlation_id": correlation_id,
                                    })
                                    continue

                                # Check for greetings
                                if self.is_greeting(text_body):
                                    logger.info(
                                        "Detected greeting from %s: correlation_id=%s, msg_id=%s",
                                        masked_sender,
                                        correlation_id,
                                        message_id,
                                    )
                                    greeting_reply = self.format_greeting_reply()
                                    send_outcome = await self.client.send_text_message(
                                        to=sender_id,
                                        text=greeting_reply,
                                        context_message_id=message_id,
                                    )
                                    if send_outcome.get("status") != "error":
                                        self.mark_processed(
                                            message_id=message_id,
                                            sender_id=sender_id,
                                            message_type=msg_type,
                                            db=db,
                                            correlation_id=correlation_id,
                                        )
                                    results.append({
                                        "status": "greeting_handled",
                                        "message_id": message_id,
                                        "correlation_id": correlation_id,
                                        "sender": sender_id,
                                        "send_outcome": send_outcome,
                                    })
                                    continue

                                # Check for overly long message
                                if len(text_body) > 3000:
                                    logger.info(
                                        "Overly long message from %s (len=%d, correlation_id=%s)",
                                        masked_sender,
                                        len(text_body),
                                        correlation_id,
                                    )
                                    long_msg_reply = (
                                        "⚠️ *Message Too Long*\n\n"
                                        "Please send a concise factual statement or viral message under 2000 characters "
                                        "so we can research it accurately."
                                    )
                                    send_outcome = await self.client.send_text_message(
                                        to=sender_id,
                                        text=long_msg_reply,
                                        context_message_id=message_id,
                                    )
                                    if send_outcome.get("status") != "error":
                                        self.mark_processed(
                                            message_id=message_id,
                                            sender_id=sender_id,
                                            message_type=msg_type,
                                            db=db,
                                            correlation_id=correlation_id,
                                        )
                                    results.append({
                                        "status": "overly_long_handled",
                                        "message_id": message_id,
                                        "correlation_id": correlation_id,
                                    })
                                    continue

                                logger.info(
                                    "Processing incoming claim: correlation_id=%s, msg_id=%s, sender=%s, len=%d, preview='%s'",
                                    correlation_id,
                                    message_id,
                                    masked_sender,
                                    len(text_body),
                                    text_body[:80].replace("\n", " "),
                                )

                                # Reuse internal verification service directly without HTTP loopback
                                verify_req = VerifyRequest(
                                    text=text_body,
                                    correlation_id=correlation_id,
                                    message_id=message_id,
                                )
                                report = await verify_text_claims(db=db, request=verify_req)
                                reply_text = self.format_verification_reply(report)

                                # Send outbound reply via Meta WhatsApp Cloud API with context threading
                                send_outcome = await self.client.send_text_message(
                                    to=sender_id,
                                    text=reply_text,
                                    context_message_id=message_id,
                                )

                                if send_outcome.get("status") != "error":
                                    self.mark_processed(
                                        message_id=message_id,
                                        sender_id=sender_id,
                                        message_type=msg_type,
                                        db=db,
                                        correlation_id=correlation_id,
                                    )
                                    logger.info(
                                        "WhatsApp Run Succeeded: correlation_id=%s, msg_id=%s, sender=%s, report_id=%s, verdict=%s, send_status=success",
                                        correlation_id,
                                        message_id,
                                        masked_sender,
                                        report.id,
                                        report.overall_verdict,
                                    )
                                else:
                                    logger.error(
                                        "WhatsApp reply dispatch failed: correlation_id=%s, msg_id=%s, sender=%s, report_id=%s, error=%s",
                                        correlation_id,
                                        message_id,
                                        masked_sender,
                                        report.id,
                                        send_outcome.get("error"),
                                    )

                                results.append({
                                    "status": "processed",
                                    "correlation_id": correlation_id,
                                    "message_id": message_id,
                                    "report_id": report.id,
                                    "sender": sender_id,
                                    "verdict": report.overall_verdict,
                                    "send_outcome": send_outcome,
                                })
                            else:
                                # Unsupported format: audio, image, video, document, sticker, etc.
                                logger.info(
                                    "Received unsupported WhatsApp message type '%s' from %s (correlation_id=%s)",
                                    msg_type,
                                    masked_sender,
                                    correlation_id,
                                )
                                guidance_reply = self.format_unsupported_reply(msg_type)
                                send_outcome = await self.client.send_text_message(
                                    to=sender_id,
                                    text=guidance_reply,
                                    context_message_id=message_id,
                                )
                                if send_outcome.get("status") != "error":
                                    self.mark_processed(
                                        message_id=message_id,
                                        sender_id=sender_id,
                                        message_type=msg_type,
                                        db=db,
                                        correlation_id=correlation_id,
                                    )
                                results.append({
                                    "status": "unsupported_type_handled",
                                    "correlation_id": correlation_id,
                                    "message_id": message_id,
                                    "message_type": msg_type,
                                    "sender": sender_id,
                                    "send_outcome": send_outcome,
                                })
                        finally:
                            # Always release in-flight lock for this message_id
                            self._in_flight_ids.discard(message_id)
            return results
        finally:
            if session_managed and db is not None:
                db.close()
