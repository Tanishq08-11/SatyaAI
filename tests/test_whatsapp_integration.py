"""Automated test suite for WhatsApp Cloud API integration.

Tests:
1. GET webhook verification (valid token, invalid token, missing params).
2. Webhook HMAC-SHA256 signature validation (valid, invalid, missing).
3. Incoming message parsing and dispatch.
4. Successful outbound replies via mocked Meta Cloud API client.
5. Unsupported message types (audio, image, etc.) receiving guidance replies.
6. Duplicate event protection (same message_id processed only once).
7. Meta Cloud API failure and timeout error handling without webhook crashing.
8. Evidence-backed verification replies formatted for mobile.
9. cannot_confirm replies with explicit insufficiency notes and no fabricated links.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ProcessedWhatsAppMessage
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.web_search import WebSearchClient
from app.integrations.whatsapp_client import WhatsAppClient
from app.schemas.common import VerdictStatus


@pytest.fixture(autouse=True)
def clean_test_state():
    """Clean mock registries and reset WhatsApp settings for deterministic testing."""
    WebSearchClient.clear_mock_registry()
    settings = get_settings()
    orig_token = settings.WHATSAPP_VERIFY_TOKEN
    orig_secret = settings.WHATSAPP_APP_SECRET
    orig_meta_secret = settings.META_APP_SECRET
    orig_phone = settings.WHATSAPP_PHONE_NUMBER_ID
    orig_access = settings.WHATSAPP_ACCESS_TOKEN

    settings.WHATSAPP_VERIFY_TOKEN = "satya_test_verify_token"
    settings.WHATSAPP_APP_SECRET = "satya_test_secret_12345"
    settings.META_APP_SECRET = None
    settings.WHATSAPP_PHONE_NUMBER_ID = "10987654321"
    settings.WHATSAPP_ACCESS_TOKEN = "EAA_TEST_TOKEN_123"

    yield

    current_settings = get_settings()
    current_settings.WHATSAPP_VERIFY_TOKEN = orig_token
    current_settings.WHATSAPP_APP_SECRET = orig_secret
    current_settings.META_APP_SECRET = orig_meta_secret
    current_settings.WHATSAPP_PHONE_NUMBER_ID = orig_phone
    current_settings.WHATSAPP_ACCESS_TOKEN = orig_access
    WebSearchClient.clear_mock_registry()


def make_whatsapp_payload(message_id: str, sender: str, text: str, msg_type: str = "text") -> dict:
    """Helper to construct a valid Meta WhatsApp webhook event payload."""
    msg_obj = {
        "from": sender,
        "id": message_id,
        "timestamp": "1710000000",
        "type": msg_type,
    }
    if msg_type == "text":
        msg_obj["text"] = {"body": text}
    elif msg_type == "audio":
        msg_obj["audio"] = {"id": "media_audio_123", "mime_type": "audio/ogg"}
    elif msg_type == "image":
        msg_obj["image"] = {"id": "media_img_123", "mime_type": "image/jpeg"}

    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550001234",
                                "phone_number_id": "10987654321",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test User"},
                                    "wa_id": sender,
                                }
                            ],
                            "messages": [msg_obj],
                        },
                    }
                ],
            }
        ],
    }


def compute_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for payload."""
    h = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={h}"


def test_get_webhook_verification_success(client: TestClient):
    """Test Meta webhook verification handshake with correct token."""
    response = client.get(
        "/api/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "satya_test_verify_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_get_webhook_verification_token_mismatch(client: TestClient):
    """Test Meta webhook verification with invalid token returns 403."""
    response = client.get(
        "/api/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_invalid_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403
    assert "mismatch" in response.json()["detail"].lower()


def test_get_webhook_verification_missing_params(client: TestClient):
    """Test Meta webhook verification with missing query parameters returns 400."""
    response = client.get("/api/whatsapp/webhook")
    assert response.status_code == 400


def test_meta_exact_verification_request_returns_200_and_challenge(client: TestClient):
    """
    Test reproducing the exact Meta verification request:
    - hub.mode=subscribe
    - hub.verify_token matching configured token
    - hub.challenge=221670983
    Confirms HTTP 200 with the challenge string returned unchanged.
    """
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "satya_test_verify_token",
            "hub.challenge": "221670983",
        },
    )
    assert response.status_code == 200
    assert response.text == "221670983"
    assert response.headers["content-type"].startswith("text/plain")


def test_meta_verification_supports_underscore_aliases(client: TestClient):
    """Test verification succeeds when proxies/tools forward underscore query params."""
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub_mode": "subscribe",
            "hub_verify_token": "satya_test_verify_token",
            "hub_challenge": "221670983",
        },
    )
    assert response.status_code == 200
    assert response.text == "221670983"


def test_meta_verification_handles_quotes_and_whitespace(client: TestClient):
    """Test verification succeeds even if quotes or whitespace surround the token."""
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": '  "satya_test_verify_token"  ',
            "hub.challenge": "999888777",
        },
    )
    assert response.status_code == 200
    assert response.text == "999888777"


def test_meta_verification_reloads_when_token_changes_dynamically(client: TestClient):
    """Test verification reloads settings dynamically when token in environment changes."""
    settings = get_settings()
    settings.WHATSAPP_VERIFY_TOKEN = "initial_token_abc"

    from app.config import Settings
    reloaded = Settings(
        WHATSAPP_VERIFY_TOKEN="updated_token_xyz",
        WHATSAPP_APP_SECRET="satya_test_secret_12345",
        META_APP_SECRET=None,
        WHATSAPP_PHONE_NUMBER_ID="10987654321",
        WHATSAPP_ACCESS_TOKEN="EAA_TEST_TOKEN_123",
    )

    # Mismatch with initial token triggers dynamic reload
    with patch("app.api.whatsapp.reload_settings", return_value=reloaded) as mock_reload:
        response = client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "updated_token_xyz",
                "hub.challenge": "54321",
            },
        )
        assert response.status_code == 200
        assert response.text == "54321"
        mock_reload.assert_called_once()



def test_post_webhook_invalid_signature_rejected(client: TestClient):
    """Test that POST webhook requests with invalid or forged HMAC signature are rejected with 403."""
    payload = make_whatsapp_payload("wamid.123", "919876543210", "Hello")
    body_bytes = json.dumps(payload).encode("utf-8")

    response = client.post(
        "/api/whatsapp/webhook",
        content=body_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=invalid_forged_hash",
        },
    )
    assert response.status_code == 403
    assert "Invalid HMAC" in response.json()["detail"]


def test_incoming_text_message_verification_and_outbound_reply(client: TestClient, db_session: Session):
    """
    Test end-to-end receipt of a WhatsApp text message:
    1. Parse incoming claim.
    2. Execute TruthGraph verification pipeline.
    3. Generate mobile-formatted reply.
    4. Dispatch reply via Meta WhatsApp Cloud API client.
    """
    claim_text = "The World Health Organization confirmed global eradication of smallpox in 1980."
    sender = "919876543210"
    msg_id = "wamid.TEST_SMALLPOX_MSG_001"

    # Register mock search evidence
    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Smallpox - World Health Organization",
                canonical_url="https://www.who.int/health-topics/smallpox",
                publisher="who.int",
                published_date="1980",
                relevant_text_excerpt="In 1980 WHO declared smallpox eradicated worldwide.",
                source_type="primary_source",
                retrieval_method="web_search_who",
                reliability_score=0.98,
            )
        ],
    )

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "sent", "messages": [{"id": "wamid.OUTBOUND_123"}]},
    ) as mock_send:
        response = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "received"
    outcomes = res_data["outcomes"]
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "processed"
    assert outcomes[0]["verdict"] == VerdictStatus.VERIFIED.value

    # Verify outbound WhatsApp message dispatched with correct recipient and formatted content
    mock_send.assert_awaited_once()
    called_to = mock_send.await_args.kwargs.get("to") or mock_send.await_args.args[0]
    called_text = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]

    assert called_to == sender
    assert "*SATYA AI — Fact Check* 🔍" in called_text
    assert "*Overall Verdict:* ✅ Verified" in called_text
    assert "https://www.who.int/health-topics/smallpox" in called_text
    assert "Evidence-first verification powered by TruthGraph" in called_text


def test_cannot_confirm_reply_has_no_fabricated_evidence(client: TestClient):
    """
    Test that an unverifiable extraordinary claim returns an explicit cannot_confirm
    WhatsApp reply without inventing sources or percentages.
    """
    claim_text = "Secret alien crystalline pyramid discovered beneath Antarctica ice sheet."
    sender = "919876543210"
    msg_id = "wamid.TEST_UNVERIFIED_002"

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "sent"},
    ) as mock_send:
        response = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )

    assert response.status_code == 200
    mock_send.assert_awaited_once()
    called_text = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]

    assert "*Overall Verdict:* ❓ Cannot Confirm" in called_text
    assert "No authoritative public evidence found to substantiate this statement" in called_text
    assert "http" not in called_text.split("*Sources & Evidence:*")[1]


def test_unsupported_message_types_reply_guidance(client: TestClient):
    """Test that incoming media (audio/image) messages receive clear guidance replies."""
    sender = "919876543210"
    msg_id = "wamid.TEST_AUDIO_003"
    payload = make_whatsapp_payload(msg_id, sender, text="", msg_type="audio")
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "sent"},
    ) as mock_send:
        response = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )

    assert response.status_code == 200
    mock_send.assert_awaited_once()
    called_text = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]

    assert "text messages and factual claims" in called_text
    assert "Audio verification is planned" in called_text


def test_duplicate_message_protection(client: TestClient, db_session: Session):
    """
    Test duplicate-event protection:
    If Meta retries a webhook delivery with the same message_id, the system
    must recognize the duplicate and skip processing without sending duplicate replies.
    """
    sender = "919876543210"
    msg_id = "wamid.TEST_DUPLICATE_ID_999"
    claim_text = "The World Health Organization confirmed global eradication of smallpox in 1980."

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "sent"},
    ) as mock_send:
        # First delivery
        res1 = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )
        assert res1.status_code == 200
        assert mock_send.await_count == 1

        # Second delivery with identical message_id (retry)
        res2 = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )
        assert res2.status_code == 200
        res2_data = res2.json()
        assert res2_data["outcomes"][0]["status"] == "duplicate_skipped"
        # Crucial: outbound send was NOT triggered a second time
        assert mock_send.await_count == 1


def test_meta_api_failures_and_timeouts_handled_gracefully(client: TestClient):
    """
    Test that failures in Meta WhatsApp Cloud API calls (e.g. timeout or network blip)
    are handled gracefully without causing a 500 error on the webhook endpoint.
    """
    sender = "919876543210"
    msg_id = "wamid.TEST_TIMEOUT_005"
    claim_text = "Some claim that triggers outbound timeout."

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    # Simulate WhatsAppClient experiencing a timeout
    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "error", "error": "WhatsApp API timeout"},
    ):
        response = client.post(
            "/api/whatsapp/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )

    # Webhook endpoint returns 200 to acknowledge Meta delivery
    assert response.status_code == 200
    assert response.json()["status"] == "received"


def test_whatsapp_client_send_text_real_http_mocked():
    """Test WhatsAppClient.send_text_message HTTP request structure using mocked httpx transport."""
    settings = get_settings()
    settings.WHATSAPP_ACCESS_TOKEN = "TEST_TOKEN_SECRET"
    settings.WHATSAPP_PHONE_NUMBER_ID = "9988776655"
    settings.WHATSAPP_GRAPH_API_VERSION = "v21.0"

    client = WhatsAppClient(settings=settings)

    mock_resp_json = {
        "messaging_product": "whatsapp",
        "contacts": [{"input": "1234567890", "wa_id": "1234567890"}],
        "messages": [{"id": "wamid.MOCK_OUTBOUND_ID"}],
    }

    async def run_send():
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json=mock_resp_json)
            res = await client.send_text_message(to="1234567890", text="Hello from SATYA AI")
            assert res["messages"][0]["id"] == "wamid.MOCK_OUTBOUND_ID"
            mock_post.assert_called_once()
            call_url = mock_post.call_args.args[0]
            assert "9988776655/messages" in call_url
            call_headers = mock_post.call_args.kwargs["headers"]
            assert call_headers["Authorization"] == "Bearer TEST_TOKEN_SECRET"
            call_json = mock_post.call_args.kwargs["json"]
            assert call_json["to"] == "1234567890"
            assert call_json["text"]["body"] == "Hello from SATYA AI"

    import asyncio
    asyncio.run(run_send())


def test_direct_root_webhook_whatsapp_get_and_post(client: TestClient):
    """Test direct root /webhook/whatsapp endpoints match Meta configuration."""
    # 1. GET verification
    get_res = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "satya_test_verify_token",
            "hub.challenge": "554433",
        },
    )
    assert get_res.status_code == 200
    assert get_res.text == "554433"

    # 2. POST event
    payload = make_whatsapp_payload("wamid.ROOT_TEST_001", "919876543210", "Hi")
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(WhatsAppClient, "send_text_message", new_callable=AsyncMock, return_value={"status": "sent"}):
        post_res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "received"


def test_meta_app_secret_alias_signature_validation(client: TestClient):
    """Test signature validation when configured via META_APP_SECRET setting."""
    settings = get_settings()
    settings.META_APP_SECRET = "custom_meta_secret_999"
    settings.WHATSAPP_APP_SECRET = None

    payload = make_whatsapp_payload("wamid.META_SEC_001", "919876543210", "Test claim")
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "custom_meta_secret_999")

    with patch.object(WhatsAppClient, "send_text_message", new_callable=AsyncMock, return_value={"status": "sent"}):
        res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )
    assert res.status_code == 200


def test_incoming_greeting_message_returns_helpful_instructions(client: TestClient):
    """Test conversational greetings (hi, hello, help) receive warm onboarding instructions."""
    sender = "919876543210"
    payload = make_whatsapp_payload("wamid.GREET_001", sender, "Hello! How does this work?")
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(WhatsAppClient, "send_text_message", new_callable=AsyncMock, return_value={"status": "sent"}) as mock_send:
        res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )
    assert res.status_code == 200
    mock_send.assert_awaited_once()
    called_text = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]
    assert "Welcome to SATYA AI" in called_text
    assert "How to use:" in called_text


def test_overly_long_message_guidance(client: TestClient):
    """Test overly long messages (>3000 chars) receive guidance asking for concise claims."""
    sender = "919876543210"
    long_text = "Factual claim repeated: " + ("lorem ipsum " * 400)  # > 4000 characters
    payload = make_whatsapp_payload("wamid.LONG_001", sender, long_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(WhatsAppClient, "send_text_message", new_callable=AsyncMock, return_value={"status": "sent"}) as mock_send:
        res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )
    assert res.status_code == 200
    mock_send.assert_awaited_once()
    called_text = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]
    assert "Message Too Long" in called_text


def test_incoming_claim_full_flow_with_mocked_meta_http(client: TestClient, db_session: Session):
    """
    Test full flow from incoming WhatsApp text message through verification to outbound Meta HTTP post:
    1. Parse incoming claim payload.
    2. Execute verification engine.
    3. Mock outbound HTTP call to Meta Graph API.
    4. Assert correct HTTP request parameters (recipient, message text, headers).
    5. Assert message marked processed in DB.
    """
    sender = "919075000492"
    msg_id = "wamid.TEST_FULL_FLOW_001"
    claim_text = "The Earth is appropriately 4.5 billion years old."

    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="Age of Earth - USGS",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="pubs.usgs.gov",
                relevant_text_excerpt="Scientists have determined the age of the Earth to be 4.54 billion years.",
                source_type="primary_source",
                retrieval_method="web_search",
                reliability_score=0.98,
            )
        ],
    )

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    mock_meta_response = httpx.Response(
        200,
        json={
            "messaging_product": "whatsapp",
            "contacts": [{"input": sender, "wa_id": sender}],
            "messages": [{"id": "wamid.MOCK_OUTBOUND_CONFIRM_99"}],
        },
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_meta_response) as mock_post:
        res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )

    assert res.status_code == 200
    res_json = res.json()
    assert res_json["status"] == "received"
    outcome = res_json["outcomes"][0]
    assert outcome["status"] == "processed"
    assert outcome["message_id"] == msg_id

    # Verify outbound HTTP call to Meta
    mock_post.assert_awaited_once()
    call_kwargs = mock_post.await_args.kwargs
    call_json = call_kwargs["json"]
    assert call_json["to"] == sender
    assert "SATYA AI — Fact Check" in call_json["text"]["body"]
    assert "pubs.usgs.gov" in call_json["text"]["body"]

    # Verify message is recorded in DB
    processed_record = (
        db_session.query(ProcessedWhatsAppMessage)
        .filter(ProcessedWhatsAppMessage.message_id == msg_id)
        .first()
    )
    assert processed_record is not None
    assert processed_record.sender_id == sender


def test_incoming_message_meta_auth_failure_does_not_poison_dedup(client: TestClient, db_session: Session):
    """
    Test that when outbound Meta dispatch encounters an auth failure (HTTP 401),
    the webhook acknowledges with 200 but does NOT mark the message processed in DB.
    """
    sender = "919075000492"
    msg_id = "wamid.TEST_AUTH_FAIL_002"
    claim_text = "Some factual claim that hits 401 auth failure."

    payload = make_whatsapp_payload(msg_id, sender, claim_text)
    body_bytes = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body_bytes, "satya_test_secret_12345")

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "error", "status_code": 401, "error": "Error validating access token: Session has expired"},
    ):
        res = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )

    assert res.status_code == 200
    outcome = res.json()["outcomes"][0]
    assert outcome["send_outcome"]["status"] == "error"
    assert outcome["send_outcome"]["status_code"] == 401

    # Crucial: message is NOT recorded in DB, allowing retries once access token is refreshed
    processed_record = (
        db_session.query(ProcessedWhatsAppMessage)
        .filter(ProcessedWhatsAppMessage.message_id == msg_id)
        .first()
    )
    assert processed_record is None


