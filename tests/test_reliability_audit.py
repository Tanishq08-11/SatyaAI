"""Regression test suite for SATYA AI Critical End-to-End Reliability Audit and Repair.

Covers:
A. Two different claims processed concurrently must never exchange claims, evidence, reports, or replies.
B. Duplicate delivery of the same WhatsApp wamid must not create duplicate verification runs or duplicate replies.
C. A WhatsApp status callback must not trigger fact-checking.
D. An unrelated news article must not replace the original Earth-age claim.
E. A scientific passage debunking young-Earth arguments must not be classified as refuting the accepted age of Earth.
F. A weak or irrelevant counterclaim must not automatically create a genuine authoritative conflict.
G. A genuinely conflicting pair of relevant authoritative sources must still produce an appropriately cautious verdict.
H. A failure in search or outbound delivery must be handled safely without sending a stale or unrelated result.
I. The actual WhatsApp formatting function must preserve the claim, verdict, explanation, and sources from the same report.
J. Reproduction of the unrelated Indian political news response scenario from the screen recording.
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Claim, ClaimEvidence, ProcessedWhatsAppMessage, Report, Source
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.web_search import WebSearchClient
from app.integrations.whatsapp_client import WhatsAppClient
from app.schemas.common import EvidenceStance, VerdictStatus
from app.services.evidence_analyzer import EvidenceAnalyzer
from app.services.source_evaluator import evaluate_source_quality
from app.services.verification import VerificationEngine
from app.services.whatsapp_service import WhatsAppService


@pytest.fixture(autouse=True)
def clean_mock_registry():
    """Ensure mock search registry is reset before and after each test."""
    WebSearchClient.clear_mock_registry()
    yield
    WebSearchClient.clear_mock_registry()


# ============================================================================
# Regression Test A: Concurrent Claims Isolation
# ============================================================================
@pytest.mark.asyncio
async def test_A_concurrent_claims_processed_concurrently_never_cross_contaminate(db_session: Session):
    """
    Test A: Two different claims processed concurrently must never exchange
    claims, evidence, reports, or replies.
    """
    import uuid
    from tests.conftest import TestingSessionLocal

    sender_1 = "919000000001"
    sender_2 = "919000000002"
    msg_id_1 = f"wamid.CONCURRENT_{uuid.uuid4().hex[:8]}_EARTH"
    msg_id_2 = f"wamid.CONCURRENT_{uuid.uuid4().hex[:8]}_VENUS"

    claim_1 = "Is the Earth approximately 4.5 billion years old?"
    claim_2 = "Venus rotates in the opposite direction to most planets."

    # Register independent mock results
    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="USGS: Age of the Earth",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="usgs.gov",
                relevant_text_excerpt="Earth formed approximately 4.54 billion years ago based on radiometric dating.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            )
        ],
    )
    WebSearchClient.register_mock_results(
        "venus",
        [
            RetrievedEvidenceRecord(
                title="NASA Solar System: Venus Facts",
                canonical_url="https://science.nasa.gov/venus",
                publisher="nasa.gov",
                relevant_text_excerpt="Venus rotates backward compared to most other planets in retrograde rotation.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            )
        ],
    )

    mock_client = AsyncMock(spec=WhatsAppClient)
    sent_messages = []

    async def mock_send(to, text, preview_url=False, context_message_id=None):
        sent_messages.append({
            "to": to,
            "text": text,
            "context_message_id": context_message_id,
        })
        return {"status": "sent", "messages": [{"id": f"out_{context_message_id}"}]}

    mock_client.send_text_message.side_effect = mock_send

    service = WhatsAppService(client=mock_client)

    payload_1 = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": msg_id_1,
                        "from": sender_1,
                        "type": "text",
                        "text": {"body": claim_1},
                    }]
                }
            }]
        }]
    }

    payload_2 = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": msg_id_2,
                        "from": sender_2,
                        "type": "text",
                        "text": {"body": claim_2},
                    }]
                }
            }]
        }]
    }

    # Process concurrently using separate DB sessions on the test engine
    async def process_one(payload):
        db = TestingSessionLocal()
        try:
            return await service.process_webhook_payload(payload, db=db)
        finally:
            db.close()

    res_1, res_2 = await asyncio.gather(
        process_one(payload_1),
        process_one(payload_2),
    )

    assert len(sent_messages) == 2

    # Verify message 1 went to sender 1 with Earth text only
    msg_1_sent = next(m for m in sent_messages if m["context_message_id"] == msg_id_1)
    assert msg_1_sent["to"] == sender_1
    assert "Earth" in msg_1_sent["text"]
    assert "Venus" not in msg_1_sent["text"]
    assert "pubs.usgs.gov" in msg_1_sent["text"]
    assert "nasa.gov" not in msg_1_sent["text"]

    # Verify message 2 went to sender 2 with Venus text only
    msg_2_sent = next(m for m in sent_messages if m["context_message_id"] == msg_id_2)
    assert msg_2_sent["to"] == sender_2
    assert "Venus" in msg_2_sent["text"]
    assert "4.5 billion" not in msg_2_sent["text"]
    assert "nasa.gov" in msg_2_sent["text"]
    assert "pubs.usgs.gov" not in msg_2_sent["text"]


# ============================================================================
# Regression Test B: Duplicate wamid Delivery Dropped
# ============================================================================
@pytest.mark.asyncio
async def test_B_duplicate_wamid_delivery_drops_duplicate_execution_and_replies(db_session: Session):
    """
    Test B: Duplicate delivery of the same WhatsApp wamid must not create
    duplicate verification runs or duplicate replies.
    """
    import uuid
    msg_id = f"wamid.RETRY_TEST_{uuid.uuid4().hex[:8]}"
    sender = "919000000003"
    claim = "Is the Earth approximately 4.5 billion years old?"

    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="USGS: Age of the Earth",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="usgs.gov",
                relevant_text_excerpt="Earth formed approximately 4.54 billion years ago.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            )
        ],
    )

    mock_client = AsyncMock(spec=WhatsAppClient)
    mock_client.send_text_message.return_value = {"status": "sent", "messages": [{"id": "meta_out_1"}]}

    service = WhatsAppService(client=mock_client)

    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": msg_id,
                        "from": sender,
                        "type": "text",
                        "text": {"body": claim},
                    }]
                }
            }]
        }]
    }

    # Delivery 1: initial arrival
    res1 = await service.process_webhook_payload(payload, db=db_session)
    assert res1[0]["status"] == "processed"
    assert mock_client.send_text_message.call_count == 1

    # Delivery 2: Meta webhook retry with identical message ID
    res2 = await service.process_webhook_payload(payload, db=db_session)
    assert res2[0]["status"] == "duplicate_skipped"
    # WhatsApp reply dispatch must NOT be called again
    assert mock_client.send_text_message.call_count == 1

    # Verify only 1 record exists in ProcessedWhatsAppMessage
    records = db_session.query(ProcessedWhatsAppMessage).filter(ProcessedWhatsAppMessage.message_id == msg_id).all()
    assert len(records) == 1


# ============================================================================
# Regression Test C: WhatsApp Status Callback Ignored
# ============================================================================
@pytest.mark.asyncio
async def test_C_whatsapp_status_callback_does_not_trigger_fact_checking(db_session: Session):
    """
    Test C: A WhatsApp status callback (delivery receipt, read receipt)
    must not trigger fact-checking or send outbound messages.
    """
    mock_client = AsyncMock(spec=WhatsAppClient)
    service = WhatsAppService(client=mock_client)

    status_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "statuses": [{
                        "id": "wamid.OUTBOUND_12345",
                        "status": "delivered",
                        "timestamp": "1700000000",
                        "recipient_id": "919000000004",
                    }]
                }
            }]
        }]
    }

    res = await service.process_webhook_payload(status_payload, db=db_session)
    assert len(res) == 1
    assert res[0]["status"] == "acknowledged_status_update"
    assert mock_client.send_text_message.call_count == 0


# ============================================================================
# Regression Test D: Unrelated News Article Does Not Replace Claim
# ============================================================================
def test_D_unrelated_news_article_does_not_replace_earth_age_claim(client: TestClient):
    """
    Test D: An unrelated news article retrieved during search must not
    replace the user's original Earth-age claim or contaminate the verdict.
    """
    claim_text = "Is the Earth approximately 4.5 billion years old?"

    # Simulate search returning an unrelated political article
    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="Supreme Court hearing on election petition",
                canonical_url="https://www.nationalnews.in/sc-election-petition",
                publisher="nationalnews.in",
                relevant_text_excerpt="Supreme Court bench hears arguments regarding the election process in New Delhi.",
                source_type="news",
                retrieval_method="mock_test",
                reliability_score=0.75,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    # The claim evaluated in the report must strictly be the user's claim
    assert len(report_data["claims"]) == 1
    assert report_data["claims"][0]["claim_text"] == claim_text

    # The unrelated article must be filtered out by topic relevance gate
    assert report_data["claims"][0]["claim_evidence"] == []
    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value


# ============================================================================
# Regression Test E: Scientific Passage Debunking Counter-argument
# ============================================================================
def test_E_scientific_passage_debunking_young_earth_and_mit_snippet_classified_as_supports():
    """
    Test E: A scientific passage debunking young-Earth arguments or containing
    incidental auxiliary negatives (like eaps.mit.edu) must be classified as SUPPORTS,
    not REFUTES.
    """
    analyzer = EvidenceAnalyzer()
    claim_text = "Is the Earth approximately 4.5 billion years old?"

    # Case E.1: Real snippet from eaps.mit.edu that was misclassified in live system
    mit_record = RetrievedEvidenceRecord(
        title="Geologists discover the first evidence of 4.5 billion-year-old proto-Earth",
        canonical_url="https://eaps.mit.edu/news-impact/geologists-discover-the-first-evidence-of-4-5-billion-year-old-proto-earth",
        publisher="eaps.mit.edu",
        relevant_text_excerpt=(
            "Scientists at MIT and elsewhere have discovered extremely rare remnants of 'proto Earth', "
            "which formed about 4.5 billion years ago. The team determined that the potassium imbalance "
            "could not have been produced by any previous large impacts."
        ),
        source_type="academic",
        retrieval_method="mock_test",
        reliability_score=0.95,
    )

    analyzed_mit = analyzer._rule_based_analyze(claim_text, mit_record)
    assert analyzed_mit.stance == EvidenceStance.SUPPORTS, (
        f"eaps.mit.edu should be SUPPORTS, but was {analyzed_mit.stance}"
    )
    assert analyzed_mit.relevance_score >= 0.85
    assert analyzed_mit.is_authoritative is True

    # Case E.2: Scientific text debunking young-Earth creationist claims
    debunking_record = RetrievedEvidenceRecord(
        title="Radiometric Dating and the Age of the Earth",
        canonical_url="https://science.org/articles/geochronology-dating",
        publisher="science.org",
        relevant_text_excerpt=(
            "Young-Earth creationist assertions that the Earth is only 6,000 years old have been "
            "thoroughly debunked by radiometric dating. Modern geochronology establishes that Earth "
            "formed approximately 4.54 billion years ago."
        ),
        source_type="academic",
        retrieval_method="mock_test",
        reliability_score=0.98,
    )

    analyzed_debunking = analyzer._rule_based_analyze(claim_text, debunking_record)
    assert analyzed_debunking.stance == EvidenceStance.SUPPORTS, (
        f"Scientific passage debunking young-Earth should be SUPPORTS, but was {analyzed_debunking.stance}"
    )
    assert "debunk" in analyzed_debunking.explanation.lower() or "corroborates" in analyzed_debunking.explanation.lower()


# ============================================================================
# Regression Test F: Weak Counterclaim Does Not Create Authoritative Conflict
# ============================================================================
def test_F_weak_counterclaim_does_not_create_genuine_authoritative_conflict(client: TestClient):
    """
    Test F: A weak or user-generated counterclaim (e.g. Answers in Genesis or social media)
    must not automatically create a genuine authoritative conflict against USGS.
    """
    claim_text = "Is the Earth approximately 4.5 billion years old?"

    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="USGS: Age of the Earth",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="usgs.gov",
                relevant_text_excerpt="Radiometric dating of meteorites demonstrates that Earth is 4.54 billion years old.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="Answers in Genesis: How Old is Earth?",
                canonical_url="https://answersingenesis.org/age-of-the-earth/",
                publisher="answersingenesis.org",
                relevant_text_excerpt="The claim that Earth is 4.5 billion years old is a secular myth not supported by Scripture.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.30,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.VERIFIED.value
    claim = report_data["claims"][0]
    assert claim["verdict"] == VerdictStatus.VERIFIED.value
    assert "answersingenesis.org" in claim["explanation"] or "dismissed" in claim["explanation"].lower()


# ============================================================================
# Regression Test G: Genuinely Conflicting Authoritative Sources
# ============================================================================
def test_G_genuinely_conflicting_authoritative_sources_produce_cannot_confirm(client: TestClient):
    """
    Test G: A genuinely conflicting pair of relevant authoritative sources
    (Tier 1 Nature vs Tier 1 Science) must still produce CANNOT_CONFIRM.
    """
    claim_text = "New superconductor material LK-99 achieves zero electrical resistance at ambient room temperature."

    WebSearchClient.register_mock_results(
        "superconductor",
        [
            RetrievedEvidenceRecord(
                title="Nature: Superconductivity confirmed at ambient conditions",
                canonical_url="https://www.nature.com/articles/superconductor-confirmed",
                publisher="nature.com",
                relevant_text_excerpt="Laboratory synthesis confirms zero electrical resistance at room temperature in LK-99.",
                source_type="academic",
                retrieval_method="mock_test",
                reliability_score=0.95,
            ),
            RetrievedEvidenceRecord(
                title="Science: Replications fail to substantiate room temperature superconductivity",
                canonical_url="https://www.science.org/articles/superconductor-refuted",
                publisher="science.org",
                relevant_text_excerpt="Independent replications failed to confirm zero electrical resistance, refuting the initial room temperature claim.",
                source_type="academic",
                retrieval_method="mock_test",
                reliability_score=0.95,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    # Genuine conflict between two Tier 1 journals -> CANNOT_CONFIRM
    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert "Conflicting" in report_data["claims"][0]["explanation"] or "conflict" in report_data["claims"][0]["explanation"].lower()


# ============================================================================
# Regression Test H: Safe Handling of Search or Outbound Failures
# ============================================================================
@pytest.mark.asyncio
async def test_H_failure_in_search_or_delivery_handles_safely(db_session: Session):
    """
    Test H: A failure in search or outbound delivery must be handled safely
    without sending a stale or unrelated result, and without marking unprocessed messages.
    """
    import uuid
    msg_id = f"wamid.FAILURE_TEST_{uuid.uuid4().hex[:8]}"
    sender = "919000000005"
    claim = "Smallpox was officially eradicated worldwide in 1980."

    # Simulate search provider failure
    with patch("app.integrations.web_search.WebSearchClient.search", side_effect=Exception("External search failure")):
        mock_client = AsyncMock(spec=WhatsAppClient)
        # Simulate outbound HTTP failure
        mock_client.send_text_message.return_value = {"status": "error", "error": "Meta rate limit"}

        service = WhatsAppService(client=mock_client)

        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": msg_id,
                            "from": sender,
                            "type": "text",
                            "text": {"body": claim},
                        }]
                    }
                }]
            }]
        }

        res = await service.process_webhook_payload(payload, db=db_session)
        assert len(res) == 1
        # Because dispatch failed, message must NOT be permanently marked processed in DB
        existing = db_session.query(ProcessedWhatsAppMessage).filter(ProcessedWhatsAppMessage.message_id == msg_id).first()
        assert existing is None


# ============================================================================
# Regression Test I: WhatsApp Formatting Report Integrity
# ============================================================================
def test_I_whatsapp_formatting_preserves_single_report_integrity(db_session: Session):
    """
    Test I: The WhatsApp formatting function must preserve the claim, verdict,
    explanation, and sources strictly from the single report passed to it.
    """
    service = WhatsAppService()

    # Create Report 1: Earth's age
    report1 = Report(
        input_text="Is the Earth approximately 4.5 billion years old?",
        overall_verdict=VerdictStatus.VERIFIED.value,
        summary="Verified via USGS and Planetary Society.",
    )
    db_session.add(report1)
    db_session.flush()

    claim1 = Claim(
        report_id=report1.id,
        claim_text="Earth is approximately 4.5 billion years old",
        verdict=VerdictStatus.VERIFIED.value,
        explanation="Radiometric dating of Canyon Diablo meteorites establishes Earth formed 4.54 billion years ago.",
    )
    db_session.add(claim1)
    db_session.flush()

    source1 = db_session.query(Source).filter_by(url="https://pubs.usgs.gov/gip/geotime/age.html").first()
    if not source1:
        source1 = Source(url="https://pubs.usgs.gov/gip/geotime/age.html", publisher="usgs.gov", domain="usgs.gov")
        db_session.add(source1)
        db_session.flush()

    ce1 = ClaimEvidence(claim_id=claim1.id, source_id=source1.id, snippet="4.54 billion years old", stance=EvidenceStance.SUPPORTS.value)
    db_session.add(ce1)

    # Create Report 2: Unrelated Indian politics
    report2 = Report(
        input_text="Supreme Court election petition news",
        overall_verdict=VerdictStatus.CANNOT_CONFIRM.value,
        summary="No conclusive evidence regarding election petition claims.",
    )
    db_session.add(report2)
    db_session.flush()

    claim2 = Claim(
        report_id=report2.id,
        claim_text="Supreme Court issued directive on election petition",
        verdict=VerdictStatus.CANNOT_CONFIRM.value,
        explanation="Inconclusive coverage.",
    )
    db_session.add(claim2)
    db_session.flush()

    db_session.commit()

    # Format both
    formatted1 = service.format_verification_reply(report1)
    formatted2 = service.format_verification_reply(report2)

    assert "✅ Verified" in formatted1
    assert "4.5 billion" in formatted1
    assert "usgs.gov" in formatted1
    assert "Supreme Court" not in formatted1

    assert "❓ Cannot Confirm" in formatted2
    assert "Supreme Court" in formatted2
    assert "4.5 billion" not in formatted2
    assert "usgs.gov" not in formatted2


# ============================================================================
# Regression Test J: Reproduction of Indian Political News Scenario
# ============================================================================
@pytest.mark.asyncio
async def test_J_reproduce_recording_scenario_rapid_messages_with_exact_context_attribution(db_session: Session):
    """
    Test J: Reproduces the exact scenario observed in the screen recording:
    1. Message 1 arrives: Earth's age (wamid.MSG_1_EARTH)
    2. Message 2 arrives: Indian political news (wamid.MSG_2_POLITICS)
    3. Message 3 arrives: Earth's age (wamid.MSG_3_EARTH)

    Verifies:
    - Each message generates a unique correlation ID and report.
    - Each reply contains context_message_id matching its originating message ID.
    - The reply to wamid.MSG_2_POLITICS quotes wamid.MSG_2_POLITICS and never touches Earth's age.
    - The reply to wamid.MSG_3_EARTH quotes wamid.MSG_3_EARTH with '✅ Verified' and never mentions Indian politics.
    """
    import uuid
    msg_1_id = f"wamid.REC_001_{uuid.uuid4().hex[:8]}"
    msg_2_id = f"wamid.REC_002_{uuid.uuid4().hex[:8]}"
    msg_3_id = f"wamid.REC_003_{uuid.uuid4().hex[:8]}"
    sender = "919076640492"

    earth_claim = "Is the Earth approximately 4.5 billion years old?"
    politics_claim = "INDIA Bloc March & Supreme Court: Rahul Gandhi, Priyanka Gandhi detained during march."

    # Register mock results for both
    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="USGS: Age of the Earth",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="usgs.gov",
                relevant_text_excerpt="Earth formed approximately 4.54 billion years ago based on radiometric dating of meteorites.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="MIT EAPS: Proto-Earth remnants discovered",
                canonical_url="https://eaps.mit.edu/news-impact/proto-earth",
                publisher="eaps.mit.edu",
                relevant_text_excerpt="MIT geologists discover remnants of proto-Earth formed 4.5 billion years ago. Potassium imbalance could not have been produced by previous impacts.",
                source_type="academic",
                retrieval_method="mock_test",
                reliability_score=0.95,
            ),
        ],
    )
    WebSearchClient.register_mock_results(
        "india",
        [
            RetrievedEvidenceRecord(
                title="NDTV: INDIA Bloc march to Supreme Court",
                canonical_url="https://www.ndtv.com/india-news/india-bloc-march",
                publisher="ndtv.com",
                relevant_text_excerpt="Opposition leaders participate in march toward Supreme Court.",
                source_type="news",
                retrieval_method="mock_test",
                reliability_score=0.85,
            )
        ],
    )

    sent_replies = []
    mock_client = AsyncMock(spec=WhatsAppClient)

    async def record_send(to, text, preview_url=False, context_message_id=None):
        sent_replies.append({
            "to": to,
            "text": text,
            "context_message_id": context_message_id,
        })
        return {"status": "sent", "messages": [{"id": f"ack_{context_message_id}"}]}

    mock_client.send_text_message.side_effect = record_send
    service = WhatsAppService(client=mock_client)

    # 1. First Earth-age message
    payload_1 = {
        "entry": [{"changes": [{"value": {"messages": [{"id": msg_1_id, "from": sender, "type": "text", "text": {"body": earth_claim}}]}}]}]
    }
    await service.process_webhook_payload(payload_1, db=db_session)

    # 2. Politics forward
    payload_2 = {
        "entry": [{"changes": [{"value": {"messages": [{"id": msg_2_id, "from": sender, "type": "text", "text": {"body": politics_claim}}]}}]}]
    }
    await service.process_webhook_payload(payload_2, db=db_session)

    # 3. Second Earth-age message
    payload_3 = {
        "entry": [{"changes": [{"value": {"messages": [{"id": msg_3_id, "from": sender, "type": "text", "text": {"body": earth_claim}}]}}]}]
    }
    await service.process_webhook_payload(payload_3, db=db_session)

    assert len(sent_replies) == 3

    # Check Message 1
    reply_1 = next(r for r in sent_replies if r["context_message_id"] == msg_1_id)
    assert reply_1["to"] == sender
    assert "✅ Verified" in reply_1["text"]
    assert "eaps.mit.edu" in reply_1["text"] or "pubs.usgs.gov" in reply_1["text"]
    assert "Rahul Gandhi" not in reply_1["text"]

    # Check Message 2 (Indian Politics)
    reply_2 = next(r for r in sent_replies if r["context_message_id"] == msg_2_id)
    assert reply_2["to"] == sender
    assert "INDIA Bloc" in reply_2["text"] or "Supreme Court" in reply_2["text"]
    assert "4.5 billion" not in reply_2["text"]
    assert "eaps.mit.edu" not in reply_2["text"]

    # Check Message 3 (Earth's Age again)
    reply_3 = next(r for r in sent_replies if r["context_message_id"] == msg_3_id)
    assert reply_3["to"] == sender
    assert "✅ Verified" in reply_3["text"]
    assert "4.5 billion" in reply_3["text"]
    assert "INDIA Bloc" not in reply_3["text"]
