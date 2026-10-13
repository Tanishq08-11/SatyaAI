"""Regression test suite for TruthGraph Evidence Quality Hardening.

Verifies:
1. Earth-age scientific claim: Authoritative institutions (USGS, Planetary Society, Wikipedia, NatGeo)
   override weak YouTube UGC counterclaim -> VERIFIED.
2. Weak social-media counterevidence: Authoritative NIST data overrides Reddit UGC -> VERIFIED.
3. Genuinely conflicting authoritative sources: Two Tier 1 peer-reviewed sources (Nature vs Science)
   disagree on measurement -> CANNOT_CONFIRM.
4. Insufficient evidence: Ungrounded extraordinary claim with no authoritative backing -> CANNOT_CONFIRM (confidence 0.0).
5. Duplicate / syndicated sources: Multiple wire reprints with identical excerpts are detected as
   a single underlying evidence cluster / lineage and not counted as independent corroboration.
6. WhatsApp formatting: Verifies that source authority badges and dismissed counterclaim notes
   appear accurately in mobile-formatted WhatsApp replies.
"""

from unittest.mock import AsyncMock, patch
import hashlib
import hmac
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Claim, ClaimEvidence, Report, Source
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.web_search import WebSearchClient
from app.integrations.whatsapp_client import WhatsAppClient
from app.schemas.common import EvidenceStance, VerdictStatus
from app.services.whatsapp_service import WhatsAppService


@pytest.fixture(autouse=True)
def clean_mock_registry():
    """Ensure mock search registry is reset before and after each test."""
    WebSearchClient.clear_mock_registry()
    yield
    WebSearchClient.clear_mock_registry()


def test_earth_age_scientific_consensus_overrides_ugc_youtube_counterclaim(
    client: TestClient, db_session: Session
):
    """
    Regression Test 1: 'Is the Earth approximately 4.5 billion years old?'
    Observed failure: previously returned cannot_confirm because a solitary YouTube video
    was counted as a refuting vote against USGS, Planetary Society, Wikipedia, and NatGeo.

    Expected behavior:
    - Authoritative scientific institutions (Tier 1 USGS, Planetary Society) and reference (Tier 2)
      substantiate 4.54 billion years.
    - Low-quality YouTube video counterclaim is evaluated and dismissed.
    - Overall verdict is 'verified'.
    - Explanation mentions authoritative consensus and dismissal of weak user-generated counterclaim.
    """
    claim_text = "Is the Earth approximately 4.5 billion years old?"

    # Register 4 authoritative / reputable sources and 1 low-quality YouTube counterclaim
    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="How Old is the Earth? - USGS",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="usgs.gov",
                published_date="2020-01-01",
                relevant_text_excerpt="The age of 4.54 billion years found for the Solar System and Earth is based on radiometric lead isotope dating of Canyon Diablo meteorite fragments.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="How Old Is Earth? - The Planetary Society",
                canonical_url="https://www.planetary.org/articles/how-old-is-the-earth",
                publisher="planetary.org",
                published_date="2021-05-15",
                relevant_text_excerpt="Radiometric dating has established that the Earth and Solar System formed approximately 4.54 billion years ago.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.95,
            ),
            RetrievedEvidenceRecord(
                title="Age of Earth - Wikipedia",
                canonical_url="https://en.wikipedia.org/wiki/Age_of_Earth",
                publisher="en.wikipedia.org",
                published_date="2023-08-01",
                relevant_text_excerpt="The age of Earth is estimated to be 4.54 ± 0.05 billion years based on radiometric dating of meteorite material.",
                source_type="secondary_source",
                retrieval_method="mock_test",
                reliability_score=0.82,
            ),
            RetrievedEvidenceRecord(
                title="How We Know the Age of Earth - National Geographic",
                canonical_url="https://www.nationalgeographic.com/science/article/age-of-earth",
                publisher="nationalgeographic.com",
                published_date="2022-11-10",
                relevant_text_excerpt="Geochemists and geologists have determined that Earth is approximately 4.5 billion years old using uranium-lead dating.",
                source_type="news",
                retrieval_method="mock_test",
                reliability_score=0.88,
            ),
            RetrievedEvidenceRecord(
                title="Is Earth 4.5 billion years old? Shocking new doubts about geology",
                canonical_url="https://www.youtube.com/watch?v=sample_video_doubts",
                publisher="youtube.com",
                published_date="2023-01-01",
                relevant_text_excerpt="In this video we reveal why the standard 4.5 billion year age is untrue and refuted by alternative theories.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.35,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    # Must be VERIFIED, never cannot_confirm
    assert report_data["overall_verdict"] == VerdictStatus.VERIFIED.value
    assert len(report_data["claims"]) == 1

    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.VERIFIED.value
    assert claim_res["confidence_score"] >= 0.90

    # Explanation must describe authoritative verification and dismissal of weak counterclaim
    assert "Verified" in claim_res["explanation"]
    assert "dismissed" in claim_res["explanation"].lower() or "usgs.gov" in claim_res["explanation"]

    # Verify WhatsApp formatting
    service = WhatsAppService()
    db_report = db_session.query(Report).filter(Report.id == report_data["id"]).first()
    assert db_report is not None
    reply_msg = service.format_verification_reply(db_report)

    assert "✅ Verified" in reply_msg
    assert "pubs.usgs.gov" in reply_msg
    assert "🏛️ Authoritative Institution" in reply_msg
    assert "📱 User-Generated Content" in reply_msg or "youtube.com" in reply_msg


def test_weak_social_media_counterevidence_does_not_block_verification(client: TestClient):
    """
    Regression Test 2: Authoritative evidence vs weak social-media post.
    A certified science institute (NIST) confirms boiling point of water,
    while a Reddit post denies it.
    Must return 'verified', not 'cannot_confirm'.
    """
    claim_text = "Water boils at 100 degrees Celsius under standard atmospheric pressure."

    WebSearchClient.register_mock_results(
        "water",
        [
            RetrievedEvidenceRecord(
                title="Thermophysical Properties of Fluid Systems - NIST",
                canonical_url="https://webbook.nist.gov/chemistry/fluid",
                publisher="nist.gov",
                published_date="2023-01-01",
                relevant_text_excerpt="Under standard atmospheric pressure of 1 atm (101.325 kPa), pure water boils at exactly 100 degrees Celsius.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="Reddit: Why 100C boiling water is a myth",
                canonical_url="https://www.reddit.com/r/cooking/comments/boiling_myth",
                publisher="reddit.com",
                published_date="2024-02-14",
                relevant_text_excerpt="Water does not boil at 100 degrees Celsius in my kitchen, this standard is false and refuted by my kettle.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.35,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.VERIFIED.value
    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.VERIFIED.value
    assert "dismissed" in claim_res["explanation"].lower() or "nist.gov" in claim_res["explanation"]


def test_genuinely_conflicting_authoritative_sources_preserve_cannot_confirm(client: TestClient):
    """
    Regression Test 3: Genuinely conflicting authoritative sources.
    Two Tier 1 peer-reviewed journals (Nature vs Science) present contradictory findings.
    Must return 'cannot_confirm' and state conflict between credible sources.
    """
    claim_text = "The cosmic expansion rate is established at 73 km/s/Mpc."

    WebSearchClient.register_mock_results(
        "expansion",
        [
            RetrievedEvidenceRecord(
                title="Cepheid Calibration of Distance Ladder - Nature Astronomy",
                canonical_url="https://www.nature.com/articles/sample-cepheid",
                publisher="nature.com",
                published_date="2023-03-01",
                relevant_text_excerpt="Observational measurements confirmed that the cosmic expansion rate is 73.0 km/s/Mpc with high precision.",
                source_type="academic",
                retrieval_method="mock_test",
                reliability_score=0.96,
            ),
            RetrievedEvidenceRecord(
                title="Planck CMB Observations Disprove Local Expansion Rate - Science",
                canonical_url="https://www.science.org/doi/sample-planck",
                publisher="science.org",
                published_date="2023-04-10",
                relevant_text_excerpt="Cosmic microwave background analysis refutes the 73 km/s/Mpc estimate; cosmological parameters establish 67.4 km/s/Mpc.",
                source_type="academic",
                retrieval_method="mock_test",
                reliability_score=0.96,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    # Genuine dispute between authoritative sources must preserve uncertainty
    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert "Conflicting" in claim_res["explanation"]


def test_insufficient_evidence_returns_cannot_confirm_not_false(client: TestClient):
    """
    Regression Test 4: Insufficient evidence.
    Claim with no relevant or authoritative evidence must return cannot_confirm with 0.0 confidence.
    """
    claim_text = "Quantum resonance in obsidian stones cures cellular aging at 432 hertz frequency."
    # No mock registered -> empty search

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert claim_res["confidence_score"] == 0.0
    assert claim_res["claim_evidence"] == []


def test_duplicate_and_syndicated_sources_detected_as_single_lineage(client: TestClient):
    """
    Regression Test 5: Duplicate and syndicated sources.
    Four different secondary news domains republishing the identical wire dispatch
    must be recognized as syndicated content sharing one lineage, not 4 independent corroborations.
    Because there is no primary institutional source and only 1 underlying cluster, verdict is partly_supported.
    """
    claim_text = "Global meteorological body records peak sea surface anomaly."
    syndicated_text = (
        "Associated Press / Reuters wire dispatch: A global meteorological survey confirmed that "
        "ocean sea surface temperatures reached an unprecedented measurement anomaly this season."
    )

    WebSearchClient.register_mock_results(
        "meteorological",
        [
            RetrievedEvidenceRecord(
                title="Wire: Ocean Temp Record - Portal Alpha",
                canonical_url="https://www.news-portal-alpha.com/article/sea-temp",
                publisher="news-portal-alpha.com",
                published_date="2024-03-01",
                relevant_text_excerpt=syndicated_text,
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.65,
            ),
            RetrievedEvidenceRecord(
                title="Wire: Ocean Temp Record - Herald Online",
                canonical_url="https://www.city-herald-online.com/news/sea-temp",
                publisher="city-herald-online.com",
                published_date="2024-03-01",
                relevant_text_excerpt=syndicated_text,
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.65,
            ),
            RetrievedEvidenceRecord(
                title="Wire: Ocean Temp Record - Global News Wire",
                canonical_url="https://www.global-wire-post.net/science/sea-temp",
                publisher="global-wire-post.net",
                published_date="2024-03-01",
                relevant_text_excerpt=syndicated_text,
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.65,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    claim_res = report_data["claims"][0]
    # Identical syndicated copy without official institutional or independent multi-lineage verification
    assert claim_res["verdict"] == VerdictStatus.PARTLY_SUPPORTED.value


def test_earth_age_exact_live_scenario_with_answersingenesis_and_social_evidence(
    client: TestClient, db_session: Session
):
    """
    Regression Test 6: Exact Live WhatsApp Scenario:
    Claim: "Is the Earth approximately 4.5 billion years old?"
    Retrieved Evidence:
    - 4 authoritative / news / reference sources supporting (USGS, Planetary Society, Wikipedia, NatGeo)
    - 3 UGC social media sources supporting (YouTube, Reddit, Facebook)
    - 1 Tier 3 web refuting source: answersingenesis.org (claims Earth is ~6,000 years old, refuting modern geology)

    Expected:
    - Overall verdict is 'verified'
    - Confidence score >= 0.90
    - 'answersingenesis.org' is dismissed as a low-authority counterclaim against scientific consensus
    - WhatsApp formatted message contains '✅ Verified', proper badges, and Source Quality Assessment note
    """
    claim_text = "Is the Earth approximately 4.5 billion years old?"

    answersingenesis_snippet = (
        "We never find rocks on earth that date back 4.54 billion years. Solid scientific reasons "
        "are available to doubt those dates. The Bible reveals that Earth is not 4.5 billion years old, "
        "but little more than 6,000 years old, refuting modern geology and evolutionary dating assumptions."
    )

    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="Age of the Earth - USGS",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="pubs.usgs.gov",
                published_date="2020-01-01",
                relevant_text_excerpt="The age of 4.54 billion years found for the Solar System and Earth is based on radiometric lead isotope dating of Canyon Diablo meteorite fragments.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="How Old Is Earth? - The Planetary Society",
                canonical_url="https://www.planetary.org/articles/how-old-is-the-earth",
                publisher="planetary.org",
                published_date="2021-05-15",
                relevant_text_excerpt="Radiometric dating has established that the Earth formed approximately 4.54 billion years ago.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.94,
            ),
            RetrievedEvidenceRecord(
                title="Age of Earth - Wikipedia",
                canonical_url="https://en.wikipedia.org/wiki/Age_of_Earth",
                publisher="en.wikipedia.org",
                published_date="2023-08-01",
                relevant_text_excerpt="The age of Earth is estimated to be 4.54 ± 0.05 billion years based on radiometric dating.",
                source_type="secondary_source",
                retrieval_method="mock_test",
                reliability_score=0.82,
            ),
            RetrievedEvidenceRecord(
                title="Age of Earth - National Geographic Education",
                canonical_url="https://education.nationalgeographic.org/resource/resource-age",
                publisher="education.nationalgeographic.org",
                published_date="2022-01-01",
                relevant_text_excerpt="Geologists have established the Earth is approximately 4.5 billion years old using radiometric dating.",
                source_type="news",
                retrieval_method="mock_test",
                reliability_score=0.88,
            ),
            RetrievedEvidenceRecord(
                title="How Old is the Earth? - Answers in Genesis",
                canonical_url="https://answersingenesis.org/age-of-the-earth/how-old-is-the-earth/",
                publisher="answersingenesis.org",
                published_date="2020-01-01",
                relevant_text_excerpt=answersingenesis_snippet,
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.65,
            ),
            RetrievedEvidenceRecord(
                title="Age of Earth scientific documentary clip",
                canonical_url="https://www.youtube.com/watch?v=sample_earth_video",
                publisher="youtube.com",
                published_date="2023-01-01",
                relevant_text_excerpt="In this documentary clip geologists confirm Earth formed 4.5 billion years ago.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.35,
            ),
            RetrievedEvidenceRecord(
                title="Reddit Science: Earth 4.54 billion year dating",
                canonical_url="https://www.reddit.com/r/science/comments/earth_age",
                publisher="reddit.com",
                published_date="2023-01-01",
                relevant_text_excerpt="Discussion on radiometric dating establishing Earth at 4.54 billion years.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.35,
            ),
            RetrievedEvidenceRecord(
                title="Facebook Science page: Age of Earth",
                canonical_url="https://www.facebook.com/science/posts/earth_age",
                publisher="facebook.com",
                published_date="2023-01-01",
                relevant_text_excerpt="Earth is approximately 4.5 billion years old according to radiometric data.",
                source_type="ugc_social",
                retrieval_method="mock_test",
                reliability_score=0.35,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.VERIFIED.value
    assert len(report_data["claims"]) == 1

    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.VERIFIED.value
    assert claim_res["confidence_score"] >= 0.90
    assert "answersingenesis.org" in claim_res["explanation"] or "dismissed" in claim_res["explanation"].lower()

    # Verify WhatsApp message generation
    service = WhatsAppService()
    db_report = db_session.query(Report).filter(Report.id == report_data["id"]).first()
    assert db_report is not None
    reply_msg = service.format_verification_reply(db_report)

    assert "✅ Verified" in reply_msg
    assert "answersingenesis.org" in reply_msg
    assert "pubs.usgs.gov" in reply_msg
    assert "🏛️ Authoritative Institution" in reply_msg
    assert "ℹ️ *Source Quality Assessment:*" in reply_msg


def test_authoritative_source_genuinely_refuting_claim_returns_false(client: TestClient):
    """
    Regression Test 7: An authoritative institutional source genuinely refutes a false claim.
    Claim: "The World Health Organization confirmed that chlorine dioxide cures viral infections."
    Evidence:
    - Official WHO release explicitly refutes and warns against chlorine dioxide.
    - Weak unverified web source promotes it.
    Must return 'false', not 'cannot_confirm'.
    """
    claim_text = "The World Health Organization confirmed that chlorine dioxide cures viral infections."

    WebSearchClient.register_mock_results(
        "chlorine",
        [
            RetrievedEvidenceRecord(
                title="WHO Statement on Chlorine Dioxide - World Health Organization",
                canonical_url="https://www.who.int/news/item/who-warns-chlorine-dioxide",
                publisher="who.int",
                published_date="2021-08-01",
                relevant_text_excerpt="The World Health Organization warns that chlorine dioxide is not an authorized treatment and does not cure viral infections. WHO refutes these false claims.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="Miracle Mineral Solution Protocol",
                canonical_url="https://www.health-alternative-blog.net/post/mms-cure",
                publisher="health-alternative-blog.net",
                published_date="2021-06-01",
                relevant_text_excerpt="Advocates assert chlorine dioxide cures all viral diseases rapidly.",
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.40,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": claim_text})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.FALSE.value
    claim_res = report_data["claims"][0]
    assert claim_res["verdict"] == VerdictStatus.FALSE.value
    assert claim_res["confidence_score"] >= 0.90
    assert "who.int" in claim_res["explanation"]


def test_full_whatsapp_webhook_path_verifies_earth_age_claim(client: TestClient):
    """
    Regression Test 8: End-to-end live WhatsApp webhook path for Earth-age claim.
    Simulates incoming Meta POST /webhook/whatsapp message with HMAC signature.
    Verifies that WhatsAppClient.send_text_message receives '✅ Verified' and NOT 'Cannot Confirm'.
    """
    claim_text = "Is the Earth approximately 4.5 billion years old?"
    sender = "919988776655"
    msg_id = "wamid.TEST_EARTH_AGE_001"

    WebSearchClient.register_mock_results(
        "earth",
        [
            RetrievedEvidenceRecord(
                title="How Old is the Earth? - USGS",
                canonical_url="https://pubs.usgs.gov/gip/geotime/age.html",
                publisher="pubs.usgs.gov",
                published_date="2020-01-01",
                relevant_text_excerpt="The age of 4.54 billion years found for the Solar System and Earth is based on radiometric lead isotope dating.",
                source_type="primary_source",
                retrieval_method="mock_test",
                reliability_score=0.98,
            ),
            RetrievedEvidenceRecord(
                title="How Old is the Earth? - Answers in Genesis",
                canonical_url="https://answersingenesis.org/age-of-the-earth/",
                publisher="answersingenesis.org",
                published_date="2020-01-01",
                relevant_text_excerpt="We never find rocks dating back 4.5 billion years. The Bible refutes modern geology.",
                source_type="web",
                retrieval_method="mock_test",
                reliability_score=0.65,
            ),
        ],
    )

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1000001",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550001",
                                "phone_number_id": "10987654321",
                            },
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": sender}],
                            "messages": [
                                {
                                    "id": msg_id,
                                    "from": sender,
                                    "timestamp": "1710000000",
                                    "type": "text",
                                    "text": {"body": claim_text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    body_bytes = json.dumps(payload).encode("utf-8")
    settings = get_settings()
    app_secret = settings.WHATSAPP_APP_SECRET or "satya_test_secret_12345"
    sig = f"sha256={hmac.new(app_secret.encode('utf-8'), body_bytes, hashlib.sha256).hexdigest()}"

    with patch.object(
        WhatsAppClient,
        "send_text_message",
        new_callable=AsyncMock,
        return_value={"status": "sent", "messages": [{"id": "wamid.OUT_EARTH_001"}]},
    ) as mock_send:
        response = client.post(
            "/webhook/whatsapp",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "received"

        # Verify outgoing WhatsApp message was dispatched
        mock_send.assert_awaited_once()
        sent_to = mock_send.await_args.kwargs.get("to") or mock_send.await_args.args[0]
        sent_body = mock_send.await_args.kwargs.get("text") or mock_send.await_args.args[1]
        assert sent_to == sender
        assert "✅ Verified" in sent_body
        assert "Cannot Confirm" not in sent_body
        assert "pubs.usgs.gov" in sent_body
        assert "🏛️ Authoritative Institution" in sent_body
