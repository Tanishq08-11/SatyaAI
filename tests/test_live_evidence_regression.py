"""Phase 2.2 Regression test suite: Evidence Quality Hardening and Verdict Safety.

Tests the 8 core regression scenarios required by Phase 2.2:
1. Valid WHO smallpox eradication claim (verified via primary institutional source).
2. Unsupported extraordinary claim (cannot_confirm with no evidence attached).
3. Passage mentioning smallpox but not supporting claimed date (date discordance -> cannot_confirm).
4. Passage contradicting the claim (contradictory evidence retained -> false).
5. Passage containing negation (negation detected -> refutes, not falsely marked supporting).
6. Irrelevant but reputable source (reputable WHO domain, irrelevant topic -> cannot_confirm).
7. Timeout or empty search response (graceful fallback -> cannot_confirm, never false).
8. Multi-part message containing both supported and unsupported claims (partly_supported).
+ Lineage check: Multiple pages repeating the same secondary source are not independent corroboration.
"""

from unittest.mock import patch
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Claim, ClaimEvidence, Report, Source
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.web_search import WebSearchClient
from app.schemas.common import EvidenceStance, VerdictStatus
from app.services.claim_extractor import ClaimExtractionResult, ClaimExtractor, ExtractedClaimItem


@pytest.fixture(autouse=True)
def clean_mock_registry():
    """Ensure mock registry is clean before and after each test."""
    WebSearchClient.clear_mock_registry()
    yield
    WebSearchClient.clear_mock_registry()


def test_1_valid_who_smallpox_eradication_claim(client: TestClient, db_session: Session):
    """
    Test 1: The valid WHO smallpox eradication claim.
    Authoritative primary institutional source from WHO confirms 1980 declaration.
    Must return 'verified', link evidence, and reflect official primary institutional documentation.
    """
    test_claim = "The World Health Organization confirmed the global eradication of smallpox in 1980."
    test_url = "https://www.who.int/health-topics/smallpox"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Smallpox - World Health Organization",
                canonical_url=test_url,
                publisher="who.int",
                published_date="1980",
                relevant_text_excerpt="In 1980 WHO declared smallpox eradicated - the only infectious disease to achieve this distinction.",
                source_type="primary_source",
                retrieval_method="web_search_who",
                reliability_score=0.98,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.VERIFIED.value
    assert len(report_data["claims"]) == 1
    claim_resp = report_data["claims"][0]
    assert claim_resp["verdict"] == VerdictStatus.VERIFIED.value
    assert len(claim_resp["claim_evidence"]) == 1
    evidence = claim_resp["claim_evidence"][0]
    assert evidence["source"]["url"] == test_url
    assert evidence["source"]["source_type"] == "primary_source"
    assert evidence["source"]["reliability_score"] == 0.98
    assert evidence["stance"] == EvidenceStance.SUPPORTS.value
    assert evidence["relevance_score"] >= 0.80
    assert "primary institutional" in claim_resp["explanation"] or "who.int" in claim_resp["explanation"]

    # Verify database persistence
    db_report = db_session.query(Report).filter(Report.id == report_data["id"]).first()
    assert db_report is not None
    assert len(db_report.claims) == 1
    assert db_report.claims[0].verdict == VerdictStatus.VERIFIED.value
    assert len(db_report.claims[0].claim_evidence) == 1


def test_2_unsupported_extraordinary_claim(client: TestClient):
    """
    Test 2: An unsupported extraordinary claim.
    Search returns empty results.
    Must return 'cannot_confirm', confidence 0.0, and empty evidence list.
    """
    test_claim = "Underground crystalline energy pyramid discovered beneath Mount Everest by secret miners."
    # No mock registered -> empty search result

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert len(report_data["claims"]) >= 1
    for claim in report_data["claims"]:
        assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
        assert claim["confidence_score"] == 0.0
        assert claim["claim_evidence"] == []
        assert "cannot_confirm" in claim["explanation"]


def test_3_passage_mentioning_smallpox_not_supporting_date(client: TestClient):
    """
    Test 3: A passage mentioning smallpox but not supporting the claimed date.
    Claim explicitly asserts 1980.
    Retrieved passage discusses smallpox in 1977 (Somalia case) but does not substantiate 1980.
    Temporal discordance must prevent false verification and return cannot_confirm.
    """
    test_claim = "The World Health Organization confirmed the global eradication of smallpox in 1980."
    test_url = "https://www.historyofvaccines.org/smallpox-timeline"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="History of Vaccines: Smallpox Timeline",
                canonical_url=test_url,
                publisher="historyofvaccines.org",
                published_date="2020-01-01",
                relevant_text_excerpt="In 1977, the last naturally occurring case of smallpox was diagnosed in Somalia.",
                source_type="secondary_source",
                retrieval_method="web_search",
                reliability_score=0.85,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    # Date discordance (1977 vs claimed 1980) caps relevance at 0.45, filtered by 0.60 gate -> cannot_confirm
    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim = report_data["claims"][0]
    assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert claim["claim_evidence"] == []


def test_4_passage_contradicting_claim(client: TestClient):
    """
    Test 4: A passage contradicting the claim.
    Retrieved evidence directly contradicts or refutes the assertion.
    Contradictory evidence must be retained and verdict set to 'false'.
    """
    test_claim = "The World Health Organization confirmed the global eradication of smallpox in 1980."
    test_url = "https://www.factcheck-health.org/smallpox-erroneous-claim"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Fact Check: Claims of Smallpox Eradication Debunked",
                canonical_url=test_url,
                publisher="factcheck-health.org",
                published_date="1980",
                relevant_text_excerpt="Contrary to persistent rumors, the World Health Organization refuted that smallpox was eradicated in 1980.",
                source_type="fact_check",
                retrieval_method="web_search",
                reliability_score=0.92,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.FALSE.value
    claim = report_data["claims"][0]
    assert claim["verdict"] == VerdictStatus.FALSE.value
    assert len(claim["claim_evidence"]) == 1
    evidence = claim["claim_evidence"][0]
    assert evidence["stance"] == EvidenceStance.REFUTES.value
    assert evidence["relevance_score"] >= 0.70
    assert "Contradicted" in claim["explanation"] or "refute" in claim["explanation"]


def test_5_passage_containing_negation(client: TestClient):
    """
    Test 5: A passage containing negation.
    Passage includes negation terms ("did not declare", "untrue", "never").
    Must not classify as supporting solely due to keyword overlap; must detect negation.
    """
    test_claim = "The World Health Organization declared that smallpox was eliminated worldwide in 1980."
    test_url = "https://www.health-bulletin.org/corrections/1980"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Historical Health Bulletin Corrections",
                canonical_url=test_url,
                publisher="health-bulletin.org",
                published_date="1980",
                relevant_text_excerpt="Archival records confirm that the World Health Organization did not declare smallpox eliminated worldwide.",
                source_type="news",
                retrieval_method="web_search",
                reliability_score=0.85,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.FALSE.value
    claim = report_data["claims"][0]
    assert claim["verdict"] == VerdictStatus.FALSE.value
    assert len(claim["claim_evidence"]) == 1
    assert claim["claim_evidence"][0]["stance"] == EvidenceStance.REFUTES.value


def test_6_irrelevant_but_reputable_source(client: TestClient):
    """
    Test 6: An irrelevant but reputable source.
    Source is from who.int with high reliability (0.98), but is about infant nutrition / breastfeeding.
    Must NOT mark verified solely because the source domain is reputable.
    Sufficiency gate must discard off-topic evidence and return cannot_confirm with empty evidence list.
    """
    test_claim = "The World Health Organization confirmed the global eradication of smallpox in 1980."
    test_url = "https://www.who.int/health-topics/breastfeeding"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Infant and Young Child Feeding - World Health Organization",
                canonical_url=test_url,
                publisher="who.int",
                published_date="2023-06-09",
                relevant_text_excerpt="The World Health Organization recommends exclusive breastfeeding starting within one hour of birth.",
                source_type="primary_source",
                retrieval_method="web_search_who",
                reliability_score=0.98,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim = report_data["claims"][0]
    assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    # Off-topic source must be discarded from qualifying evidence used to justify a verdict
    assert claim["claim_evidence"] == []


def test_7_timeout_or_empty_search_response(client: TestClient):
    """
    Test 7: A timeout or empty search response.
    Upstream search provider experiences an HTTP timeout or network error.
    Must return cannot_confirm with empty evidence list.
    Must NOT interpret a failed search or timeout as evidence that the claim is false.
    """
    test_claim = "Authorities announced an emergency power grid shutdown across northern districts."

    with patch.object(WebSearchClient, "search", side_effect=httpx.TimeoutException("Search provider timed out")):
        response = client.post("/api/verify", json={"text": test_claim})
        assert response.status_code == 200
        report_data = response.json()["report"]

        assert report_data["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
        for claim in report_data["claims"]:
            assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
            assert claim["claim_evidence"] == []
            # Explanation must be cautious, not declaring the claim false
            assert "cannot_confirm" in claim["explanation"] or "Insufficient" in claim["explanation"]
            assert "false" not in claim["verdict"].lower()


def test_8_multipart_message_both_supported_and_unsupported_claims(client: TestClient):
    """
    Test 8: A multi-part message containing both supported and unsupported claims.
    Claim 1 is verified with official primary institutional evidence from WHO.
    Claim 2 is unsupported extraordinary assertion with no search evidence.
    Overall verdict must reflect all extracted claims -> 'partly_supported'.
    """
    input_text = (
        "The World Health Organization confirmed global eradication of smallpox in 1980. "
        "A clandestine underground alien civilization generates magnetic pulses beneath the Andes."
    )

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
    # Alien claim query returns empty
    WebSearchClient.register_mock_results("alien", [])

    claim_1 = ExtractedClaimItem(text="The World Health Organization confirmed global eradication of smallpox in 1980.")
    claim_2 = ExtractedClaimItem(text="A clandestine underground alien civilization generates magnetic pulses beneath the Andes.")
    mock_result = ClaimExtractionResult(
        claims=[claim_1, claim_2],
        extraction_method="llm_extraction",
        is_fallback=False,
    )

    with patch.object(ClaimExtractor, "extract_claims", return_value=mock_result):
        response = client.post("/api/verify", json={"text": input_text})

    assert response.status_code == 200
    report_data = response.json()["report"]

    # Overall verdict must be partly_supported reflecting mixed status
    assert report_data["overall_verdict"] == VerdictStatus.PARTLY_SUPPORTED.value
    assert len(report_data["claims"]) == 2

    smallpox_claim = next(c for c in report_data["claims"] if "smallpox" in c["claim_text"].lower())
    alien_claim = next(c for c in report_data["claims"] if "alien" in c["claim_text"].lower())

    assert smallpox_claim["verdict"] == VerdictStatus.VERIFIED.value
    assert len(smallpox_claim["claim_evidence"]) == 1
    assert smallpox_claim["claim_evidence"][0]["source"]["url"] == "https://www.who.int/health-topics/smallpox"

    assert alien_claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert alien_claim["claim_evidence"] == []


def test_multiple_pages_same_source_not_independent_corroboration(client: TestClient):
    """
    Test that multiple pages repeating the same underlying secondary source
    (e.g., multiple Wikipedia URLs) do NOT count as independent corroboration.
    Verdict must be partly_supported rather than verified.
    """
    test_claim = "Astronomers measured rapid orbital decay of binary neutron star."
    wiki_url_1 = "https://en.wikipedia.org/wiki/Binary_pulsar"
    wiki_url_2 = "https://en.wikipedia.org/wiki/PSR_B1913%2B16"

    WebSearchClient.register_mock_results(
        "neutron",
        [
            RetrievedEvidenceRecord(
                title="Binary Pulsar - Wikipedia",
                canonical_url=wiki_url_1,
                publisher="en.wikipedia.org",
                published_date="2024-01-01",
                relevant_text_excerpt="Observations confirmed rapid orbital decay matching general relativistic predictions.",
                source_type="secondary_source",
                retrieval_method="web_search_wikipedia",
                reliability_score=0.82,
            ),
            RetrievedEvidenceRecord(
                title="PSR B1913+16 - Wikipedia",
                canonical_url=wiki_url_2,
                publisher="en.wikipedia.org",
                published_date="2024-01-01",
                relevant_text_excerpt="Measurements confirmed the orbital decay of the binary system over multiple decades.",
                source_type="secondary_source",
                retrieval_method="web_search_wikipedia",
                reliability_score=0.82,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    report_data = response.json()["report"]

    claim = report_data["claims"][0]
    # Single secondary publisher lineage does NOT qualify as verified; returns partly_supported
    assert claim["verdict"] == VerdictStatus.PARTLY_SUPPORTED.value
    assert "not count as independent corroboration" in claim["explanation"]
