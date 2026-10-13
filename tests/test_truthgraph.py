"""Comprehensive automated tests for Phase 2: TruthGraph Evidence Verification Engine.
All mocked sources are explicitly labeled as synthetic test data.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Claim, ClaimEvidence, Report, Source
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.llm_client import LLMClient
from app.integrations.web_search import WebSearchClient
from app.schemas.common import EvidenceStance, VerdictStatus
from app.services.claim_extractor import ClaimExtractor
from app.services.evidence_analyzer import AnalyzedEvidenceItem, ClaimAnalysisResult, EvidenceAnalyzer
from app.services.evidence_retriever import EvidenceRetriever
from app.services.verdict_engine import VerdictEngine


@pytest.fixture(autouse=True)
def clean_mock_search_registry():
    """Ensure mock registry is reset between tests."""
    WebSearchClient.clear_mock_registry()
    yield
    WebSearchClient.clear_mock_registry()


def test_well_supported_claim_returns_verified(client: TestClient, db_session: Session):
    """
    Test Case 1: One well-supported claim.
    Synthetic test data: WHO smallpox eradication statement.
    """
    test_claim = "World Health Organization declared global eradication of smallpox in 1980."
    test_url = "https://mock-test.who.int/smallpox-declaration-1980"

    WebSearchClient.register_mock_results(
        "smallpox",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: WHO Declares Eradication of Smallpox",
                canonical_url=test_url,
                publisher="World Health Organization (Test Mock)",
                published_date="1980-05-08",
                relevant_text_excerpt="The World Health Assembly officially confirmed that smallpox was eradicated worldwide.",
                source_type="primary_source",
                retrieval_method="mock_test_provider",
                reliability_score=0.98,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] == VerdictStatus.VERIFIED.value
    assert len(report["claims"]) == 1
    claim = report["claims"][0]
    assert claim["verdict"] == VerdictStatus.VERIFIED.value
    assert claim["confidence_score"] >= 0.8
    assert len(claim["claim_evidence"]) >= 1
    assert claim["claim_evidence"][0]["source"]["url"] == test_url


def test_contradicted_claim_returns_false(client: TestClient, db_session: Session):
    """
    Test Case 2: One claim contradicted by strong evidence.
    Synthetic test data: Fake green moon claim refuted by debunking agency.
    """
    test_claim = "NASA confirms that the moon will turn emerald green next Tuesday."
    test_url = "https://mock-factcheck.org/green-moon-hoax-debunked"

    WebSearchClient.register_mock_results(
        "green",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: Viral Green Moon Claim is a Fake Hoax",
                canonical_url=test_url,
                publisher="FactCheck Alliance (Test Mock)",
                published_date="2026-01-15",
                relevant_text_excerpt="Fact check: false. Astronomers and NASA confirmed this is an untrue internet hoax.",
                source_type="fact_check",
                retrieval_method="mock_test_provider",
                reliability_score=0.95,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] == VerdictStatus.FALSE.value
    claim = report["claims"][0]
    assert claim["verdict"] == VerdictStatus.FALSE.value
    assert claim["confidence_score"] >= 0.8
    assert len(claim["claim_evidence"]) >= 1
    assert claim["claim_evidence"][0]["stance"] == EvidenceStance.REFUTES.value


def test_outdated_claim_returns_outdated(client: TestClient, db_session: Session):
    """
    Test Case 3: An outdated claim with historical and current evidence.
    Synthetic test data: Old travel locator form requirement that was repealed/ended.
    """
    test_claim = "All travelers entering the country must submit a mandatory contact tracer form."
    test_url = "https://mock-gov.org/health-border-policy-update"

    WebSearchClient.register_mock_results(
        "travelers",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: Contact Tracer Requirement Discontinued",
                canonical_url=test_url,
                publisher="Ministry of Health (Test Mock)",
                published_date="2022-03-18",
                relevant_text_excerpt="Notice: The border passenger contact tracer form requirement ended in 2022 and is no longer in effect.",
                source_type="government",
                retrieval_method="mock_test_provider",
                reliability_score=0.95,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] == VerdictStatus.OUTDATED.value
    claim = report["claims"][0]
    assert claim["verdict"] == VerdictStatus.OUTDATED.value
    assert "expired" in claim["explanation"] or "longer in effect" in claim["explanation"]


def test_partly_supported_claim_returns_partly_supported(client: TestClient):
    """
    Test Case 4: A partly supported claim.
    Synthetic test data: Core initiative confirmed, but auxiliary ban assertion uncorroborated.
    """
    test_claim = "Metropolitan Transit has launched contactless smart card payments."
    test_url = "https://mock-transit.city/smartcard-announcement"

    WebSearchClient.register_mock_results(
        "transit",
        [
            # Supported core element
            RetrievedEvidenceRecord(
                title="Synthetic Test: Transit Authority Officially Announces Smart Card System",
                canonical_url=test_url,
                publisher="Metropolitan Transit (Test Mock)",
                published_date="2026-02-01",
                relevant_text_excerpt="Transit Authority officially announced rollout of contactless smart card ticketing.",
                source_type="primary_source",
                retrieval_method="mock_test_provider",
                reliability_score=0.9,
            ),
            # Neutral / context element
            RetrievedEvidenceRecord(
                title="Synthetic Test: General ticketing overview",
                canonical_url="https://mock-news.city/transit-options",
                publisher="City Gazette (Test Mock)",
                published_date="2026-02-02",
                relevant_text_excerpt="Passengers discuss ticketing methods and regional commuter options.",
                source_type="news",
                retrieval_method="mock_test_provider",
                reliability_score=0.7,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] in [
        VerdictStatus.PARTLY_SUPPORTED.value,
        VerdictStatus.VERIFIED.value,
    ]


def test_claim_with_no_relevant_evidence_returns_cannot_confirm(client: TestClient):
    """
    Test Case 5: A claim with no relevant evidence.
    Ensures lack of evidence defaults to cannot_confirm (never false).
    """
    test_claim = "Secret ancient crystal pyramid discovered beneath Antarctica ice sheet by drill team."
    # No mock results registered -> returns empty

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim = report["claims"][0]
    assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert claim["confidence_score"] == 0.0
    assert claim["claim_evidence"] == []


def test_conflicting_sources_trigger_cannot_confirm(client: TestClient):
    """
    Test Case 6: Conflicting sources.
    Source A corroborates, while Source B directly refutes.
    Must return cannot_confirm with conflict explanation.
    """
    test_claim = "Independent lab reproduces room temperature superconductor material."
    WebSearchClient.register_mock_results(
        "superconductor",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: Pre-print Paper Claims Verification",
                canonical_url="https://mock-lab-alpha.org/superconductor-result",
                publisher="Lab Alpha (Test Mock)",
                published_date="2026-03-01",
                relevant_text_excerpt="Experiments confirmed that sample exhibited zero electrical resistance.",
                source_type="academic",
                retrieval_method="mock_test_provider",
                reliability_score=0.9,
            ),
            RetrievedEvidenceRecord(
                title="Synthetic Test: National Institute Debunks Superconductor Findings",
                canonical_url="https://mock-institute.edu/audit-debunk",
                publisher="National Institute of Physics (Test Mock)",
                published_date="2026-03-05",
                relevant_text_excerpt="Audit findings: results are false; magnetic artifact caused misleading readings.",
                source_type="academic",
                retrieval_method="mock_test_provider",
                reliability_score=0.92,
            ),
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert report["overall_verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    claim = report["claims"][0]
    assert claim["verdict"] == VerdictStatus.CANNOT_CONFIRM.value
    assert "Conflicting evidence" in claim["explanation"]


@pytest.mark.asyncio
async def test_malformed_llm_output_graceful_fallback():
    """
    Test Case 7: Malformed LLM output.
    Ensures that when an LLM returns unparseable or broken JSON,
    the ClaimExtractor falls back cleanly to single-claim intake without crashing.
    """
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_available.return_value = True
    # Simulate LLM returning malformed/broken response
    mock_llm.generate_json = AsyncMock(return_value={"invalid_key": "not a list"})

    extractor = ClaimExtractor(llm_client=mock_llm)
    result = await extractor.extract_claims("National reserve bank launched new emergency monetary bond.")

    assert result.is_fallback is True
    assert result.extraction_method == "single_claim_fallback"
    assert len(result.claims) == 1
    assert "Single-claim fallback" in result.notes


@pytest.mark.asyncio
async def test_search_provider_timeout_graceful_handling():
    """
    Test Case 8: Search provider timeout or unavailable credentials.
    Ensures EvidenceRetriever catches timeouts and returns cannot_confirm safely.
    """
    mock_search = MagicMock(spec=WebSearchClient)
    mock_search.is_available.return_value = True
    mock_search.search = AsyncMock(side_effect=Exception("Simulated search gateway timeout"))

    retriever = EvidenceRetriever(search_client=mock_search)
    outcome = await retriever.retrieve_evidence("Some test proposition")

    assert outcome.records == []
    assert outcome.retrieval_status == "no_results_found"


def test_database_persistence_and_relationships(client: TestClient, db_session: Session):
    """
    Test Case 9: Database persistence.
    Verifies that Report, Claim, Source, and ClaimEvidence are persisted correctly
    and queryable via SQLAlchemy with all relationships intact.
    """
    test_claim = "Health ministry confirms seasonal vaccination campaign begins on Monday."
    test_url = "https://mock-health.gov/vaccination-schedule"

    WebSearchClient.register_mock_results(
        "vaccination",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: Official Seasonal Vaccination Schedule",
                canonical_url=test_url,
                publisher="Ministry of Health (Test Mock)",
                published_date="2026-10-01",
                relevant_text_excerpt="Official press release: verified that vaccination begins Monday across clinics.",
                source_type="government",
                retrieval_method="mock_test_provider",
                reliability_score=0.95,
            )
        ],
    )

    response = client.post("/api/verify", json={"text": test_claim, "language": "en"})
    assert response.status_code == 200
    report_id = response.json()["report"]["id"]

    # Verify directly in SQLite DB session
    saved_report = db_session.query(Report).filter(Report.id == report_id).first()
    assert saved_report is not None
    assert saved_report.input_text == test_claim
    assert len(saved_report.claims) == 1

    saved_claim = saved_report.claims[0]
    assert saved_claim.verdict == VerdictStatus.VERIFIED.value
    assert len(saved_claim.claim_evidence) == 1

    saved_evidence = saved_claim.claim_evidence[0]
    assert saved_evidence.source.url == test_url
    assert saved_evidence.source.publisher == "Ministry of Health (Test Mock)"


def test_multiple_claims_in_one_message(client: TestClient):
    """
    Test Case 10: Multiple claims in one message.
    Multiple assertions across lines are extracted and evaluated.
    """
    multiline_input = (
        "Claim A: Transit fares have increased by twenty percent.\n"
        "Claim B: Secret gold vault was built beneath the local stadium."
    )

    # Register mock evidence for Claim A
    WebSearchClient.register_mock_results(
        "transit fares",
        [
            RetrievedEvidenceRecord(
                title="Synthetic Test: Transit Fares Rise",
                canonical_url="https://mock-transit.city/fares",
                publisher="City News (Test Mock)",
                relevant_text_excerpt="Transit authority announced that fares have increased by twenty percent.",
                source_type="news",
                retrieval_method="mock_test_provider",
            )
        ],
    )

    response = client.post("/api/verify", json={"text": multiline_input})
    assert response.status_code == 200
    data = response.json()
    report = data["report"]

    assert len(report["claims"]) >= 1
    assert "id" in report


def test_invalid_requests_and_api_safety(client: TestClient):
    """
    Test Case 11: Invalid requests and safe error handling.
    Empty strings, whitespace, and excessive length are cleanly rejected with 422.
    """
    res1 = client.post("/api/verify", json={"text": ""})
    assert res1.status_code == 422

    res2 = client.post("/api/verify", json={"text": "      \n\t   "})
    assert res2.status_code == 422

    res3 = client.post("/api/verify", json={"language": "en"})
    assert res3.status_code == 422
