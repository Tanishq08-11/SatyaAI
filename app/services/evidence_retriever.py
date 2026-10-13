"""Evidence retriever service: orchestrates multi-source evidence discovery and deduplication."""

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.fact_check_api import GoogleFactCheckClient
from app.integrations.source_fetcher import SourceFetcher
from app.integrations.url_normalizer import normalize_url
from app.integrations.web_search import WebSearchClient

logger = logging.getLogger(__name__)


class RetrievalOutcome(BaseModel):
    """Aggregate result of evidence retrieval for a specific claim."""

    claim_text: str
    records: List[RetrievedEvidenceRecord] = Field(default_factory=list)
    retrieval_status: str = Field(
        ...,
        description="Status: 'success', 'no_results_found', 'credentials_missing', or 'retrieval_failed'",
    )
    notes: Optional[str] = None


class EvidenceRetriever:
    """Orchestrates evidence discovery across fact check APIs and search providers."""

    def __init__(
        self,
        fact_check_client: Optional[GoogleFactCheckClient] = None,
        search_client: Optional[WebSearchClient] = None,
        source_fetcher: Optional[SourceFetcher] = None,
    ):
        self.fact_check_client = fact_check_client or GoogleFactCheckClient()
        self.search_client = search_client or WebSearchClient()
        self.source_fetcher = source_fetcher or SourceFetcher()

    async def retrieve_evidence(self, claim_text: str) -> RetrievalOutcome:
        """
        Discover evidence for a single factual claim:
        1. Query Google Fact Check Tools API (highest priority for debunked/verified claims).
        2. Query Web Search provider for primary reports/articles.
        3. Deduplicate by canonical URL.
        4. Enhance snippets with safe direct fetch if snippet is brief.
        """
        cleaned_claim = claim_text.strip()
        if not cleaned_claim:
            return RetrievalOutcome(
                claim_text=claim_text,
                records=[],
                retrieval_status="no_results_found",
                notes="Claim text was empty.",
            )

        has_fact_check = self.fact_check_client.is_available()
        has_search = self.search_client.is_available()

        if not has_fact_check and not has_search:
            logger.info("No external evidence retrieval providers or credentials configured.")
            return RetrievalOutcome(
                claim_text=cleaned_claim,
                records=[],
                retrieval_status="credentials_missing",
                notes="Evidence retrieval credentials not configured. Cautious pipeline cannot confirm without evidence.",
            )

        discovered_by_url: Dict[str, RetrievedEvidenceRecord] = {}

        # 1. Fact Check Tools API lookup
        if has_fact_check:
            try:
                fact_check_results = await self.fact_check_client.search_claims(cleaned_claim)
                for rec in fact_check_results:
                    norm_url = normalize_url(rec.canonical_url)
                    if norm_url and norm_url not in discovered_by_url:
                        rec.canonical_url = norm_url
                        discovered_by_url[norm_url] = rec
            except Exception as exc:
                logger.warning(f"Fact check retrieval error: {exc}")

        # 2. Web search lookup
        if has_search:
            try:
                search_results = await self.search_client.search(cleaned_claim)
                for rec in search_results:
                    norm_url = normalize_url(rec.canonical_url)
                    if norm_url and norm_url not in discovered_by_url:
                        rec.canonical_url = norm_url
                        discovered_by_url[norm_url] = rec
            except Exception as exc:
                logger.warning(f"Web search retrieval error: {exc}")

        records = list(discovered_by_url.values())

        # 3. Optional snippet enhancement via direct fetch (only for top web results with very brief snippets)
        for rec in records[:2]:
            if rec.source_type == "web" and (not rec.relevant_text_excerpt or len(rec.relevant_text_excerpt) < 60):
                try:
                    direct_text = await self.source_fetcher.fetch_excerpt(rec.canonical_url)
                    if direct_text:
                        rec.relevant_text_excerpt = direct_text[:500]
                except Exception:
                    pass

        status = "success" if records else "no_results_found"
        notes = (
            f"Retrieved {len(records)} deduplicated evidence sources."
            if records
            else "No matching evidence records discovered from available providers."
        )

        return RetrievalOutcome(
            claim_text=cleaned_claim,
            records=records,
            retrieval_status=status,
            notes=notes,
        )
