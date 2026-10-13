"""Google Fact Check Tools API integration."""

import logging
from typing import List, Optional
import httpx

from app.config import Settings, get_settings
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.url_normalizer import normalize_url

logger = logging.getLogger(__name__)


class GoogleFactCheckClient:
    """Client for interacting with Google Fact Check Tools API."""

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def is_available(self) -> bool:
        """Check if Google Fact Check API key is configured."""
        return bool(self.settings.GOOGLE_FACT_CHECK_API_KEY)

    async def search_claims(self, query: str) -> List[RetrievedEvidenceRecord]:
        """Query Google Fact Check Tools API for previously fact-checked claims."""
        if not self.is_available():
            logger.info("Google Fact Check API key not configured; skipping fact check lookup.")
            return []

        api_key = self.settings.GOOGLE_FACT_CHECK_API_KEY
        timeout = self.settings.RETRIEVAL_TIMEOUT_SECONDS

        try:
            params = {
                "query": query,
                "key": api_key,
                "pageSize": 5,
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.BASE_URL, params=params)
                if response.status_code == 429:
                    logger.warning("Google Fact Check API rate limited (HTTP 429).")
                    return []
                response.raise_for_status()
                data = response.json()

            records: List[RetrievedEvidenceRecord] = []
            claims = data.get("claims", [])
            for c in claims:
                for review in c.get("claimReview", []):
                    raw_url = review.get("url")
                    if not raw_url:
                        continue
                    canonical_url = normalize_url(raw_url)
                    publisher_name = review.get("publisher", {}).get("name", "Unknown Fact Checker")
                    textual_rating = review.get("textualRating", "Unrated")
                    review_title = review.get("title") or c.get("text") or "Fact Check Article"
                    review_date = review.get("reviewDate")

                    records.append(
                        RetrievedEvidenceRecord(
                            title=review_title,
                            canonical_url=canonical_url,
                            publisher=publisher_name,
                            published_date=review_date,
                            relevant_text_excerpt=f"Verdict/Rating: {textual_rating}. Summary: {c.get('text', '')}",
                            source_type="fact_check",
                            retrieval_method="google_fact_check",
                            reliability_score=0.95,
                        )
                    )

            logger.info(f"Google Fact Check API returned {len(records)} verified reviews for query '{query[:30]}...'.")
            return records

        except httpx.TimeoutException:
            logger.warning(f"Google Fact Check API request timed out for query '{query[:30]}...'.")
            return []
        except Exception as exc:
            logger.warning(f"Google Fact Check API request failed: {exc}")
            return []
