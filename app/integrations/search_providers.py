"""Modular, independent search providers for Tavily, Serper, and combined multi-provider research."""

import asyncio
import logging
from typing import Dict, List, Optional
import httpx

from app.config import Settings, get_settings
from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.url_normalizer import extract_domain, normalize_url
from app.services.source_evaluator import evaluate_source_quality

logger = logging.getLogger(__name__)


class TavilySearchProvider:
    """Independent Tavily search API provider."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 8.0):
        self.api_key = api_key
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[RetrievedEvidenceRecord]:
        """
        Research a factual claim using Tavily Search API.
        Extracts titles, canonical URLs, content excerpts, and publication metadata.
        """
        if not self.is_available():
            logger.debug("Tavily search provider skipped: API key not configured.")
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_domains": [],
            "exclude_domains": [],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 429:
                    logger.warning("Tavily search rate limit reached (HTTP 429).")
                    return []
                if res.is_error:
                    logger.warning("Tavily search request returned HTTP %d: %s", res.status_code, res.text)
                    return []
                data = res.json()

            records: List[RetrievedEvidenceRecord] = []
            for item in data.get("results", []):
                raw_url = item.get("url")
                if not raw_url:
                    continue
                canonical_url = normalize_url(raw_url)
                domain = extract_domain(canonical_url)
                snippet = (item.get("content") or "").strip()
                title = (item.get("title") or f"Source from {domain}").strip()
                published_date = item.get("published_date")

                # Institutional / secondary / UGC classification
                quality = evaluate_source_quality(canonical_url, raw_publisher=domain)
                source_type = quality.source_type
                reliability = quality.reliability_score

                records.append(
                    RetrievedEvidenceRecord(
                        title=title,
                        canonical_url=canonical_url,
                        publisher=domain,
                        published_date=published_date,
                        relevant_text_excerpt=snippet,
                        source_type=source_type,
                        retrieval_method="web_search_tavily",
                        reliability_score=reliability,
                    )
                )

            logger.info("Tavily search retrieved %d evidence record(s) for query: '%s'", len(records), query[:50])
            return records
        except httpx.TimeoutException:
            logger.warning("Tavily search request timed out after %ss.", self.timeout)
            return []
        except Exception as exc:
            logger.warning("Tavily search request failed: %s", exc)
            return []


class SerperSearchProvider:
    """Independent Serper Google search API provider."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 8.0):
        self.api_key = api_key
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Serper API key is configured."""
        return bool(self.api_key)

    async def search(self, query: str, num: int = 5) -> List[RetrievedEvidenceRecord]:
        """
        Search the web through Serper Google Search API.
        Extracts organic results, titles, snippets, dates, and canonical URLs.
        """
        if not self.is_available():
            logger.debug("Serper search provider skipped: API key not configured.")
            return []

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key or "",
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 429:
                    logger.warning("Serper search rate limit reached (HTTP 429).")
                    return []
                if res.is_error:
                    logger.warning("Serper search request returned HTTP %d: %s", res.status_code, res.text)
                    return []
                data = res.json()

            records: List[RetrievedEvidenceRecord] = []
            for item in data.get("organic", []):
                raw_url = item.get("link")
                if not raw_url:
                    continue
                canonical_url = normalize_url(raw_url)
                domain = extract_domain(canonical_url)
                snippet = (item.get("snippet") or "").strip()
                title = (item.get("title") or f"Source from {domain}").strip()
                date_str = item.get("date")

                # Institutional / secondary / UGC classification
                quality = evaluate_source_quality(canonical_url, raw_publisher=domain)
                source_type = quality.source_type
                reliability = quality.reliability_score

                records.append(
                    RetrievedEvidenceRecord(
                        title=title,
                        canonical_url=canonical_url,
                        publisher=domain,
                        published_date=date_str,
                        relevant_text_excerpt=snippet,
                        source_type=source_type,
                        retrieval_method="web_search_serper",
                        reliability_score=reliability,
                    )
                )

            logger.info("Serper search retrieved %d evidence record(s) for query: '%s'", len(records), query[:50])
            return records
        except httpx.TimeoutException:
            logger.warning("Serper search request timed out after %ss.", self.timeout)
            return []
        except Exception as exc:
            logger.warning("Serper search request failed: %s", exc)
            return []


class CombinedSearchEngine:
    """
    Intelligently orchestrates combined research across Tavily and Serper:
    - Runs both providers concurrently when both credentials are available.
    - Provides automatic graceful fallback when one provider fails or times out.
    - Normalizes and deduplicates results across providers by canonical URL.
    - Enriches shorter snippets when the other provider returned a longer excerpt.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        tavily_key = self.settings.TAVILY_API_KEY or (
            self.settings.SEARCH_API_KEY if (self.settings.SEARCH_PROVIDER or "").lower() == "tavily" else None
        )
        serper_key = self.settings.SERPER_API_KEY or (
            self.settings.SEARCH_API_KEY if (self.settings.SEARCH_PROVIDER or "").lower() == "serper" else None
        )
        timeout = self.settings.RETRIEVAL_TIMEOUT_SECONDS

        self.tavily = TavilySearchProvider(api_key=tavily_key, timeout=timeout)
        self.serper = SerperSearchProvider(api_key=serper_key, timeout=timeout)

    def is_available(self) -> bool:
        """True if either Tavily or Serper is configured."""
        return self.tavily.is_available() or self.serper.is_available()

    async def search(self, query: str) -> List[RetrievedEvidenceRecord]:
        """
        Execute combined research across available search providers.
        """
        tasks = []
        if self.tavily.is_available():
            tasks.append(self.tavily.search(query))
        if self.serper.is_available():
            tasks.append(self.serper.search(query))

        if not tasks:
            logger.info("Neither Tavily nor Serper API key configured.")
            return []

        # Execute available providers concurrently
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        all_records: List[RetrievedEvidenceRecord] = []
        for outcome in outcomes:
            if isinstance(outcome, list):
                all_records.extend(outcome)
            elif isinstance(outcome, Exception):
                logger.warning("Search provider task encountered exception: %s", outcome)

        # Cross-provider deduplication by canonical URL
        deduped: Dict[str, RetrievedEvidenceRecord] = {}
        for rec in all_records:
            url = rec.canonical_url
            if url not in deduped:
                deduped[url] = rec
            else:
                # Keep the richer excerpt
                existing = deduped[url]
                if len(rec.relevant_text_excerpt or "") > len(existing.relevant_text_excerpt or ""):
                    deduped[url] = rec

        combined_list = list(deduped.values())
        logger.info(
            "CombinedSearchEngine: aggregated %d unique evidence record(s) across providers for '%s'",
            len(combined_list),
            query[:50],
        )
        return combined_list
