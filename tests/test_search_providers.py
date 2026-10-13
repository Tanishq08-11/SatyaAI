"""Automated test suite for Tavily, Serper, and CombinedSearchEngine modules."""

import asyncio
from unittest.mock import patch
import httpx
import pytest

from app.config import get_settings
from app.integrations.search_providers import (
    CombinedSearchEngine,
    SerperSearchProvider,
    TavilySearchProvider,
)


@pytest.mark.asyncio
async def test_tavily_search_provider_success():
    """Test Tavily provider correctly parses response and normalizes evidence records."""
    provider = TavilySearchProvider(api_key="tvly-mock-key-123", timeout=5.0)

    mock_tavily_data = {
        "results": [
            {
                "title": "WHO Smallpox Fact Sheet",
                "url": "https://www.who.int/news-room/fact-sheets/detail/smallpox",
                "content": "In 1980, the World Health Assembly officially declared the world free of smallpox.",
                "published_date": "1980-05-08",
            },
            {
                "title": "Reuters Archives",
                "url": "https://www.reuters.com/article/idUSKBN123",
                "content": "Smallpox eradication remains one of public health's greatest achievements.",
                "published_date": "2020-05-08",
            },
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_tavily_data)
        records = await provider.search("Smallpox eradication 1980")

    assert len(records) == 2
    rec1 = records[0]
    assert "who.int" in rec1.publisher
    assert rec1.source_type == "primary_source"
    assert rec1.reliability_score == 0.98
    assert "declared the world free of smallpox" in rec1.relevant_text_excerpt

    rec2 = records[1]
    assert "reuters.com" in rec2.publisher
    assert rec2.source_type == "news"
    assert rec2.reliability_score == 0.88


@pytest.mark.asyncio
async def test_tavily_search_provider_rate_limit_and_timeout():
    """Test Tavily provider gracefully handles 429 rate limit and timeouts."""
    provider = TavilySearchProvider(api_key="tvly-mock-key-123", timeout=2.0)

    # 1. Rate limit (429)
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(429, text="Rate limit exceeded")
        records = await provider.search("test claim")
        assert records == []

    # 2. Timeout
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Tavily timeout")):
        records = await provider.search("test claim")
        assert records == []


@pytest.mark.asyncio
async def test_serper_search_provider_success():
    """Test Serper provider correctly parses organic results and extracts evidence."""
    provider = SerperSearchProvider(api_key="serper-mock-key-456", timeout=5.0)

    mock_serper_data = {
        "organic": [
            {
                "title": "Smallpox - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Smallpox",
                "snippet": "The global eradication of smallpox was certified by the World Health Organization in 1980.",
                "date": "Oct 1, 2026",
            },
            {
                "title": "CDC Smallpox Overview",
                "link": "https://www.cdc.gov/smallpox/index.html",
                "snippet": "Smallpox was eliminated worldwide following a global immunization campaign.",
                "date": "Feb 15, 2024",
            },
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_serper_data)
        records = await provider.search("smallpox eradication certified")

    assert len(records) == 2
    rec1 = records[0]
    assert rec1.publisher == "en.wikipedia.org"
    assert rec1.source_type == "secondary_source"
    assert rec1.reliability_score == 0.82

    rec2 = records[1]
    assert "cdc.gov" in rec2.publisher
    assert rec2.source_type == "primary_source"
    assert rec2.reliability_score == 0.98


@pytest.mark.asyncio
async def test_serper_search_provider_rate_limit_and_timeout():
    """Test Serper provider handles 429 rate limit and timeouts without crashing."""
    provider = SerperSearchProvider(api_key="serper-mock-key-456", timeout=2.0)

    # 1. Rate limit
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(429, text="Rate limit exceeded")
        records = await provider.search("test claim")
        assert records == []

    # 2. Timeout
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Serper timeout")):
        records = await provider.search("test claim")
        assert records == []


@pytest.mark.asyncio
async def test_combined_search_engine_merges_and_deduplicates():
    """
    Test CombinedSearchEngine:
    - Runs both Tavily and Serper concurrently.
    - Merges results and deduplicates across providers by canonical URL.
    - Preserves the richer excerpt when URLs overlap.
    """
    settings = get_settings()
    settings.TAVILY_API_KEY = "mock-tavily-key"
    settings.SERPER_API_KEY = "mock-serper-key"

    engine = CombinedSearchEngine(settings=settings)
    assert engine.is_available() is True

    # Shared URL with different snippet lengths + unique URLs
    tavily_records = [
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "WHO Smallpox",
                        "url": "https://www.who.int/health-topics/smallpox",
                        "content": "Short excerpt from Tavily.",
                    },
                    {
                        "title": "BBC News",
                        "url": "https://www.bbc.com/news/health-12345",
                        "content": "BBC report on global eradication milestones.",
                    },
                ]
            },
        )
    ]

    serper_records = [
        httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "WHO Smallpox Portal",
                        "link": "https://www.who.int/health-topics/smallpox",
                        # Much longer excerpt for the same canonical URL
                        "snippet": "In 1980 WHO declared smallpox eradicated - the only human disease to achieve this distinction worldwide.",
                    },
                    {
                        "title": "CDC Archives",
                        "link": "https://www.cdc.gov/smallpox",
                        "snippet": "CDC official historical records on smallpox campaign.",
                    },
                ]
            },
        )
    ]

    async def mock_async_post(url, *args, **kwargs):
        if "tavily" in str(url):
            return tavily_records[0]
        return serper_records[0]

    with patch("httpx.AsyncClient.post", side_effect=mock_async_post):
        results = await engine.search("smallpox eradication 1980")

    # Total unique URLs: who.int, bbc.com, cdc.gov = 3
    assert len(results) == 3
    urls = [r.canonical_url for r in results]
    assert "https://www.who.int/health-topics/smallpox" in urls
    assert "https://www.bbc.com/news/health-12345" in urls
    assert "https://www.cdc.gov/smallpox" in urls

    # Check deduplicated who.int record has the longer snippet
    who_rec = next(r for r in results if "who.int" in r.canonical_url)
    assert "only human disease to achieve this distinction" in who_rec.relevant_text_excerpt


@pytest.mark.asyncio
async def test_combined_search_engine_fallback_when_one_fails():
    """
    Test fallback behavior:
    If Tavily raises a timeout or error, Serper results are still returned successfully.
    """
    settings = get_settings()
    settings.TAVILY_API_KEY = "mock-tavily-key"
    settings.SERPER_API_KEY = "mock-serper-key"

    engine = CombinedSearchEngine(settings=settings)

    serper_response = httpx.Response(
        200,
        json={
            "organic": [
                {
                    "title": "Serper Resilient Result",
                    "link": "https://www.example.org/resilient-finding",
                    "snippet": "Evidence retrieved successfully even when peer provider fails.",
                }
            ]
        },
    )

    async def mock_async_post(url, *args, **kwargs):
        if "tavily" in str(url):
            raise httpx.ConnectError("Tavily service unreachable")
        return serper_response

    with patch("httpx.AsyncClient.post", side_effect=mock_async_post):
        results = await engine.search("resilience test query")

    assert len(results) == 1
    assert results[0].canonical_url == "https://www.example.org/resilient-finding"
