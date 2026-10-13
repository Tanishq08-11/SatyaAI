"""Integrations package for external APIs (LLM, Fact Check Tools, Search, URL normalizer)."""

from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.fact_check_api import GoogleFactCheckClient
from app.integrations.llm_client import LLMClient
from app.integrations.source_fetcher import SourceFetcher
from app.integrations.url_normalizer import extract_domain, normalize_url
from app.integrations.web_search import WebSearchClient

__all__ = [
    "RetrievedEvidenceRecord",
    "GoogleFactCheckClient",
    "WebSearchClient",
    "SourceFetcher",
    "LLMClient",
    "normalize_url",
    "extract_domain",
]
