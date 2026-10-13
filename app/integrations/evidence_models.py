"""Data models representing retrieved evidence records."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class RetrievedEvidenceRecord(BaseModel):
    """Normalized evidence record returned by retrieval providers."""

    title: Optional[str] = Field(None, description="Title of the source article or review")
    canonical_url: str = Field(..., description="Deduplicated canonical URL of the source")
    publisher: Optional[str] = Field(None, description="Publishing organization or outlet")
    published_date: Optional[str] = Field(
        None, description="Publication date string if provided by source metadata"
    )
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when this record was retrieved",
    )
    relevant_text_excerpt: Optional[str] = Field(
        None, description="Exact textual excerpt, summary, or review rating from the source"
    )
    source_type: str = Field(
        default="web",
        description="Type of source: 'fact_check', 'news', 'primary_source', 'government', 'academic', 'web'",
    )
    retrieval_method: str = Field(
        ...,
        description="Method/API used: 'google_fact_check', 'web_search', 'direct_fetch', 'mock_test_provider'",
    )
    reliability_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Baseline reliability score assigned to publisher/source type",
    )
