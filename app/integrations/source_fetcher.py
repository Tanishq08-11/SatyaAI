"""Safe direct source retriever for fetching text passages from primary sources."""

import logging
import re
from typing import Optional
import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024  # 64 KB max
SAFE_HEADERS = {
    "User-Agent": "SATYA-AI-TruthGraph-Verifier/1.0 (+https://github.com/satya-ai; research fact-check bot)",
    "Accept": "text/html,application/xhtml+xml,text/plain",
}


class SourceFetcher:
    """Safe, rate-limited single-URL excerpt fetcher."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    async def fetch_excerpt(self, url: str) -> Optional[str]:
        """
        Safely fetch a short plain-text passage from a source URL.
        Guards against oversized payloads, non-text responses, and timeouts.
        """
        if not url.startswith(("http://", "https://")):
            return None

        timeout = min(self.settings.RETRIEVAL_TIMEOUT_SECONDS, 5.0)

        try:
            async with httpx.AsyncClient(timeout=timeout, headers=SAFE_HEADERS, follow_redirects=True) as client:
                # Stream up to MAX_BODY_BYTES
                async with client.stream("GET", url) as response:
                    content_type = response.headers.get("content-type", "").lower()
                    if not any(ct in content_type for ct in ["text/html", "text/plain"]):
                        logger.info(f"Skipping direct fetch for non-text content-type '{content_type}' at {url}")
                        return None

                    if response.status_code != 200:
                        return None

                    chunks = []
                    bytes_read = 0
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                        bytes_read += len(chunk)
                        if bytes_read >= MAX_BODY_BYTES:
                            break

                    raw_html = b"".join(chunks).decode("utf-8", errors="replace")

            # Extract basic text passage: remove tags and excess whitespace
            cleaned_text = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
            cleaned_text = re.sub(r"<[^>]+>", " ", cleaned_text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            return cleaned_text[:1000] if cleaned_text else None

        except httpx.TimeoutException:
            logger.info(f"Direct source fetch timed out for URL: {url}")
            return None
        except Exception as exc:
            logger.info(f"Direct source fetch skipped for {url}: {exc}")
            return None
