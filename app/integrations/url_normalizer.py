"""URL normalization and canonicalization utilities for evidence sources."""

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
    "_hsenc",
    "_hsmi",
    "mc_cid",
    "mc_eid",
}


def normalize_url(raw_url: str) -> str:
    """
    Produce a canonical, deduplicated URL representation by:
    - Lowercasing scheme and host
    - Removing tracking query parameters (e.g. utm_*)
    - Removing fragment anchors
    - Normalizing trailing slash on path
    """
    if not raw_url:
        return ""

    try:
        parsed = urlparse(raw_url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path
        path = parsed.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        # Filter out tracking query params
        query_pairs = parse_qs(parsed.query, keep_blank_values=False)
        clean_query_pairs = {
            k: v for k, v in query_pairs.items() if k.lower() not in TRACKING_PARAMS
        }
        # Rebuild sorted query string
        clean_query = urlencode(clean_query_pairs, doseq=True)

        return urlunparse((scheme, netloc, path, "", clean_query, ""))
    except Exception:
        return raw_url.strip()


def extract_domain(url: str) -> str:
    """Extract registered domain name from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() or "unknown"
    except Exception:
        return "unknown"
