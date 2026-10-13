"""Source quality assessment, domain authority tiers, and lineage detection."""

import re
from typing import List, NamedTuple, Optional, Set
from urllib.parse import urlparse


class SourceQuality(NamedTuple):
    """Categorized source authority evaluation."""

    source_type: str
    tier: int  # 1 = Authoritative / Primary, 2 = High Quality Reference/News, 3 = General Web, 4 = Low Quality / UGC
    reliability_score: float
    is_authoritative: bool
    is_user_generated: bool
    publisher_display: str


# Tier 1: Scientific Institutions, National Geological Surveys, Health Bodies, Government, Peer-Reviewed Portals
SCIENTIFIC_AND_PRIMARY_DOMAINS = {
    # Institutional & Multilateral
    "who.int", "un.org", "wmo.int", "ipcc.ch", "iaea.org", "unesco.org",
    # Government Science, Health, and Geological Surveys
    "usgs.gov", "nasa.gov", "nih.gov", "cdc.gov", "noaa.gov", "nist.gov",
    "energy.gov", "epa.gov", "bgs.ac.uk", "geology.gov",
    # Peer-reviewed academic journals & scientific societies
    "nature.com", "science.org", "cell.com", "pnas.org", "sciencedirect.com",
    "royalsocietypublishing.org", "planetary.org", "agu.org", "geosociety.org",
    "annualreviews.org", "iop.org", "aps.org", "springer.com", "wiley.com",
}

# Certified Fact-Checking Outlets
FACT_CHECK_DOMAINS = {
    "factcheck.org", "snopes.com", "politifact.com", "fullfact.org",
    "leadstories.com", "boomlive.in", "altnews.in",
}

# Tier 2: Reputable Reference, Science Journalism & Major Wire Services
REPUTABLE_REFERENCE_AND_NEWS = {
    "wikipedia.org", "britannica.com", "nationalgeographic.com",
    "education.nationalgeographic.org", "scientificamerican.com", "newscientist.com",
    "smithsonianmag.com", "reuters.com", "apnews.com", "bbc.com",
    "bbc.co.uk", "nytimes.com", "theguardian.com", "wsj.com", "economist.com",
}

# Tier 4: User-Generated Content, Video Sharing, Discussion Forums & Social Media
UGC_AND_SOCIAL_DOMAINS = {
    "youtube.com", "youtu.be", "reddit.com", "facebook.com", "twitter.com",
    "x.com", "tiktok.com", "instagram.com", "quora.com", "medium.com",
    "substack.com", "blogspot.com", "wordpress.com", "threads.net",
    "rumble.com", "dailymotion.com", "vimeo.com", "pinterest.com",
    "tumblr.com", "vk.com", "telegram.org", "t.me",
}


def extract_clean_domain(url_or_domain: str) -> str:
    """Extract clean domain without www, protocol, or path."""
    if not url_or_domain:
        return "unknown"
    val = url_or_domain.strip().lower()
    if "://" in val:
        parsed = urlparse(val)
        val = parsed.netloc or val
    val = val.split("/")[0].split(":")[0]
    if val.startswith("www."):
        val = val[4:]
    return val


def evaluate_source_quality(
    canonical_url: str,
    raw_publisher: Optional[str] = None,
    source_type: Optional[str] = None,
    reliability_score: Optional[float] = None,
) -> SourceQuality:
    """
    Classify a source into an authority tier and assign an evidence-calibrated reliability score:
    - Tier 1 (0.92 - 0.98): Primary institutional, government, peer-reviewed scientific journals, certified fact-checkers.
    - Tier 2 (0.80 - 0.88): Established encyclopedias (Wikipedia, Britannica), science journalism, wire news.
    - Tier 3 (0.60 - 0.70): General web portals and standard publishers.
    - Tier 4 (0.25 - 0.40): User-generated content, video streaming (YouTube), social media, discussion forums.
    """
    domain = extract_clean_domain(canonical_url or raw_publisher or "")

    # Check Tier 4: UGC / Social Media first to catch youtube.com, reddit.com, etc.
    if any(ugc in domain for ugc in UGC_AND_SOCIAL_DOMAINS):
        return SourceQuality(
            source_type="ugc_social",
            tier=4,
            reliability_score=0.35,
            is_authoritative=False,
            is_user_generated=True,
            publisher_display=raw_publisher or domain,
        )

    # If source is already explicitly classified as fact_check or domain is in fact check list
    if (source_type == "fact_check") or any(fc in domain for fc in FACT_CHECK_DOMAINS):
        return SourceQuality(
            source_type="fact_check",
            tier=1,
            reliability_score=reliability_score or 0.95,
            is_authoritative=True,
            is_user_generated=False,
            publisher_display=raw_publisher or domain,
        )

    # Check Tier 1: Scientific Institutions, Government, Peer-Reviewed, or explicit academic/primary
    if (
        any(inst in domain for inst in SCIENTIFIC_AND_PRIMARY_DOMAINS)
        or domain.endswith(".gov")
        or domain.endswith(".int")
        or domain.endswith(".mil")
        or domain.endswith(".edu")
        or domain.endswith(".ac.uk")
        or (source_type in ("primary_source", "academic", "government", "official"))
    ):
        score = reliability_score or (0.98 if any(inst in domain for inst in ["who.int", "usgs.gov", "nasa.gov", "cdc.gov"]) else 0.94)
        return SourceQuality(
            source_type=source_type if source_type in ("primary_source", "academic", "government") else "primary_source",
            tier=1,
            reliability_score=score,
            is_authoritative=True,
            is_user_generated=False,
            publisher_display=raw_publisher or domain,
        )

    # Check Tier 2: Wikipedia & Reference
    if "wikipedia.org" in domain:
        return SourceQuality(
            source_type="secondary_source",
            tier=2,
            reliability_score=reliability_score or 0.82,
            is_authoritative=True,
            is_user_generated=False,
            publisher_display="en.wikipedia.org" if "en.wikipedia" in domain else "wikipedia.org",
        )

    # Check Tier 2: Reputable News, Secondary Reference, or high reliability non-UGC
    if (
        any(news in domain for news in REPUTABLE_REFERENCE_AND_NEWS)
        or (source_type in ("news", "secondary_source") and (reliability_score and reliability_score >= 0.80))
        or (domain.endswith(".org") and (reliability_score and reliability_score >= 0.85))
    ):
        return SourceQuality(
            source_type=source_type or "news",
            tier=2,
            reliability_score=reliability_score or 0.88,
            is_authoritative=True,
            is_user_generated=False,
            publisher_display=raw_publisher or domain,
        )

    # Tier 3: General Web
    return SourceQuality(
        source_type=source_type or "web",
        tier=3,
        reliability_score=reliability_score or 0.65,
        is_authoritative=False,
        is_user_generated=False,
        publisher_display=raw_publisher or domain,
    )


def format_source_tier_badge(tier: int, source_type: str = "") -> str:
    """Format human-readable tier badge for WhatsApp reports and logs."""
    if tier == 1:
        return "🏛️ Authoritative Institution / Primary"
    elif tier == 2:
        return "📚 Reputable Reference / Journalism"
    elif tier == 3:
        return "🌐 Web Source"
    else:
        return "📱 User-Generated Content / Social Media"


# Known empirical/scientific benchmarks and institutional reports for lineage detection
KNOWN_BENCHMARK_PATTERNS = [
    (r"\b(patterson|canyon diablo|radiometric|lead-lead|uranium-lead|isochron)\b", "Geochronology / Radiometric Dating Benchmark"),
    (r"\b(world health assembly|global commission|resolution wha33\.3|eradication of smallpox)\b", "WHO Official Eradication Declaration"),
    (r"\b(intergovernmental panel|ipcc ar[456]|sixth assessment)\b", "IPCC Assessment Report"),
    (r"\b(associated press|ap news|reuters wire|agence france-presse|afp)\b", "Syndicated Wire Service Dispatch"),
]


def extract_lineage_signature(text_corpus: str) -> Optional[str]:
    """Detect underlying scientific benchmark, primary study, or syndication lineage in text."""
    if not text_corpus:
        return None
    corpus_lower = text_corpus.lower()
    for pat, label in KNOWN_BENCHMARK_PATTERNS:
        if re.search(pat, corpus_lower):
            return label

    # Check for general academic citation e.g. "Author et al., 2021" or "Author (1985)"
    cite_match = re.search(r"\b([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(?(?:19|20)\d{2}\)?)\b", text_corpus)
    if cite_match:
        return f"Academic Citation: {cite_match.group(1)}"
    return None


def calculate_text_similarity(t1: str, t2: str) -> float:
    """Compute token Jaccard similarity between two text snippets for syndication detection."""
    if not t1 or not t2:
        return 0.0
    stops = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "which", "are", "was", "were"}
    tokens1 = {w for w in re.findall(r"\b[a-z]{4,}\b", t1.lower()) if w not in stops}
    tokens2 = {w for w in re.findall(r"\b[a-z]{4,}\b", t2.lower()) if w not in stops}
    if not tokens1 or not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return inter / union if union > 0 else 0.0

