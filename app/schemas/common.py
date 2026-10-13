"""Common enumeration types and base definitions for SATYA AI schemas."""

from enum import Enum


class VerdictStatus(str, Enum):
    """Supported fact-checking verdict statuses."""

    VERIFIED = "verified"
    FALSE = "false"
    OUTDATED = "outdated"
    PARTLY_SUPPORTED = "partly_supported"
    CANNOT_CONFIRM = "cannot_confirm"


class EvidenceStance(str, Enum):
    """Stance of a source/evidence snippet relative to a specific claim."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"
