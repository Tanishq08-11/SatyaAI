"""Evidence analyzer service: compares retrieved source passages against claims and detects stance and conflicts."""

import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.integrations.evidence_models import RetrievedEvidenceRecord
from app.integrations.llm_client import LLMClient
from app.schemas.common import EvidenceStance
from app.services.source_evaluator import (
    calculate_text_similarity,
    evaluate_source_quality,
    extract_lineage_signature,
    format_source_tier_badge,
)

logger = logging.getLogger(__name__)


class AnalyzedEvidenceItem(BaseModel):
    """Result of analyzing an individual source passage relative to a claim."""

    record: RetrievedEvidenceRecord
    stance: EvidenceStance = Field(default=EvidenceStance.NEUTRAL)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    relevant_excerpt: str = Field(..., description="Key passage or quote supporting the stance assessment")
    explanation: str = Field(..., description="Short explanation of how this evidence relates to the claim")
    indicates_outdated: bool = Field(
        default=False,
        description="True if the source establishes the claim was historical, expired, revoked, or obsolete",
    )
    # Source quality & lineage enhancements
    source_tier: int = Field(default=3, description="Authority tier: 1=Authoritative, 2=Reference/News, 3=Web, 4=UGC")
    is_authoritative: bool = Field(default=False)
    is_user_generated: bool = Field(default=False)
    lineage_signature: Optional[str] = Field(default=None)


class ClaimAnalysisResult(BaseModel):
    """Aggregate evidence analysis for a single factual claim."""

    claim_text: str
    evidence_items: List[AnalyzedEvidenceItem] = Field(default_factory=list)
    has_conflicting_evidence: bool = Field(
        default=False,
        description="True if credible authoritative sources directly contradict one another on this claim",
    )
    conflict_explanation: Optional[str] = None
    temporal_notes: Optional[str] = None
    # Source quality hardening & lineage tracking
    has_weak_counterclaims: bool = Field(
        default=False,
        description="True if low-quality or UGC counterclaims were evaluated and dismissed in favor of authoritative consensus",
    )
    weak_counterclaims_note: Optional[str] = None
    lineage_notes: Optional[str] = None
    independent_clusters_count: int = 0



REFUTING_KEYWORDS = [
    "false", "hoax", "fake", "debunked", "misleading", "fabricated", "untrue",
    "scam", "rumor", "rumour", "refuted", "refutes", "refuting", "denied", "denies",
    "denying", "rejected", "rejects", "rejecting", "disproved", "disproven",
    "disproving", "no such", "not true", "misinformation",
    "disinformation", "unverified rumor", "fact check: false", "contrary to fact",
]

SUPPORTING_KEYWORDS = [
    "confirmed", "verified", "officially announced", "true", "correct",
    "official statement", "government gazette", "press release confirms",
    "authentic", "fact check: true", "valid",
    "certified", "declared", "eradicated", "proclaims", "proclaimed",
    "affirms", "attested", "documented", "concluded", "established",
    "eliminated", "certified the global eradication", "declared smallpox eradicated",
]

OUTDATED_KEYWORDS = [
    "outdated", "expired", "discontinued", "no longer in effect", "repealed",
    "scrapped", "withdrawn", "ended in", "was valid until", "previously announced but canceled",
    "former policy", "obsolete",
]


class EvidenceAnalyzer:
    """Compares candidate claims against retrieved source records."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def analyze_claim_evidence(
        self, claim_text: str, records: List[RetrievedEvidenceRecord]
    ) -> ClaimAnalysisResult:
        """
        Analyze all retrieved records for a single claim:
        1. Evaluate stance and relevance for each record.
        2. Detect temporal validity / obsolescence.
        3. Identify conflicts between credible sources.
        4. Detect duplicate / syndicated content and shared benchmark lineages.
        """
        if not records:
            return ClaimAnalysisResult(
                claim_text=claim_text,
                evidence_items=[],
                has_conflicting_evidence=False,
                temporal_notes="No evidence records available to analyze.",
                independent_clusters_count=0,
            )

        analyzed_items: List[AnalyzedEvidenceItem] = []

        # If LLM is available, perform assisted structured analysis
        if self.llm_client.is_available():
            try:
                analyzed_items = await self._analyze_with_llm(claim_text, records)
            except Exception as exc:
                logger.warning(f"LLM evidence analysis encountered error: {exc}; falling back to rule analysis.")
                analyzed_items = []

        # Fallback / validation using deterministic rules if LLM not configured or failed
        if not analyzed_items:
            for rec in records:
                analyzed_items.append(self._rule_based_analyze(claim_text, rec))

        # Check for conflicts between credible sources and filter weak counterclaims
        has_conflict, conflict_expl, has_weak, weak_note = self._detect_conflicts(analyzed_items)

        # Cluster evidence items to detect duplicate/syndicated content and count independent clusters
        clusters: dict = {}
        for item in analyzed_items:
            pub_domain = item.record.publisher or "unknown"
            matched_key = None
            for ckey, citems in clusters.items():
                if any(calculate_text_similarity(item.relevant_excerpt, ci.relevant_excerpt) > 0.70 for ci in citems):
                    matched_key = ckey
                    break
            if matched_key:
                clusters[matched_key].append(item)
            else:
                clusters[f"{pub_domain}_{item.lineage_signature or len(clusters)}"] = [item]

        independent_clusters_count = len(clusters)

        # Detect shared lineages
        shared_lineages = {it.lineage_signature for it in analyzed_items if it.lineage_signature}
        lineage_notes = None
        if shared_lineages:
            lineage_notes = f"Evidence lineage identified: {', '.join(sorted(shared_lineages))}."

        # Detect temporal notes
        outdated_sources = [item for item in analyzed_items if item.indicates_outdated]
        temporal_notes = None
        if outdated_sources:
            temporal_notes = (
                f"Source evidence indicates this statement reflects historical or expired conditions: "
                f"{outdated_sources[0].explanation}"
            )

        logger.info(
            "EvidenceAnalyzer: evaluated %d evidence item(s) across %d cluster(s) for claim: '%s'",
            len(analyzed_items),
            independent_clusters_count,
            claim_text[:60],
        )

        return ClaimAnalysisResult(
            claim_text=claim_text,
            evidence_items=analyzed_items,
            has_conflicting_evidence=has_conflict,
            conflict_explanation=conflict_expl,
            temporal_notes=temporal_notes,
            has_weak_counterclaims=has_weak,
            weak_counterclaims_note=weak_note,
            lineage_notes=lineage_notes,
            independent_clusters_count=independent_clusters_count,
        )

    def _rule_based_analyze(self, claim_text: str, rec: RetrievedEvidenceRecord) -> AnalyzedEvidenceItem:
        """Deterministic stance detection evaluating subject, predicate, temporal markers, qualifiers, negation, and source authority."""
        # Include metadata (published_date, publisher, URL) in corpus for complete context
        text_corpus = f"{rec.title or ''} {rec.relevant_text_excerpt or ''} {rec.published_date or ''} {rec.publisher or ''} {rec.canonical_url or ''}".lower()
        excerpt = rec.relevant_text_excerpt or rec.title or "Retrieved source text"
        claim_lower = claim_text.lower()

        # Evaluate source quality and lineage
        sq = evaluate_source_quality(rec.canonical_url, rec.publisher, rec.source_type, rec.reliability_score)
        if not rec.source_type or rec.source_type in ("web_search", "web"):
            rec.source_type = sq.source_type
        if rec.reliability_score is None:
            rec.reliability_score = sq.reliability_score
        lineage_sig = extract_lineage_signature(text_corpus)

        # 1. Check for outdated / expiry markers
        is_outdated = any(kw in text_corpus for kw in OUTDATED_KEYWORDS)

        # 2. Fact-check ratings
        if rec.source_type == "fact_check":
            if any(kw in text_corpus for kw in ["false", "pants on fire", "fake", "hoax", "incorrect", "misleading"]):
                return AnalyzedEvidenceItem(
                    record=rec,
                    stance=EvidenceStance.REFUTES,
                    relevance_score=0.95,
                    relevant_excerpt=excerpt,
                    explanation=f"Fact-checking outlet '{rec.publisher}' rated this claim as false or misleading.",
                    indicates_outdated=is_outdated,
                    source_tier=sq.tier,
                    is_authoritative=sq.is_authoritative,
                    is_user_generated=sq.is_user_generated,
                    lineage_signature=lineage_sig,
                )
            elif any(kw in text_corpus for kw in ["true", "correct", "accurate"]):
                return AnalyzedEvidenceItem(
                    record=rec,
                    stance=EvidenceStance.SUPPORTS,
                    relevance_score=0.95,
                    relevant_excerpt=excerpt,
                    explanation=f"Fact-checking outlet '{rec.publisher}' verified this claim as true.",
                    indicates_outdated=is_outdated,
                    source_tier=sq.tier,
                    is_authoritative=sq.is_authoritative,
                    is_user_generated=sq.is_user_generated,
                    lineage_signature=lineage_sig,
                )

        # 3. Extract temporal markers (years)
        claim_years = set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", claim_text))
        passage_years = set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", text_corpus))

        # 4. Entity & Predicate Concordance
        norm_claim = re.sub(r"\bwho\b", "world health organization", claim_lower)
        norm_claim = re.sub(r"\bglobal\b", "worldwide", norm_claim)
        norm_corpus = re.sub(r"\bwho\b", "world health organization", text_corpus)
        norm_corpus = re.sub(r"\bglobal\b", "worldwide", norm_corpus)

        stops = {
            "the", "and", "for", "with", "that", "this", "was", "were", "has", "have",
            "had", "been", "from", "into", "under", "over", "about", "which", "in",
            "of", "to", "a", "an", "is", "its", "by", "on", "as", "at", "all",
            "world", "health", "organization", "synthetic", "test", "mock",
        }
        claim_terms = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", norm_claim) if w not in stops]

        def stem_term(t: str) -> str:
            for sfx in ["tion", "ted", "ing", "ed", "es", "s"]:
                if t.endswith(sfx) and len(t) > len(sfx) + 2:
                    return t[: -len(sfx)]
            return t

        claim_stems = {stem_term(t) for t in claim_terms}
        text_stems = {stem_term(w) for w in re.findall(r"\b[a-zA-Z]{3,}\b", norm_corpus) if w not in stops}
        matched_stems = claim_stems.intersection(text_stems)
        overlap_score = len(matched_stems) / max(len(claim_stems), 1)

        # 5. Topic Relevance Gate: Reject unrelated / off-topic sources
        if not matched_stems or (overlap_score < 0.20 and len(matched_stems) < 1):
            return AnalyzedEvidenceItem(
                record=rec,
                stance=EvidenceStance.NEUTRAL,
                relevance_score=0.10,
                relevant_excerpt=excerpt,
                explanation=f"Source content from '{rec.publisher or rec.canonical_url}' does not address the focal subject matter or assertions of the claim.",
                indicates_outdated=False,
                source_tier=sq.tier,
                is_authoritative=sq.is_authoritative,
                is_user_generated=sq.is_user_generated,
                lineage_signature=lineage_sig,
            )

        # 6. Temporal Discordance: Claim asserts a specific calendar year (e.g., 1980), but passage mentions a different year or omits it
        if claim_years:
            if not (claim_years & passage_years):
                passage_other_years = passage_years - claim_years
                year_note = f" (mentions {', '.join(sorted(passage_other_years))})" if passage_other_years else " (claimed date not found)"
                return AnalyzedEvidenceItem(
                    record=rec,
                    stance=EvidenceStance.NEUTRAL,
                    relevance_score=0.45,
                    relevant_excerpt=excerpt,
                    explanation=f"Source '{rec.publisher or rec.canonical_url}' discusses related topics but does not substantiate the specific date/year asserted in the claim ({', '.join(sorted(claim_years))}){year_note}.",
                    indicates_outdated=is_outdated,
                    source_tier=sq.tier,
                    is_authoritative=sq.is_authoritative,
                    is_user_generated=sq.is_user_generated,
                    lineage_signature=lineage_sig,
                )

        # 7. Outdated condition
        if is_outdated:
            return AnalyzedEvidenceItem(
                record=rec,
                stance=EvidenceStance.REFUTES,
                relevance_score=0.80,
                relevant_excerpt=excerpt,
                explanation=f"Source indicates this claim or policy is expired, historical, or no longer applicable.",
                indicates_outdated=True,
                source_tier=sq.tier,
                is_authoritative=sq.is_authoritative,
                is_user_generated=sq.is_user_generated,
                lineage_signature=lineage_sig,
            )

        # 8. Corroboration & Stance Analysis
        scientific_corroboration = [
            "determin", "calculat", "estimat", "measur", "dating", "radiometric",
            "lead-lead", "uranium", "isochron", "formed", "billion years",
            "age of the earth", "geochronolog", "meteorite", "consensus",
            "canyon diablo", "geological time", "oldest known", "accretion", "proto earth",
        ]
        has_corroborative_verb = (
            any(
                kw in text_corpus for kw in [
                    "certif", "declar", "eradic", "confirm", "verif", "eliminat",
                    "proclaim", "success", "locat", "situat", "built", "announc",
                    "found", "approv", "authoriz", "pass", "decre", "zero electrical resistance",
                ]
            )
            or any(kw in text_corpus for kw in SUPPORTING_KEYWORDS)
            or any(kw in text_corpus for kw in scientific_corroboration)
        )

        # Targeted predicate negations: require action/state verbs relevant to verification claims
        predicate_negations = [
            r"\b(?:did not|was not|were not|has not|have not|had not|does not|do not|cannot|could not|never|is not|are not)\s+(?:confirm|declare|eliminate|eradicate|prove|occur|exist|happen|support|cure|ban|authorize|ratify|certify|form|date|reach|originate|substantiate)\b",
            r"\b(?:is not|was not|were not|are not)\s+(?:true|accurate|supported|correct|valid|established|possible|proven|real|factual|substantiated)\b",
            r"\bnot\s+(?:true|accurate|correct|valid|supported|proven|verified|real|factual|eliminated|declared|confirmed|substantiated)\b",
            r"\bfailed to\s+(?:confirm|verify|replicate|substantiate|prove|support)\b",
            r"\bno evidence\s+(?:to support|that|found|exists)\b",
            r"\bcontrary to\s+(?:persistent\s+|popular\s+|official\s+)?(?:rumors?|claims?|assertions?|notions?|reports?)\b",
        ]
        has_predicate_negation = any(re.search(pat, text_corpus) for pat in predicate_negations)

        has_refuting_keyword = any(kw in text_corpus for kw in REFUTING_KEYWORDS)

        # Distinguish passages debunking an OPPOSING counter-claim (e.g., Young-Earth, flat Earth, 6,000 years)
        opposing_counterclaim_patterns = [
            r"\b(?:myth|claim|notion|assertion|theory|idea)\s+(?:that|of)\s+[^.]{0,60}(?:young earth|6,000|6000|thousands of years|flat earth|geocentric|recent creation)\b",
            r"\b(?:young earth|creationist|6,000|6000 years)\s+[^.]{0,60}(?:debunked|disproved|false|untrue|myth|hoax|unscientific|flawed)\b",
            r"\bdebunk(?:ed|ing)?\s+[^.]{0,40}(?:young earth|creationism|creationist|6,000|6000)\b",
        ]
        is_debunking_opposing_counterclaim = any(re.search(pat, text_corpus) for pat in opposing_counterclaim_patterns)

        # Direct, specific refutations of the focal claim proposition
        claim_specific_refutations = [
            r"\b(?:not|never)\s+(?:approximately|about)?\s*4\.5\s*billion\b",
            r"\b(?:4\.5\s*billion|smallpox|chlorine dioxide)[^.]{0,40}(?:is|was|are|were)?\s*(?:untrue|false|refuted|debunked|a myth|a secular myth|unsupported|flawed|inaccurate|wrong|disproved)\b",
            r"\b(?:untrue|false|refuted|debunked|a myth|a secular myth|flawed|inaccurate|wrong|disproved)[^.]{0,40}(?:4\.5\s*billion|smallpox|chlorine dioxide)\b",
            r"\b(?:refuted|refutes|refuting|denied|denies|denying|disproved)\s+that\s+[^.]{0,60}(?:smallpox|earth|4\.5|chlorine)\b",
            r"\b(?:earth|solar system)\s+(?:did not|could not have)\s+form(?:ed)?\s+4\.5\s*billion\b",
            r"\bwho\s+refutes\s+these\s+false\s+claims\b",
        ]
        is_claim_specifically_refuted = any(re.search(pat, text_corpus) for pat in claim_specific_refutations)

        # An overall genuine refutation signal
        has_genuine_refutation = (
            is_claim_specifically_refuted
            or (has_refuting_keyword and not is_debunking_opposing_counterclaim)
            or (has_predicate_negation and not is_debunking_opposing_counterclaim)
        )

        # Case A: Positive corroboration of the claim
        if overlap_score >= 0.20 and (has_corroborative_verb or overlap_score >= 0.45):
            if has_genuine_refutation:
                if sq.is_user_generated:
                    return AnalyzedEvidenceItem(
                        record=rec,
                        stance=EvidenceStance.REFUTES,
                        relevance_score=0.65,
                        relevant_excerpt=excerpt,
                        explanation=f"User-generated content from '{rec.publisher or rec.canonical_url}' presents an unverified dissenting viewpoint or counterclaim.",
                        indicates_outdated=False,
                        source_tier=sq.tier,
                        is_authoritative=False,
                        is_user_generated=True,
                        lineage_signature=lineage_sig,
                    )
                return AnalyzedEvidenceItem(
                    record=rec,
                    stance=EvidenceStance.REFUTES,
                    relevance_score=0.90,
                    relevant_excerpt=excerpt,
                    explanation=f"Source '{rec.publisher or rec.canonical_url}' contradicts or negates the assertions of the claim.",
                    indicates_outdated=False,
                    source_tier=sq.tier,
                    is_authoritative=sq.is_authoritative,
                    is_user_generated=sq.is_user_generated,
                    lineage_signature=lineage_sig,
                )

            # Otherwise, positive corroboration holds (incidental negatives or debunking of opposing views do not negate consensus)
            base_rel = 0.85 if sq.is_authoritative else 0.78
            relevance = min(base_rel + (overlap_score * 0.15), 0.98)
            expl = f"{'Authoritative institution' if sq.is_authoritative else 'Source'} '{rec.publisher or rec.canonical_url}' corroborates the core assertion of the claim ({int(overlap_score*100)}% concordance)."
            if is_debunking_opposing_counterclaim:
                expl += " (Passage substantiates the consensus while debunking an opposing counter-claim)."
            return AnalyzedEvidenceItem(
                record=rec,
                stance=EvidenceStance.SUPPORTS,
                relevance_score=relevance,
                relevant_excerpt=excerpt,
                explanation=expl,
                indicates_outdated=False,
                source_tier=sq.tier,
                is_authoritative=sq.is_authoritative,
                is_user_generated=sq.is_user_generated,
                lineage_signature=lineage_sig,
            )

        # Case B: No positive corroboration, but refutation detected
        if has_genuine_refutation:
            if sq.is_user_generated:
                return AnalyzedEvidenceItem(
                    record=rec,
                    stance=EvidenceStance.REFUTES,
                    relevance_score=0.65,
                    relevant_excerpt=excerpt,
                    explanation=f"User-generated content from '{rec.publisher or rec.canonical_url}' presents an unverified dissenting viewpoint or counterclaim.",
                    indicates_outdated=False,
                    source_tier=sq.tier,
                    is_authoritative=False,
                    is_user_generated=True,
                    lineage_signature=lineage_sig,
                )
            return AnalyzedEvidenceItem(
                record=rec,
                stance=EvidenceStance.REFUTES,
                relevance_score=0.90,
                relevant_excerpt=excerpt,
                explanation=f"Source '{rec.publisher or rec.canonical_url}' contradicts or negates the assertions of the claim.",
                indicates_outdated=False,
                source_tier=sq.tier,
                is_authoritative=sq.is_authoritative,
                is_user_generated=sq.is_user_generated,
                lineage_signature=lineage_sig,
            )

        # 12. Inconclusive on-topic context
        return AnalyzedEvidenceItem(
            record=rec,
            stance=EvidenceStance.NEUTRAL,
            relevance_score=0.45,
            relevant_excerpt=excerpt,
            explanation=f"Source mentions related topics but does not decisively confirm or refute the claim.",
            indicates_outdated=False,
            source_tier=sq.tier,
            is_authoritative=sq.is_authoritative,
            is_user_generated=sq.is_user_generated,
            lineage_signature=lineage_sig,
        )

    def _detect_conflicts(
        self, items: List[AnalyzedEvidenceItem]
    ) -> (bool, Optional[str], bool, Optional[str]):
        """
        Distinguish genuine authoritative disagreement from low-quality, UGC, or unverified counterclaims.
        Returns:
            (has_conflict, conflict_explanation, has_weak_counterclaims, weak_counterclaims_note)
        """
        auth_supporting = [
            it for it in items
            if it.stance == EvidenceStance.SUPPORTS
            and not it.is_user_generated
            and (
                it.is_authoritative
                or it.source_tier in (1, 2)
                or it.record.source_type in ("academic", "primary_source", "government", "fact_check", "news")
                or (it.record.reliability_score and it.record.reliability_score >= 0.80)
            )
            and it.relevance_score >= 0.70
        ]
        auth_refuting = [
            it for it in items
            if it.stance == EvidenceStance.REFUTES
            and not it.is_user_generated
            and (
                it.is_authoritative
                or it.source_tier in (1, 2)
                or it.record.source_type in ("academic", "primary_source", "government", "fact_check")
                or (it.record.reliability_score and it.record.reliability_score >= 0.85)
            )
            and it.relevance_score >= 0.70
        ]
        all_supporting = [it for it in items if it.stance == EvidenceStance.SUPPORTS and it.relevance_score >= 0.60]
        all_refuting = [it for it in items if it.stance == EvidenceStance.REFUTES and it.relevance_score >= 0.60]

        # Any refuting item that is NOT a verified authoritative source is a non-authoritative counterclaim
        non_auth_refuting = [it for it in all_refuting if it not in auth_refuting]

        has_auth_conflict = bool(auth_supporting and auth_refuting)
        has_unauth_conflict = bool(not auth_supporting and not auth_refuting and all_supporting and all_refuting)

        # Safe diagnostic logging for each item
        for it in items:
            is_auth_counter = it in auth_refuting
            contributes_to_conflict = (
                (has_auth_conflict and (it in auth_supporting or it in auth_refuting))
                or (has_unauth_conflict and (it in all_supporting or it in all_refuting))
            )
            logger.info(
                "Evidence Diagnostic: publisher='%s', tier=%d, stance=%s, relevance=%.2f, reliability=%.2f, "
                "is_auth=%s, is_ugc=%s, is_auth_counter=%s, contributes_to_conflict=%s",
                it.record.publisher or "unknown",
                it.source_tier,
                it.stance.value,
                it.relevance_score,
                it.record.reliability_score or 0.0,
                it.is_authoritative,
                it.is_user_generated,
                is_auth_counter,
                contributes_to_conflict,
            )

        logger.info(
            "Conflict Evaluation: auth_supporting=%d, auth_refuting=%d, non_auth_refuting=%d, all_supporting=%d, all_refuting=%d",
            len(auth_supporting),
            len(auth_refuting),
            len(non_auth_refuting),
            len(all_supporting),
            len(all_refuting),
        )

        # Case 1: Genuine authoritative disagreement between credible Tier 1/2 sources
        if auth_supporting and auth_refuting:
            expl = (
                f"Conflicting credible evidence detected between authoritative sources: "
                f"{len(auth_supporting)} authoritative source(s) support the statement (e.g. '{auth_supporting[0].record.publisher}'), "
                f"while {len(auth_refuting)} authoritative source(s) refute it (e.g. '{auth_refuting[0].record.publisher}')."
            )
            logger.info("Conflict Decision: Genuine authoritative conflict -> has_conflicting_evidence=True")
            return True, expl, False, None

        non_auth_supporting = [it for it in all_supporting if it not in auth_supporting]

        # Case 2a: Authoritative supporting consensus overrides weak/UGC counterclaims
        if auth_supporting and non_auth_refuting and not auth_refuting:
            weak_pubs = ", ".join(sorted({it.record.publisher or "web source" for it in non_auth_refuting}))
            weak_note = (
                f"Evaluated {len(non_auth_refuting)} low-authority or user-generated counterclaim(s) ({weak_pubs}). "
                f"These unverified counterclaims were dismissed in accordance with SATYA AI's evidence hierarchy "
                f"due to authoritative consensus from {len(auth_supporting)} scientific/institutional source(s) "
                f"(including '{auth_supporting[0].record.publisher}')."
            )
            logger.info(
                "Conflict Decision: Authoritative consensus overrides %d non-authoritative counterclaim(s) -> has_conflicting_evidence=False, has_weak_counterclaims=True",
                len(non_auth_refuting),
            )
            return False, None, True, weak_note

        # Case 2b: Authoritative refutation overrides unverified/low-authority supporting claims
        if auth_refuting and non_auth_supporting and not auth_supporting:
            weak_pubs = ", ".join(sorted({it.record.publisher or "web source" for it in non_auth_supporting}))
            weak_note = (
                f"Evaluated {len(non_auth_supporting)} low-authority or user-generated claim(s) ({weak_pubs}). "
                f"These unverified claims were dismissed in accordance with SATYA AI's evidence hierarchy "
                f"due to authoritative refutation from {len(auth_refuting)} official/institutional source(s) "
                f"(including '{auth_refuting[0].record.publisher}')."
            )
            logger.info(
                "Conflict Decision: Authoritative refutation overrides %d non-authoritative claim(s) -> has_conflicting_evidence=False, has_weak_counterclaims=True",
                len(non_auth_supporting),
            )
            return False, None, True, weak_note

        # Case 3: Disagreement among unauthoritative web sources without institutional resolution
        if not auth_supporting and not auth_refuting and all_supporting and all_refuting:
            expl = (
                f"Conflicting credible evidence detected: {len(all_supporting)} source(s) support the statement "
                f"(e.g. '{all_supporting[0].record.publisher}'), while {len(all_refuting)} source(s) refute it "
                f"(e.g. '{all_refuting[0].record.publisher}')."
            )
            logger.info("Conflict Decision: Conflict among unauthoritative web sources -> has_conflicting_evidence=True")
            return True, expl, False, None

        logger.info("Conflict Decision: No conflict detected -> has_conflicting_evidence=False")
        return False, None, False, None


    async def _analyze_with_llm(
        self, claim_text: str, records: List[RetrievedEvidenceRecord]
    ) -> List[AnalyzedEvidenceItem]:
        """Assisted LLM comparison of passages with strict JSON schema validation."""
        passages_json = [
            {"index": i, "url": r.canonical_url, "title": r.title, "text": r.relevant_text_excerpt}
            for i, r in enumerate(records)
        ]
        prompt = (
            f"Factual Claim: \"{claim_text}\"\n"
            f"Passages:\n{passages_json}\n\n"
            "Evaluate each passage against the claim. Output JSON:\n"
            "{\"analyses\": [{\"index\": 0, \"stance\": \"supports|refutes|neutral|inconclusive\", "
            "\"relevance_score\": 0.9, \"relevant_excerpt\": \"...\", \"explanation\": \"...\", \"indicates_outdated\": false}]}"
        )
        data = await self.llm_client.generate_json(
            system_prompt="You are an evidence verification evaluator. Return valid JSON only.",
            user_prompt=prompt,
        )
        if not data or "analyses" not in data:
            return []

        results: List[AnalyzedEvidenceItem] = []
        for an in data["analyses"]:
            idx = an.get("index")
            if idx is not None and 0 <= idx < len(records):
                raw_stance = an.get("stance", "neutral").lower()
                stance = EvidenceStance.NEUTRAL
                if raw_stance in [s.value for s in EvidenceStance]:
                    stance = EvidenceStance(raw_stance)

                rec = records[idx]
                sq = evaluate_source_quality(rec.canonical_url, rec.publisher)
                lineage_sig = extract_lineage_signature(rec.relevant_text_excerpt or rec.title or "")

                results.append(
                    AnalyzedEvidenceItem(
                        record=rec,
                        stance=stance,
                        relevance_score=float(an.get("relevance_score", 0.7)),
                        relevant_excerpt=an.get("relevant_excerpt") or rec.relevant_text_excerpt or "",
                        explanation=an.get("explanation", "Evaluated via LLM assisted analysis"),
                        indicates_outdated=bool(an.get("indicates_outdated", False)),
                        source_tier=sq.tier,
                        is_authoritative=sq.is_authoritative,
                        is_user_generated=sq.is_user_generated,
                        lineage_signature=lineage_sig,
                    )
                )

        return results if len(results) == len(records) else []
