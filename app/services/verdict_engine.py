"""Verdict engine implementing the evidence sufficiency gate and explicit verification rules."""

import logging
from typing import List, Tuple
from pydantic import BaseModel, Field

from app.schemas.common import EvidenceStance, VerdictStatus
from app.services.evidence_analyzer import AnalyzedEvidenceItem, ClaimAnalysisResult

logger = logging.getLogger(__name__)


class ClaimVerdictEvaluation(BaseModel):
    """Evaluated verdict for a single claim with confidence and explicit reasoning."""

    verdict: VerdictStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    qualifying_evidence: List[AnalyzedEvidenceItem] = Field(default_factory=list)


class ReportVerdictEvaluation(BaseModel):
    """Aggregate verdict and summary across all claims in a report."""

    overall_verdict: VerdictStatus
    summary: str


class VerdictEngine:
    """Applies strict, testable sufficiency rules to determine claim and report verdicts."""

    # Thresholds
    MIN_RELEVANCE_THRESHOLD: float = 0.6
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8

    def evaluate_claim_verdict(self, analysis: ClaimAnalysisResult) -> ClaimVerdictEvaluation:
        """
        Evaluate a single claim against explicit sufficiency gate rules:
        1. Conflicting evidence -> cannot_confirm
        2. Insufficient relevant evidence -> cannot_confirm (lack of results is NEVER false)
        3. Documented expiry/obsolescence -> outdated
        4. Strong refutation without contradiction -> false
        5. Corroborated with unverified/neutral components -> partly_supported
        6. Fully corroborated by reliable sources -> verified
        """
        # Rule 1: Conflicting evidence from credible sources
        if analysis.has_conflicting_evidence:
            logger.info(
                "Final Reason for CANNOT_CONFIRM [Rule 1]: Conflicting credible evidence between sources. Details: %s",
                analysis.conflict_explanation,
            )
            return ClaimVerdictEvaluation(
                verdict=VerdictStatus.CANNOT_CONFIRM,
                confidence_score=0.4,
                explanation=(
                    f"Conflicting evidence detected between credible sources. Cannot confirm without "
                    f"authoritative consensus. {analysis.conflict_explanation or ''}"
                ),
                qualifying_evidence=analysis.evidence_items,
            )

        # Filter relevant evidence (relevance >= 0.6)
        relevant_items = [
            it for it in analysis.evidence_items if it.relevance_score >= self.MIN_RELEVANCE_THRESHOLD
        ]

        # Rule 2: Insufficient relevant evidence -> cannot_confirm
        if not relevant_items:
            logger.info(
                "Final Reason for CANNOT_CONFIRM [Rule 2]: Insufficient relevant evidence (0 items with relevance >= %.2f).",
                self.MIN_RELEVANCE_THRESHOLD,
            )
            return ClaimVerdictEvaluation(
                verdict=VerdictStatus.CANNOT_CONFIRM,
                confidence_score=0.0,
                explanation=(
                    "Insufficient relevant evidence found from authoritative sources. In accordance with "
                    "SATYA AI's evidence-first policy, ungrounded claims are assigned a cautious verdict of cannot_confirm."
                ),
                qualifying_evidence=[],
            )

        # Rule 3: Outdated (historical validity established, but superseded or expired)
        outdated_items = [it for it in relevant_items if it.indicates_outdated]
        if outdated_items:
            first_outdated = outdated_items[0]
            logger.info(
                "Sufficiency Gate [Rule 3]: Outdated/historical claim detected -> outdated from '%s'",
                first_outdated.record.publisher,
            )
            return ClaimVerdictEvaluation(
                verdict=VerdictStatus.OUTDATED,
                confidence_score=0.9,
                explanation=(
                    f"The claim references historical facts, decrees, or notices that are no longer in effect or have expired. "
                    f"Evidence from '{first_outdated.record.publisher}': {first_outdated.explanation}"
                ),
                qualifying_evidence=relevant_items,
            )

        # Categorize stance and authority
        auth_refuting_items = [
            it for it in relevant_items
            if it.stance == EvidenceStance.REFUTES
            and not it.is_user_generated
            and (
                it.is_authoritative
                or it.source_tier in (1, 2)
                or it.record.source_type in ("fact_check", "academic", "primary_source", "government")
                or (it.record.reliability_score and it.record.reliability_score >= 0.85)
            )
            and it.relevance_score >= 0.70
        ]
        auth_refuting_items.sort(
            key=lambda it: (
                it.source_tier,
                0 if it.is_authoritative else 1,
                -it.relevance_score,
                -(it.record.reliability_score or 0.0),
            )
        )

        supporting_items = [
            it for it in relevant_items if it.stance == EvidenceStance.SUPPORTS and it.relevance_score >= 0.70
        ]
        supporting_items.sort(
            key=lambda it: (
                it.source_tier,
                0 if it.is_authoritative else 1,
                -it.relevance_score,
                -(it.record.reliability_score or 0.0),
            )
        )

        neutral_or_inconclusive = [
            it for it in relevant_items if it.stance in (EvidenceStance.NEUTRAL, EvidenceStance.INCONCLUSIVE)
        ]

        logger.info(
            "VerdictEngine Evaluation: relevant=%d, supporting=%d, auth_refuting=%d, has_conflict=%s, has_weak_counterclaims=%s",
            len(relevant_items),
            len(supporting_items),
            len(auth_refuting_items),
            analysis.has_conflicting_evidence,
            analysis.has_weak_counterclaims,
        )

        auth_supporting_items = [
            it for it in supporting_items
            if not it.is_user_generated
            and (
                it.is_authoritative
                or it.source_tier in (1, 2)
                or it.record.source_type in ("academic", "primary_source", "government", "official", "fact_check", "news")
                or (it.record.reliability_score and it.record.reliability_score >= 0.80)
            )
        ]

        # Rule 4: Strong contradictory evidence from authoritative sources -> false
        # Authoritative refutation overrides absence of support OR unverified/low-authority claims
        if auth_refuting_items and not auth_supporting_items:
            primary_refuter = auth_refuting_items[0]
            weak_note = (
                f" Note: {analysis.weak_counterclaims_note}"
                if analysis.has_weak_counterclaims and analysis.weak_counterclaims_note
                else ""
            )
            logger.info(
                "Sufficiency Gate [Rule 4]: Authoritative contradiction substantiated -> false from '%s'",
                primary_refuter.record.publisher,
            )
            return ClaimVerdictEvaluation(
                verdict=VerdictStatus.FALSE,
                confidence_score=0.95,
                explanation=(
                    f"Contradicted based on authoritative evidence from '{primary_refuter.record.publisher}'. "
                    f"{primary_refuter.explanation} "
                    f"Excerpt: \"{primary_refuter.relevant_excerpt[:200]}\"{weak_note}"
                ).strip(),
                qualifying_evidence=relevant_items,
            )

        # Rule 5: Fully verified by credible evidence without authoritative contradiction
        if supporting_items and not auth_refuting_items:
            # Evaluate distinct publisher lineages and independent clusters to prevent counting repeated/syndicated pages as independent
            has_primary_institutional = any(
                it.source_tier == 1 or it.record.source_type in ("primary_source", "government", "official", "academic")
                for it in supporting_items
            )
            primary_supporter = supporting_items[0]

            # True independent corroboration requires distinct independent clusters
            supporting_publishers = {it.record.publisher or "Unknown Source" for it in supporting_items}
            if analysis.independent_clusters_count > 0:
                independent_clusters = min(analysis.independent_clusters_count, len(supporting_publishers))
            else:
                independent_clusters = len(supporting_publishers)

            weak_note = (
                f" Note: {analysis.weak_counterclaims_note}"
                if analysis.has_weak_counterclaims and analysis.weak_counterclaims_note
                else ""
            )
            lineage_note = f" ({analysis.lineage_notes})" if analysis.lineage_notes else ""

            if has_primary_institutional:
                primary_inst = next(
                    (
                        it for it in supporting_items
                        if it.source_tier == 1 or it.record.source_type in ("primary_source", "government", "official", "academic")
                    ),
                    primary_supporter,
                )
                logger.info(
                    "Sufficiency Gate [Rule 5]: Verified via primary institutional documentation from '%s'",
                    primary_inst.record.publisher,
                )
                return ClaimVerdictEvaluation(
                    verdict=VerdictStatus.VERIFIED,
                    confidence_score=0.95,
                    explanation=(
                        f"Verified based on official primary institutional documentation from '{primary_inst.record.publisher}'. "
                        f"Corroborating excerpt: \"{primary_inst.relevant_excerpt[:250]}\"{lineage_note}.{weak_note}"
                    ).strip(),
                    qualifying_evidence=relevant_items,
                )
            elif independent_clusters >= 2:
                logger.info(
                    "Sufficiency Gate [Rule 5]: Verified via independent corroboration across %d distinct publisher lineages",
                    independent_clusters,
                )
                return ClaimVerdictEvaluation(
                    verdict=VerdictStatus.VERIFIED,
                    confidence_score=0.90,
                    explanation=(
                        f"Verified with independent corroboration across {independent_clusters} distinct publisher lineages "
                        f"(including '{primary_supporter.record.publisher}'). "
                        f"Corroborating excerpt: \"{primary_supporter.relevant_excerpt[:250]}\"{lineage_note}.{weak_note}"
                    ).strip(),
                    qualifying_evidence=relevant_items,
                )
            elif neutral_or_inconclusive:
                logger.info("Sufficiency Gate [Rule 5b]: Partial support with uncorroborated elements -> partly_supported.")
                return ClaimVerdictEvaluation(
                    verdict=VerdictStatus.PARTLY_SUPPORTED,
                    confidence_score=0.75,
                    explanation=(
                        f"Partially supported: core elements are substantiated by '{primary_supporter.record.publisher}', "
                        f"but additional assertions in the statement remain unverified or lack conclusive corroboration."
                    ),
                    qualifying_evidence=relevant_items,
                )
            else:
                # Multiple pages repeating the same underlying source do not count as independent corroboration
                logger.info("Sufficiency Gate [Rule 5c]: Single secondary publisher lineage or syndicated sources without independent corroboration -> partly_supported.")
                return ClaimVerdictEvaluation(
                    verdict=VerdictStatus.PARTLY_SUPPORTED,
                    confidence_score=0.70,
                    explanation=(
                        f"Reported by secondary source '{primary_supporter.record.publisher}'. "
                        f"Multiple pages repeating the same underlying source or syndicated wire copies do not count as independent "
                        f"corroboration; independent multi-source corroboration or official institutional sources are required for full verification."
                    ),
                    qualifying_evidence=relevant_items,
                )

        # Rule 7: Ambiguous or weak evidence -> cannot_confirm
        logger.info(
            "Final Reason for CANNOT_CONFIRM [Rule 7]: Available evidence is ambiguous or inconclusive (supporting=%d, auth_refuting=%d, neutral=%d) and does not satisfy sufficiency threshold.",
            len(supporting_items),
            len(auth_refuting_items),
            len(neutral_or_inconclusive),
        )
        return ClaimVerdictEvaluation(
            verdict=VerdictStatus.CANNOT_CONFIRM,
            confidence_score=0.2,
            explanation=(
                "Available source evidence is neutral or inconclusive and does not satisfy the "
                "rigorous threshold required for confirmation."
            ),
            qualifying_evidence=relevant_items,
        )

    def evaluate_report_verdict(
        self, claim_evaluations: List[ClaimVerdictEvaluation]
    ) -> ReportVerdictEvaluation:
        """Aggregate report-level verdict and summary from individual claim evaluations."""
        if not claim_evaluations:
            return ReportVerdictEvaluation(
                overall_verdict=VerdictStatus.CANNOT_CONFIRM,
                summary="No factual claims were identified in the input message.",
            )

        verdicts = [c.verdict for c in claim_evaluations]

        # All identical
        if all(v == verdicts[0] for v in verdicts):
            primary = verdicts[0]
            if primary == VerdictStatus.VERIFIED:
                summary = "All extracted claims have been verified against authoritative evidence."
            elif primary == VerdictStatus.FALSE:
                summary = "The claims are contradicted by authoritative evidence."
            elif primary == VerdictStatus.OUTDATED:
                summary = "The claims reflect outdated or expired information."
            elif primary == VerdictStatus.PARTLY_SUPPORTED:
                summary = "The claims are partially supported but lack complete substantiation."
            else:
                summary = "Insufficient authoritative evidence was discovered to confirm the claims."
            return ReportVerdictEvaluation(overall_verdict=primary, summary=summary)

        # Mixed verdicts
        if VerdictStatus.FALSE in verdicts:
            if VerdictStatus.VERIFIED in verdicts:
                return ReportVerdictEvaluation(
                    overall_verdict=VerdictStatus.PARTLY_SUPPORTED,
                    summary="Mixed verification: some claims are verified while others are contradicted by evidence.",
                )
            return ReportVerdictEvaluation(
                overall_verdict=VerdictStatus.FALSE,
                summary="One or more central claims are contradicted by authoritative evidence.",
            )

        if VerdictStatus.VERIFIED in verdicts:
            return ReportVerdictEvaluation(
                overall_verdict=VerdictStatus.PARTLY_SUPPORTED,
                summary="Some claims are supported by authoritative evidence, while others cannot be confirmed.",
            )

        if VerdictStatus.OUTDATED in verdicts:
            return ReportVerdictEvaluation(
                overall_verdict=VerdictStatus.OUTDATED,
                summary="Information in the message has been identified as outdated.",
            )

        return ReportVerdictEvaluation(
            overall_verdict=VerdictStatus.CANNOT_CONFIRM,
            summary="Evidence is insufficient across claims to confirm the message.",
        )
