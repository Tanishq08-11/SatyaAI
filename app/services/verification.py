"""TruthGraph verification engine: orchestrates claim extraction, evidence retrieval, analysis, sufficiency gate, and DB persistence."""

import logging
from typing import Dict, List, Optional
import uuid
from sqlalchemy.orm import Session, selectinload

from app.db.models import Claim, ClaimEvidence, Report, Source
from app.integrations.url_normalizer import extract_domain, normalize_url
from app.schemas.common import VerdictStatus
from app.schemas.verify import VerifyRequest
from app.services.claim_extractor import ClaimExtractor
from app.services.evidence_analyzer import EvidenceAnalyzer
from app.services.evidence_retriever import EvidenceRetriever
from app.services.verdict_engine import ClaimVerdictEvaluation, VerdictEngine

logger = logging.getLogger(__name__)


class VerificationEngine:
    """Orchestrates the complete evidence-first verification pipeline."""

    def __init__(
        self,
        extractor: Optional[ClaimExtractor] = None,
        retriever: Optional[EvidenceRetriever] = None,
        analyzer: Optional[EvidenceAnalyzer] = None,
        verdict_engine: Optional[VerdictEngine] = None,
    ):
        self.extractor = extractor or ClaimExtractor()
        self.retriever = retriever or EvidenceRetriever()
        self.analyzer = analyzer or EvidenceAnalyzer()
        self.verdict_engine = verdict_engine or VerdictEngine()

    async def verify(self, db: Session, request: VerifyRequest) -> Report:
        """Execute the end-to-end verification pipeline."""
        corr_id = request.correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
        msg_id = request.message_id or "unknown"
        logger.info(
            "Initiating TruthGraph verification run: correlation_id=%s, message_id=%s, input_len=%d, lang=%s, preview='%s'",
            corr_id,
            msg_id,
            len(request.text),
            request.language,
            request.text[:80].replace("\n", " "),
        )

        try:
            # 1. Extract claims
            extraction_result = await self.extractor.extract_claims(request.text)
            claims_to_evaluate = extraction_result.claims

            if not claims_to_evaluate:
                logger.info(
                    "No factual claims extracted: correlation_id=%s, msg_id=%s",
                    corr_id,
                    msg_id,
                )
                report = Report(
                    correlation_id=corr_id,
                    message_id=request.message_id,
                    input_text=request.text,
                    language=request.language,
                    overall_verdict=VerdictStatus.CANNOT_CONFIRM.value,
                    summary=extraction_result.notes or "No verifiable factual claims detected in message.",
                )
                db.add(report)
                db.commit()
                db.refresh(report)
                return report

            logger.info(
                "Extracted %d claim(s) for verification: correlation_id=%s, claims=%s",
                len(claims_to_evaluate),
                corr_id,
                [c.text[:60] for c in claims_to_evaluate],
            )

            # 2. Retrieve, analyze, and evaluate each claim
            claim_evaluations: List[ClaimVerdictEvaluation] = []
            analyzed_evidence_map: Dict[str, List] = {}

            for extracted_claim in claims_to_evaluate:
                # Retrieve evidence
                retrieval = await self.retriever.retrieve_evidence(extracted_claim.text)

                # Analyze evidence relationships
                analysis = await self.analyzer.analyze_claim_evidence(
                    extracted_claim.text, retrieval.records
                )

                # Apply sufficiency gate
                evaluation = self.verdict_engine.evaluate_claim_verdict(analysis)
                claim_evaluations.append(evaluation)
                analyzed_evidence_map[extracted_claim.id] = evaluation.qualifying_evidence

                logger.info(
                    "Claim Evaluated: correlation_id=%s, claim='%s', verdict=%s, confidence=%.2f, qualifying_ev=%d",
                    corr_id,
                    extracted_claim.text[:60],
                    evaluation.verdict.value,
                    evaluation.confidence_score,
                    len(evaluation.qualifying_evidence),
                )

            # 3. Evaluate aggregate report verdict
            report_evaluation = self.verdict_engine.evaluate_report_verdict(claim_evaluations)

            # 4. Persist to Database (Report -> Claims -> Sources -> ClaimEvidence)
            report = Report(
                correlation_id=corr_id,
                message_id=request.message_id,
                input_text=request.text,
                language=request.language,
                overall_verdict=report_evaluation.overall_verdict.value,
                summary=report_evaluation.summary,
            )
            db.add(report)
            db.flush()

            # Cache sources by canonical URL in this session to deduplicate
            source_cache: Dict[str, Source] = {}

            total_evidence_links = 0
            for extracted_claim, eval_result in zip(claims_to_evaluate, claim_evaluations):
                claim_orm = Claim(
                    report_id=report.id,
                    claim_text=extracted_claim.text,
                    verdict=eval_result.verdict.value,
                    confidence_score=eval_result.confidence_score,
                    explanation=eval_result.explanation,
                )
                db.add(claim_orm)
                db.flush()

                # Attach qualifying evidence
                for ev_item in eval_result.qualifying_evidence:
                    norm_url = normalize_url(ev_item.record.canonical_url)
                    if not norm_url:
                        continue

                    # Lookup existing source or create new
                    if norm_url in source_cache:
                        source_orm = source_cache[norm_url]
                    else:
                        existing_source = (
                            db.query(Source).filter(Source.url == norm_url).first()
                        )
                        if existing_source:
                            source_orm = existing_source
                            source_orm.source_type = ev_item.record.source_type
                            source_orm.reliability_score = ev_item.record.reliability_score
                            db.flush()
                        else:
                            source_orm = Source(
                                url=norm_url,
                                domain=extract_domain(norm_url),
                                title=ev_item.record.title,
                                publisher=ev_item.record.publisher,
                                published_date=ev_item.record.published_date,
                                source_type=ev_item.record.source_type,
                                retrieval_method=ev_item.record.retrieval_method,
                                reliability_score=ev_item.record.reliability_score,
                            )
                            db.add(source_orm)
                            db.flush()
                        source_cache[norm_url] = source_orm

                    # Create ClaimEvidence link
                    claim_evidence_orm = ClaimEvidence(
                        claim_id=claim_orm.id,
                        source_id=source_orm.id,
                        snippet=ev_item.relevant_excerpt,
                        stance=ev_item.stance.value,
                        explanation=ev_item.explanation,
                        relevance_score=ev_item.relevance_score,
                    )
                    db.add(claim_evidence_orm)
                    total_evidence_links += 1

            db.commit()
            logger.info("VerificationEngine: Persisted %d claim(s) and %d evidence link(s) for report %s", len(claims_to_evaluate), total_evidence_links, report.id)

            # 5. Query back with complete eager loading
            saved_report = (
                db.query(Report)
                .options(
                    selectinload(Report.claims)
                    .selectinload(Claim.claim_evidence)
                    .selectinload(ClaimEvidence.source)
                )
                .filter(Report.id == report.id)
                .first()
            )

            logger.info(
                f"TruthGraph verification completed: verdict={report.overall_verdict}, "
                f"claims={len(claims_to_evaluate)}, evidence_links={total_evidence_links}"
            )
            return saved_report or report

        except Exception as exc:
            db.rollback()
            logger.error(f"Error during verification pipeline: {exc}", exc_info=True)
            # Safe truthful fallback: cannot_confirm with explicit error explanation
            fallback_report = Report(
                correlation_id=request.correlation_id,
                message_id=request.message_id,
                input_text=request.text,
                language=request.language,
                overall_verdict=VerdictStatus.CANNOT_CONFIRM.value,
                summary=(
                    "An error occurred during automated evidence processing. In accordance with SATYA AI's "
                    "evidence integrity principles, no verdict is fabricated and status is set to cannot_confirm."
                ),
            )
            db.add(fallback_report)
            db.flush()
            fallback_claim = Claim(
                report_id=fallback_report.id,
                claim_text=request.text,
                verdict=VerdictStatus.CANNOT_CONFIRM.value,
                confidence_score=0.0,
                explanation="Pipeline encountered an execution issue during retrieval/analysis. Marked cannot_confirm.",
            )
            db.add(fallback_claim)
            db.commit()

            return (
                db.query(Report)
                .options(
                    selectinload(Report.claims).selectinload(Claim.claim_evidence)
                )
                .filter(Report.id == fallback_report.id)
                .first()
                or fallback_report
            )


# Default singleton verification service
default_engine = VerificationEngine()


async def verify_text_claims(db: Session, request: VerifyRequest) -> Report:
    """Entry point for claim verification used by the API router."""
    return await default_engine.verify(db=db, request=request)
