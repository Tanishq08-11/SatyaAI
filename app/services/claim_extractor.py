"""Claim extraction service: isolates factual claims from user messages using LLM or structured fallback."""

import logging
import re
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from app.integrations.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ExtractedClaimItem(BaseModel):
    """An individual extracted claim proposition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=3, description="Self-contained claim statement")
    is_factual: bool = Field(
        default=True,
        description="True if statement asserts a verifiable factual proposition rather than opinion/question",
    )
    category: str = Field(
        default="factual_claim",
        description="Classification: factual_claim, opinion, question, instruction, greeting",
    )


class ClaimExtractionResult(BaseModel):
    """Validated result containing extracted claims and extraction provenance."""

    claims: List[ExtractedClaimItem] = Field(default_factory=list)
    extraction_method: str = Field(
        ..., description="Method used: 'llm_extraction' or 'single_claim_fallback'"
    )
    is_fallback: bool = Field(
        default=False,
        description="Whether a fallback was used instead of configured multi-claim LLM segmentation",
    )
    notes: Optional[str] = None


SYSTEM_EXTRACTION_PROMPT = """You are an objective claim extraction engine for a fact-checking pipeline.
Analyze the input text and extract all distinct, verifiable FACTUAL claims.

Rules:
1. Extract only verifiable statements of fact (e.g. statistics, events, government policies, scientific claims, dates).
2. DO NOT treat opinions, feelings, questions, greetings, or instructions as factual claims.
3. If an input contains multiple factual statements, separate them into distinct self-contained claims.
4. Normalize pronouns so each extracted claim can be independently searched (e.g., replace 'he' with the named entity if obvious).
5. Output ONLY a valid JSON object matching this schema:
{
  "claims": [
    {
      "text": "Exact claim statement",
      "is_factual": true,
      "category": "factual_claim"
    }
  ]
}
If the text contains NO factual claims (e.g. purely 'Hello', 'What do you think?', 'Can you help me?'), return {"claims": []}.
Never fabricate claims or make judgements on whether the claim is true or false.
"""


class ClaimExtractor:
    """Service to extract factual claims from incoming messages."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def extract_claims(self, text: str) -> ClaimExtractionResult:
        """
        Extract factual claims from text. Uses LLM if available; otherwise falls back
        to single-claim intake with explicit fallback labeling.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return ClaimExtractionResult(
                claims=[],
                extraction_method="single_claim_fallback",
                is_fallback=True,
                notes="Input text was empty.",
            )

        if self.llm_client.is_available():
            try:
                user_prompt = f"Extract all verifiable factual claims from this message:\n\"\"\"\n{cleaned_text}\n\"\"\""
                raw_json = await self.llm_client.generate_json(
                    system_prompt=SYSTEM_EXTRACTION_PROMPT,
                    user_prompt=user_prompt,
                )

                if raw_json and isinstance(raw_json, dict) and "claims" in raw_json:
                    items: List[ExtractedClaimItem] = []
                    for idx, c in enumerate(raw_json["claims"]):
                        claim_text = c.get("text", "").strip() if isinstance(c, dict) else str(c).strip()
                        is_factual = c.get("is_factual", True) if isinstance(c, dict) else True
                        category = c.get("category", "factual_claim") if isinstance(c, dict) else "factual_claim"

                        if claim_text and is_factual and category == "factual_claim":
                            items.append(
                                ExtractedClaimItem(
                                    id=str(uuid.uuid4()),
                                    text=claim_text,
                                    is_factual=True,
                                    category=category,
                                )
                            )

                    logger.info(f"LLM extracted {len(items)} factual claims from input.")
                    return ClaimExtractionResult(
                        claims=items,
                        extraction_method="llm_extraction",
                        is_fallback=False,
                        notes="Multi-claim LLM extraction executed successfully.",
                    )
                else:
                    logger.warning("LLM response was empty or malformed; using single-claim fallback.")
            except Exception as exc:
                logger.warning(f"Error during LLM claim extraction: {exc}; using single-claim fallback.")

        # Single-claim fallback
        return self._single_claim_fallback(cleaned_text)

    def _single_claim_fallback(self, text: str) -> ClaimExtractionResult:
        """
        Deterministic single-claim fallback when no LLM is configured or available.
        Clearly labels that multi-claim extraction was not performed.
        """
        # Detect pure greetings or questions
        is_greeting = bool(
            re.match(r"^(hi|hello|hey|namaste|good morning|good evening|test)\b", text, re.IGNORECASE)
            and len(text.split()) <= 4
        )
        is_pure_question = text.endswith("?") and not any(
            kw in text.lower() for kw in ["breaking", "announced", "banned", "killed", "confirmed", "cure"]
        )

        if is_greeting or (is_pure_question and len(text.split()) < 6):
            return ClaimExtractionResult(
                claims=[],
                extraction_method="single_claim_fallback",
                is_fallback=True,
                notes="Single-claim fallback identified input as greeting, conversational remark, or non-assertive question.",
            )

        claim_item = ExtractedClaimItem(
            id=str(uuid.uuid4()),
            text=text,
            is_factual=True,
            category="factual_claim",
        )
        return ClaimExtractionResult(
            claims=[claim_item],
            extraction_method="single_claim_fallback",
            is_fallback=True,
            notes="Single-claim fallback used (LLM provider not configured or unavailable). Multi-claim segmentation not performed.",
        )
