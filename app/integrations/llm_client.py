"""Lightweight, resilient LLM client supporting configurable providers with JSON formatting."""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for querying LLMs for structured claim extraction and evidence analysis."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def is_available(self) -> bool:
        """Check if an LLM provider and credentials are configured."""
        provider = (self.settings.LLM_PROVIDER or "").lower()
        return bool(provider != "none" and self.settings.LLM_API_KEY)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Request structured JSON from the configured LLM.
        Returns parsed dictionary or fallback_json on error/timeout/malformed output.
        """
        if not self.is_available():
            logger.info("LLM provider not configured or credentials missing; skipping LLM call.")
            return fallback_json

        provider = self.settings.LLM_PROVIDER.lower()
        timeout = self.settings.RETRIEVAL_TIMEOUT_SECONDS

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if provider in ("openai", "generic"):
                    return await self._call_openai_compatible(client, system_prompt, user_prompt)
                elif provider == "gemini":
                    return await self._call_gemini(client, system_prompt, user_prompt)
                elif provider == "anthropic":
                    return await self._call_anthropic(client, system_prompt, user_prompt)
                else:
                    logger.warning(f"Unsupported LLM provider '{provider}', falling back.")
                    return fallback_json
        except httpx.TimeoutException:
            logger.warning(f"LLM request timed out after {timeout}s.")
            return fallback_json
        except Exception as exc:
            logger.warning(f"LLM request failed: {exc}")
            return fallback_json

    async def _call_openai_compatible(
        self, client: httpx.AsyncClient, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        base_url = self.settings.LLM_BASE_URL or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._extract_json(content)

    async def _call_gemini(
        self, client: httpx.AsyncClient, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        model = self.settings.LLM_MODEL or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.settings.LLM_API_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0},
        }
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._extract_json(content)

    async def _call_anthropic(
        self, client: httpx.AsyncClient, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.settings.LLM_API_KEY or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.LLM_MODEL or "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0,
        }
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["content"][0]["text"]
        return self._extract_json(content)

    def _extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON from raw text or markdown code blocks."""
        text = raw_text.strip()
        # Strip markdown fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except Exception as exc:
            logger.warning(f"Failed to parse LLM JSON response: {exc}. Raw content: {raw_text[:200]}")
            return None
