"""Tier 4 — optional external API fallback.

Self-hosted models are always tried first. Only when the self-hosted verdict
is low-confidence (or the caller marks the case high-risk / enterprise) does
the pipeline optionally escalate to an external LLM for a second opinion.

OFF by default and fully model-agnostic: any OpenAI-compatible chat endpoint
works (set VIGISCAM_AI_EXTERNAL_API_URL / _KEY / _MODEL). If disabled or the
call fails, the pipeline keeps the self-hosted verdict — the external tier is
never a hard dependency.
"""
from __future__ import annotations

import json
import logging

from .config import get_settings

logger = logging.getLogger("vigiscam-ai.external")


def should_escalate(confidence: float, high_risk: bool) -> bool:
    s = get_settings()
    if not s.external_api_enabled:
        return False
    return high_risk or confidence < s.external_escalation_threshold


def external_classify(text: str) -> dict | None:
    """Ask the external model for a category + scam likelihood. Returns
    {"category", "scamScore", "reason"} or None on any failure/disabled."""
    s = get_settings()
    if not (s.external_api_enabled and s.external_api_url and s.external_api_key and s.external_api_model):
        return None
    try:
        import urllib.request

        prompt = (
            "You are a scam-detection classifier. Given a message, respond with "
            "ONLY compact JSON: {\"category\": <one of BANK_IMPERSONATION, "
            "GIFT_CARD_SCAM, REMOTE_ACCESS_SCAM, TECH_SUPPORT_SCAM, "
            "GOVERNMENT_IMPERSONATION, CRYPTO_SCAM, ROMANCE_SCAM, or null>, "
            "\"scamScore\": <0-100>, \"reason\": <short string>}.\n\nMessage: "
            + text
        )
        body = json.dumps(
            {
                "model": s.external_api_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            s.external_api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {s.external_api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 — operator-configured URL
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {
            "category": parsed.get("category"),
            "scamScore": float(parsed.get("scamScore", 0)),
            "reason": str(parsed.get("reason", "")),
        }
    except Exception as exc:  # noqa: BLE001 — never let the fallback break the pipeline
        logger.warning("External classify failed: %s", exc)
        return None
