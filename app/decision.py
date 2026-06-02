"""The decision envelope — the audit record attached to every AI verdict.

The user's requirement: *every AI decision logs the model used, model version,
confidence score, reason codes, risk score, and evidence reference.* This
dataclass is that record. Routers attach it to every response under a
`decision` key; the backend persists it on the AIDecision row (the output
JSON is stored verbatim), so each verdict is reproducible and explainable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


class Tier:
    """Which layer of the hybrid pipeline produced (or dominated) the verdict."""

    RULES = "RULES"  # tier 2 — rule-based obvious-signal detection
    SELF_HOSTED = "SELF_HOSTED"  # tier 1 — open model inference
    VECTOR = "VECTOR"  # tier 3 — embedding similarity search
    EXTERNAL = "EXTERNAL"  # tier 4 — external API fallback
    HUMAN_REVIEW = "HUMAN_REVIEW"  # tier 5 — flagged for a reviewer


@dataclass
class Decision:
    model_used: str
    model_version: str
    confidence: float  # 0–100
    risk_score: float  # 0–100
    reason_codes: list[str] = field(default_factory=list)
    tier: str = Tier.SELF_HOSTED
    requires_human_review: bool = False
    evidence_ref: str | None = None

    def as_dict(self) -> dict:
        # camelCase keys to match the backend's JSON conventions.
        d = asdict(self)
        return {
            "modelUsed": d["model_used"],
            "modelVersion": d["model_version"],
            "confidence": round(d["confidence"], 1),
            "riskScore": round(d["risk_score"], 1),
            "reasonCodes": d["reason_codes"],
            "tier": d["tier"],
            "requiresHumanReview": d["requires_human_review"],
            "evidenceRef": d["evidence_ref"],
        }
