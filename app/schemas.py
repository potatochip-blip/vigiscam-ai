"""Pydantic request/response models — EXACT mirrors of the backend's AI
client contracts (vigiscam-backend/src/modules/**/*.types.ts).

If a field name or enum value here drifts from the backend, the integration
silently breaks, so these are kept 1:1 with the TypeScript interfaces.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── NLP classification — POST /nlp/classify ──────────────────────────────────


class NlpClassificationInput(BaseModel):
    text: str
    indicatorType: str | None = None
    hintedCategory: str | None = None
    # Caller hint: route enterprise / high-risk cases to the external tier.
    highRisk: bool = False
    # Optional backend evidence id, echoed into the decision envelope.
    evidenceRef: str | None = None


class NlpClassificationOutput(BaseModel):
    category: str | None
    categoryConfidence: float  # 0–100
    scamScore: float  # 0–100
    manipulationTactics: list[str]
    modelVersion: str
    # Hybrid-pipeline audit envelope (model, version, confidence, reason codes,
    # risk score, evidence ref, tier, requiresHumanReview). The backend stores
    # this verbatim on the AIDecision row.
    decision: dict[str, Any] | None = None


# ── Embeddings — POST /embeddings ────────────────────────────────────────────


class EmbeddingInput(BaseModel):
    text: str


class EmbeddingOutput(BaseModel):
    vector: list[float]
    modelVersion: str


# ── Insights — POST /insights/{journey,victim-state,predicted-move} ──────────


class InsightHints(BaseModel):
    transcript: str | None = None
    notes: str | None = None
    forceStage: str | None = None
    forceState: str | None = None


class JourneyResult(BaseModel):
    stage: str
    confidence: float
    modelVersion: str
    evidence: dict[str, Any]


class VictimStateResult(BaseModel):
    state: str
    confidence: float
    modelVersion: str
    signals: dict[str, Any]


class PredictedMoveInput(BaseModel):
    currentStage: str


class PredictedMoveResult(BaseModel):
    action: str
    confidence: float
    modelVersion: str
    rationale: str


# ── OSINT enrichment — POST /osint/{indicatorType} ───────────────────────────


class OsintEnrichmentInput(BaseModel):
    indicatorType: str
    normalizedIndicator: str


class OsintEnrichmentOutput(BaseModel):
    provider: str
    modelVersion: str
    data: dict[str, Any]
    riskHints: list[str]


# ── Authenticity — POST /authenticity/{checkType} ────────────────────────────


class AuthenticityRequest(BaseModel):
    checkType: str
    sessionId: str
    payload: dict[str, Any] | None = None


class AuthenticityResponse(BaseModel):
    result: str  # PASS | FAIL | INCONCLUSIVE
    score: float  # 0–100
    modelVersion: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Hybrid-pipeline audit envelope, persisted verbatim by the backend.
    decision: dict[str, Any] | None = None
