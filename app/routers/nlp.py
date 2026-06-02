"""POST /nlp/classify — hybrid scam classification.

Runs the full pipeline (rules + self-hosted model + vector similarity +
optional external fallback + human-review flag) and returns the backend
contract fields plus the `decision` audit envelope.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..pipeline import classify_nlp
from ..schemas import NlpClassificationInput, NlpClassificationOutput

router = APIRouter()


@router.post("/nlp/classify", response_model=NlpClassificationOutput)
def classify(body: NlpClassificationInput) -> NlpClassificationOutput:
    out, decision = classify_nlp(
        text=body.text,
        hinted_category=body.hintedCategory,
        high_risk=body.highRisk,
        evidence_ref=body.evidenceRef,
    )
    return NlpClassificationOutput(**out, decision=decision.as_dict())
