"""POST /insights/{journey,victim-state,predicted-move} — real semantic
classification of the Phase 6E insight engines.

Journey stage and victim state are scored by embedding the transcript/notes
and matching against stage/state prototypes (zero-shot). Predicted next move
is a state-machine over the inferred journey stage (same contract the backend
stub used, but driven by the real stage inference).
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from ..labels import JOURNEY_STAGES, NEXT_MOVE_MAP, VICTIM_STATES
from ..models import PrototypeClassifier
from ..schemas import (
    InsightHints,
    JourneyResult,
    PredictedMoveInput,
    PredictedMoveResult,
    VictimStateResult,
)

router = APIRouter()

JOURNEY_VERSION = "fraud-journey-minilm-1.0.0"
VICTIM_VERSION = "victim-state-minilm-1.0.0"
PREDICT_VERSION = "predicted-move-minilm-1.0.0"

MIN_SIM = 0.25  # below this we fall back to the neutral default stage/state

_journey_clf = PrototypeClassifier(JOURNEY_STAGES)
_victim_clf = PrototypeClassifier(VICTIM_STATES)


def _text(hints: InsightHints) -> str:
    return f"{hints.transcript or ''} {hints.notes or ''}".strip()


@router.post("/insights/journey", response_model=JourneyResult)
def journey(hints: InsightHints) -> JourneyResult:
    if hints.forceStage:
        return JourneyResult(
            stage=hints.forceStage,
            confidence=80.0,
            modelVersion=JOURNEY_VERSION,
            evidence={"matchedKeywords": ["forced"]},
        )
    text = _text(hints)
    if not text:
        return JourneyResult(
            stage="INITIAL_CONTACT",
            confidence=35.0,
            modelVersion=JOURNEY_VERSION,
            evidence={"matchedKeywords": []},
        )
    scores = _journey_clf.score(text)
    stage = max(scores, key=scores.get)
    sim = scores[stage]
    if sim < MIN_SIM:
        stage = "INITIAL_CONTACT"
    confidence = float(np.clip(sim, 0, 1) * 100)
    return JourneyResult(
        stage=stage,
        confidence=round(confidence, 1),
        modelVersion=JOURNEY_VERSION,
        evidence={"topScores": {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]}},
    )


@router.post("/insights/victim-state", response_model=VictimStateResult)
def victim_state(hints: InsightHints) -> VictimStateResult:
    if hints.forceState:
        return VictimStateResult(
            state=hints.forceState,
            confidence=80.0,
            modelVersion=VICTIM_VERSION,
            signals={"matchedKeywords": ["forced"]},
        )
    text = _text(hints)
    if not text:
        return VictimStateResult(
            state="CALM",
            confidence=40.0,
            modelVersion=VICTIM_VERSION,
            signals={"matchedKeywords": []},
        )
    scores = _victim_clf.score(text)
    state = max(scores, key=scores.get)
    sim = scores[state]
    if sim < MIN_SIM:
        state = "CALM"
    confidence = float(np.clip(sim, 0, 1) * 100)
    return VictimStateResult(
        state=state,
        confidence=round(confidence, 1),
        modelVersion=VICTIM_VERSION,
        signals={"topScores": {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]}},
    )


@router.post("/insights/predicted-move", response_model=PredictedMoveResult)
def predicted_move(body: PredictedMoveInput) -> PredictedMoveResult:
    action = NEXT_MOVE_MAP.get(body.currentStage, "REQUEST_PERSONAL_INFO")
    return PredictedMoveResult(
        action=action,
        confidence=40.0 if action == "DROP_OFF" else 55.0,
        modelVersion=PREDICT_VERSION,
        rationale=f"Stage {body.currentStage} typically progresses to {action}",
    )
