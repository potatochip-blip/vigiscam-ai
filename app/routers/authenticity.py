"""POST /authenticity/{checkType} — authenticity verification suite.

This is the honest tier boundary. Real face/voice/scene deepfake detection
requires heavy vision/audio models (and a GPU) plus the raw media — a major
sub-project of its own. Rather than fake a verdict, this service:

  - returns INCONCLUSIVE (low score) by default for the ML-heavy checks
    (LIVE_FACE_SEAL, VOICE_MATCH_SEAL, SCENE_SEAL, ANTI_FAKE_VIDEO) until a
    real model is configured (VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL etc.);
  - performs a genuine, lightweight check where the payload carries a usable
    deterministic signal (CAM_VIGUARD device-fingerprint consistency,
    DUAL_AUTH challenge match).

Wiring a real model: set the model env var and implement the corresponding
branch — the contract (result/score/modelVersion/metadata) stays identical,
so the backend needs no change.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings
from ..schemas import AuthenticityRequest, AuthenticityResponse

router = APIRouter()

MODEL_VERSION = "authenticity-baseline-1.0.0"

# Checks that need a heavy ML model + raw media to produce a real verdict.
ML_HEAVY = {"LIVE_FACE_SEAL", "VOICE_MATCH_SEAL", "SCENE_SEAL", "ANTI_FAKE_VIDEO"}


@router.post("/authenticity/{check_path}", response_model=AuthenticityResponse)
def authenticity(check_path: str, body: AuthenticityRequest) -> AuthenticityResponse:
    check = (body.checkType or check_path.replace("-", "_")).upper()
    payload = body.payload or {}
    settings = get_settings()

    # ── Deterministic, genuinely-checkable variants ──
    if check == "DUAL_AUTH":
        expected = payload.get("expectedChallenge")
        provided = payload.get("providedChallenge")
        if expected is not None and provided is not None:
            ok = str(expected) == str(provided)
            return AuthenticityResponse(
                result="PASS" if ok else "FAIL",
                score=95.0 if ok else 5.0,
                modelVersion=MODEL_VERSION,
                metadata={"check": check, "method": "challenge-match"},
            )

    if check == "CAM_VIGUARD":
        # Device-fingerprint consistency: PASS when the session's fingerprint
        # matches the enrolled one, FAIL on mismatch, INCONCLUSIVE if unknown.
        enrolled = payload.get("enrolledFingerprint")
        seen = payload.get("observedFingerprint")
        if enrolled is not None and seen is not None:
            ok = str(enrolled) == str(seen)
            return AuthenticityResponse(
                result="PASS" if ok else "FAIL",
                score=90.0 if ok else 10.0,
                modelVersion=MODEL_VERSION,
                metadata={"check": check, "method": "device-fingerprint"},
            )

    # ── ML-heavy variants: only a real verdict if a model is configured ──
    if check in ML_HEAVY:
        configured = (
            settings.deepfake_image_model
            if check in ("LIVE_FACE_SEAL", "SCENE_SEAL", "ANTI_FAKE_VIDEO")
            else settings.voice_spoof_model
        )
        if not configured:
            return AuthenticityResponse(
                result="INCONCLUSIVE",
                score=0.0,
                modelVersion=MODEL_VERSION,
                metadata={
                    "check": check,
                    "reason": "no deepfake/voice model configured; set the model env var to enable",
                },
            )
        # A configured model would run here; until one is wired we stay honest.
        return AuthenticityResponse(
            result="INCONCLUSIVE",
            score=0.0,
            modelVersion=MODEL_VERSION,
            metadata={"check": check, "reason": "model configured but inference not yet implemented"},
        )

    return AuthenticityResponse(
        result="INCONCLUSIVE",
        score=0.0,
        modelVersion=MODEL_VERSION,
        metadata={"check": check, "reason": "insufficient signal in payload"},
    )
