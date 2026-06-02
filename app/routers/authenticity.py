"""POST /authenticity/{checkType} — authenticity verification suite.

Routing:
  DUAL_AUTH, CAM_VIGUARD                         → deterministic challenge/
                                                   fingerprint checks (real)
  LIVE_FACE_SEAL, SCENE_SEAL, ANTI_FAKE_VIDEO    → deepfake image model (vision)
  VOICE_MATCH_SEAL                               → voice-spoof model (audio)

The heavy checks need media in the payload (imageBase64/imageUrl,
audioBase64/audioUrl). With no media or an unloadable model they return
INCONCLUSIVE rather than guessing. Every response carries the decision
envelope (model, version, confidence, reason codes, risk score, evidence ref).
"""
from __future__ import annotations

from fastapi import APIRouter

from ..audio import analyze_audio
from ..config import get_settings
from ..decision import Decision, Tier
from ..media import load_audio_bytes, load_image_bytes
from ..schemas import AuthenticityRequest, AuthenticityResponse
from ..vision import analyze_image

router = APIRouter()

BASE_VERSION = "authenticity-1.1.0"
IMAGE_CHECKS = {"LIVE_FACE_SEAL", "SCENE_SEAL", "ANTI_FAKE_VIDEO"}


def _envelope(model_used: str, confidence: float, risk: float, reasons: list[str], tier: str, ev: str | None) -> dict:
    requires_review = confidence < get_settings().review_confidence_threshold
    if requires_review:
        reasons = [*reasons, "LOW_CONFIDENCE_REVIEW"]
    return Decision(
        model_used=model_used,
        model_version=BASE_VERSION,
        confidence=confidence,
        risk_score=risk,
        reason_codes=reasons,
        tier=tier,
        requires_human_review=requires_review,
        evidence_ref=ev,
    ).as_dict()


def _media_verdict(check: str, res: dict | None, ev: str | None) -> AuthenticityResponse:
    """Map a vision/audio result into the response + decision envelope."""
    model_used = (res or {}).get("model") or "n/a"
    if res is None:
        return AuthenticityResponse(
            result="INCONCLUSIVE",
            score=0.0,
            modelVersion=BASE_VERSION,
            metadata={"check": check, "reason": "no media, model unavailable, or inference failed"},
            decision=_envelope(model_used, 0.0, 0.0, ["NO_VERDICT"], Tier.SELF_HOSTED, ev),
        )
    if not res.get("labels_recognized"):
        return AuthenticityResponse(
            result="INCONCLUSIVE",
            score=0.0,
            modelVersion=BASE_VERSION,
            metadata={"check": check, "topLabel": res.get("top_label"), "reason": "label mapping unrecognized"},
            decision=_envelope(model_used, 0.0, 0.0, ["UNMAPPED_LABELS"], Tier.SELF_HOSTED, ev),
        )

    fake_prob = float(res["fake_prob"])
    real_prob = float(res["real_prob"])
    total = fake_prob + real_prob or 1.0
    fake_norm = fake_prob / total
    is_fake = fake_prob > real_prob
    confidence = float(max(fake_prob, real_prob) / total * 100)
    risk = float(fake_norm * 100)
    reasons = [
        "DEEPFAKE_IMAGE_MODEL" if check in IMAGE_CHECKS else "VOICE_SPOOF_MODEL",
        f"TOP_LABEL:{res.get('top_label')}:{round(float(res.get('top_score', 0)), 3)}",
        "FAKE_DOMINANT" if is_fake else "REAL_DOMINANT",
    ]
    return AuthenticityResponse(
        result="FAIL" if is_fake else "PASS",  # FAIL = not authentic (likely fake)
        score=round(confidence, 1),
        modelVersion=BASE_VERSION,
        metadata={"check": check, "fakeProbability": round(fake_norm, 3), "topLabel": res.get("top_label")},
        decision=_envelope(model_used, confidence, risk, reasons, Tier.SELF_HOSTED, ev),
    )


@router.post("/authenticity/{check_path}", response_model=AuthenticityResponse)
def authenticity(check_path: str, body: AuthenticityRequest) -> AuthenticityResponse:
    check = (body.checkType or check_path.replace("-", "_")).upper()
    payload = body.payload or {}
    ev = str(payload.get("evidenceRef") or body.sessionId or "")

    # ── Deterministic, genuinely-checkable variants (tier 2 / rules) ──
    if check == "DUAL_AUTH":
        expected, provided = payload.get("expectedChallenge"), payload.get("providedChallenge")
        if expected is not None and provided is not None:
            ok = str(expected) == str(provided)
            return AuthenticityResponse(
                result="PASS" if ok else "FAIL",
                score=95.0 if ok else 5.0,
                modelVersion=BASE_VERSION,
                metadata={"check": check, "method": "challenge-match"},
                decision=_envelope("challenge-match", 95.0 if ok else 90.0, 5.0 if ok else 95.0,
                                   ["DUAL_AUTH_MATCH" if ok else "DUAL_AUTH_MISMATCH"], Tier.RULES, ev),
            )

    if check == "CAM_VIGUARD":
        enrolled, seen = payload.get("enrolledFingerprint"), payload.get("observedFingerprint")
        if enrolled is not None and seen is not None:
            ok = str(enrolled) == str(seen)
            return AuthenticityResponse(
                result="PASS" if ok else "FAIL",
                score=90.0 if ok else 10.0,
                modelVersion=BASE_VERSION,
                metadata={"check": check, "method": "device-fingerprint"},
                decision=_envelope("device-fingerprint", 90.0, 10.0 if ok else 90.0,
                                   ["FINGERPRINT_MATCH" if ok else "FINGERPRINT_MISMATCH"], Tier.RULES, ev),
            )

    # ── Heavy ML tier ──
    if check in IMAGE_CHECKS:
        img = load_image_bytes(payload)
        if img is None:
            return _media_verdict(check, None, ev)
        return _media_verdict(check, analyze_image(img), ev)

    if check == "VOICE_MATCH_SEAL":
        aud = load_audio_bytes(payload)
        if aud is None:
            return _media_verdict(check, None, ev)
        return _media_verdict(check, analyze_audio(aud), ev)

    # Unknown / insufficient signal.
    return AuthenticityResponse(
        result="INCONCLUSIVE",
        score=0.0,
        modelVersion=BASE_VERSION,
        metadata={"check": check, "reason": "insufficient signal in payload"},
        decision=_envelope("n/a", 0.0, 0.0, ["INSUFFICIENT_SIGNAL"], Tier.HUMAN_REVIEW, ev),
    )
