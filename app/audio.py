"""Heavy tier — synthetic / cloned voice detection (VOICE_MATCH_SEAL).

Model (open, self-hosted, swappable via VIGISCAM_AI_VOICE_SPOOF_MODEL):
  default `MelodyMachine/Deepfake-audio-detection-V2` — a wav2vec2-based
  audio classifier fine-tuned for real-vs-fake (spoofed/synthetic) speech,
  served through the transformers audio-classification pipeline.

The model id is configuration, not hardcoded — swap for any ASVspoof /
anti-spoofing wav2vec2 checkpoint without touching the contract. The label
mapping is token-based so "fake/real" and "spoof/bonafide" vocabularies both
work. Lazy-loaded and graceful (failure → None → INCONCLUSIVE).

Audio is decoded via a temp file so the pipeline's ffmpeg/soundfile backend
handles wav/mp3/m4a/ogg. (ffmpeg is installed in the image.)
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading

from .config import get_settings

logger = logging.getLogger("vigiscam-ai.audio")

_FAKE_TOKENS = ("fake", "spoof", "synthetic", "deepfake", "clone", "tts", "ai")
_REAL_TOKENS = ("real", "bonafide", "genuine", "human", "authentic")

_pipeline = None
_pipeline_lock = threading.Lock()
_load_failed = False


def _label_kind(label: str) -> str | None:
    low = label.lower()
    if any(t in low for t in _FAKE_TOKENS):
        return "fake"
    if any(t in low for t in _REAL_TOKENS):
        return "real"
    return None


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None and not _load_failed:
            model_id = get_settings().voice_spoof_model
            if not model_id:
                _load_failed = True
                return None
            try:
                from transformers import pipeline

                logger.info("Loading voice-spoof model %s", model_id)
                _pipeline = pipeline("audio-classification", model=model_id, device=-1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load audio model %s: %s", model_id, exc)
                _load_failed = True
    return _pipeline


def analyze_audio(audio_bytes: bytes) -> dict | None:
    """Returns {fake_prob, real_prob, top_label, top_score, model} or None."""
    pipe = _get_pipeline()
    if pipe is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        preds = pipe(tmp_path)  # [{label, score}, ...]
        fake_prob = 0.0
        real_prob = 0.0
        for p in preds:
            kind = _label_kind(str(p["label"]))
            if kind == "fake":
                fake_prob += float(p["score"])
            elif kind == "real":
                real_prob += float(p["score"])
        top = max(preds, key=lambda p: p["score"])
        return {
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "top_label": str(top["label"]),
            "top_score": float(top["score"]),
            "model": get_settings().voice_spoof_model,
            "labels_recognized": (fake_prob + real_prob) > 0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio inference failed: %s", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
