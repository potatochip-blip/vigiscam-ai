"""Heavy tier — image/video deepfake & face/scene authenticity.

Model (open, self-hosted, swappable via VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL):
  default `dima806/deepfake_vs_real_image_detection` — a ViT-base classifier
  fine-tuned for real-vs-fake face/image detection, the most-downloaded
  general deepfake image classifier on the Hugging Face Hub, runs on CPU.

Used for LIVE_FACE_SEAL, SCENE_SEAL, and ANTI_FAKE_VIDEO (one sampled frame).
The model id is configuration, not hardcoded — swap it for a dedicated
face-anti-spoofing or forgery model without touching the contract.

Lazy-loaded on first use and fully graceful: any load/inference failure
returns None so the check degrades to INCONCLUSIVE instead of crashing.
"""
from __future__ import annotations

import io
import logging
import threading

from .config import get_settings

logger = logging.getLogger("vigiscam-ai.vision")

_FAKE_TOKENS = ("fake", "deepfake", "spoof", "manipulat", "synthetic", "generated", "ai")
_REAL_TOKENS = ("real", "bonafide", "genuine", "authentic", "live")

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
            model_id = get_settings().deepfake_image_model
            if not model_id:
                _load_failed = True
                return None
            try:
                from transformers import pipeline

                logger.info("Loading deepfake image model %s", model_id)
                _pipeline = pipeline("image-classification", model=model_id, device=-1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load image model %s: %s", model_id, exc)
                _load_failed = True
    return _pipeline


def analyze_image(image_bytes: bytes) -> dict | None:
    """Returns {fake_prob, top_label, top_score, model} or None on failure."""
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preds = pipe(img)  # [{label, score}, ...]
        fake_prob = 0.0
        real_prob = 0.0
        for p in preds:
            kind = _label_kind(str(p["label"]))
            if kind == "fake":
                fake_prob += float(p["score"])
            elif kind == "real":
                real_prob += float(p["score"])
        # If labels were unrecognized, normalize by assuming index 0 is the
        # model's positive class is unknowable → leave both 0 → INCONCLUSIVE.
        top = max(preds, key=lambda p: p["score"])
        return {
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "top_label": str(top["label"]),
            "top_score": float(top["score"]),
            "model": get_settings().deepfake_image_model,
            "labels_recognized": (fake_prob + real_prob) > 0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image inference failed: %s", exc)
        return None
