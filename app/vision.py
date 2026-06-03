"""Heavy tier — image/video deepfake & face/scene authenticity (ENSEMBLE).

Runs an ensemble of open deepfake-image classifiers (configured via
VIGISCAM_AI_DEEPFAKE_IMAGE_MODELS, default = dima806 + prithivMLmods) and
aggregates their verdicts: the ensemble fake-probability is the MAX across
models, so if any one detector confidently flags a fake the verdict reflects
it (sensitivity over a single model's blind spot). Each model's individual
vote is returned for explainability.

Used for LIVE_FACE_SEAL, SCENE_SEAL, and ANTI_FAKE_VIDEO (one sampled frame).
Model ids are configuration — point the env var at a single fine-tuned
checkpoint (local path or HF id) once trained (see training/README.md).

Lazy-loaded and fully graceful: a model that fails to load is skipped; if no
model loads or inference fails, returns None → the check is INCONCLUSIVE.
"""
from __future__ import annotations

import io
import logging
import threading

from .config import get_settings

logger = logging.getLogger("vigiscam-ai.vision")

_FAKE_TOKENS = ("fake", "deepfake", "spoof", "manipulat", "synthetic", "generated", "ai")
_REAL_TOKENS = ("real", "bonafide", "genuine", "authentic", "live")

# model_id -> pipeline (or False if it failed to load)
_pipelines: dict[str, object] = {}
_lock = threading.Lock()


def _label_kind(label: str) -> str | None:
    low = label.lower()
    if any(t in low for t in _FAKE_TOKENS):
        return "fake"
    if any(t in low for t in _REAL_TOKENS):
        return "real"
    return None


def _get_pipeline(model_id: str):
    cached = _pipelines.get(model_id)
    if cached is not None:
        return cached or None
    with _lock:
        if model_id not in _pipelines:
            try:
                from transformers import pipeline

                logger.info("Loading deepfake image model %s", model_id)
                _pipelines[model_id] = pipeline("image-classification", model=model_id, device=-1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load image model %s: %s", model_id, exc)
                _pipelines[model_id] = False  # remember the failure, don't retry
    return _pipelines.get(model_id) or None


def _score_one(pipe, img) -> dict | None:
    try:
        preds = pipe(img)  # [{label, score}, ...]
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
            "labels_recognized": (fake_prob + real_prob) > 0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image inference failed: %s", exc)
        return None


def analyze_image(image_bytes: bytes) -> dict | None:
    """Run the ensemble. Returns aggregated verdict + per-model votes, or None."""
    models = get_settings().image_model_list
    if not models:
        return None
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not decode image: %s", exc)
        return None

    per_model: dict[str, dict] = {}
    for model_id in models:
        pipe = _get_pipeline(model_id)
        if pipe is None:
            continue
        res = _score_one(pipe, img)
        if res and res["labels_recognized"]:
            per_model[model_id] = res

    if not per_model:
        return None

    # Aggregate: MAX fake-probability across models (sensitivity), and report
    # the dominant label of the model that drove the verdict.
    fake_probs = {m: r["fake_prob"] for m, r in per_model.items()}
    driver = max(fake_probs, key=fake_probs.get)
    agg_fake = per_model[driver]["fake_prob"]
    agg_real = max((r["real_prob"] for r in per_model.values()), default=0.0)
    # If fake didn't dominate anywhere, real_prob should win.
    if agg_fake <= 0:
        agg_real = max(r["real_prob"] for r in per_model.values())

    return {
        "fake_prob": agg_fake,
        "real_prob": agg_real if agg_fake <= agg_real else (1.0 - agg_fake),
        "top_label": per_model[driver]["top_label"],
        "top_score": per_model[driver]["top_score"],
        "model": "+".join(per_model.keys()),
        "labels_recognized": True,
        "ensemble": {m: round(r["fake_prob"], 3) for m, r in per_model.items()},
    }
