"""POST /nlp/classify — real semantic scam classification.

Replaces the backend keyword stub with embedding-based zero-shot scoring:
  - category   = best-matching scam category prototype (cosine similarity)
  - scamScore  = calibrated from category similarity vs. a benign anchor set,
                 boosted by detected manipulation tactics
  - tactics    = every tactic whose prototypes the text matches above threshold

modelVersion is stamped so every backend AIDecision row is reproducible.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from ..labels import BENIGN_PROTOTYPES, SCAM_CATEGORIES, TACTIC_PROTOTYPES
from ..models import EmbeddingBackbone, PrototypeClassifier, cosine_sim
from ..schemas import NlpClassificationInput, NlpClassificationOutput

router = APIRouter()

MODEL_VERSION = "nlp-minilm-zeroshot-1.0.0"

# Similarity thresholds (cosine, MiniLM). Tuned conservatively so benign prose
# doesn't trip a category.
CATEGORY_MIN_SIM = 0.30
TACTIC_MIN_SIM = 0.32

_category_clf = PrototypeClassifier(SCAM_CATEGORIES)
_tactic_clf = PrototypeClassifier(TACTIC_PROTOTYPES)
_benign_embeds: np.ndarray | None = None


def _benign() -> np.ndarray:
    global _benign_embeds
    if _benign_embeds is None:
        _benign_embeds = EmbeddingBackbone.instance().embed(BENIGN_PROTOTYPES)
    return _benign_embeds


@router.post("/nlp/classify", response_model=NlpClassificationOutput)
def classify(body: NlpClassificationInput) -> NlpClassificationOutput:
    text = (body.text or "").strip()
    if not text:
        return NlpClassificationOutput(
            category=body.hintedCategory,
            categoryConfidence=0.0,
            scamScore=0.0,
            manipulationTactics=[],
            modelVersion=MODEL_VERSION,
        )

    backbone = EmbeddingBackbone.instance()
    vec = backbone.embed_one(text)

    # ── Category ──
    cat_scores = _category_clf.score(text)
    best_cat = max(cat_scores, key=cat_scores.get)
    best_cat_sim = cat_scores[best_cat]
    if best_cat_sim < CATEGORY_MIN_SIM:
        category = body.hintedCategory
        category_conf = 0.0
    else:
        category = best_cat
        # Map cosine [CATEGORY_MIN_SIM..0.8] → confidence [40..95].
        category_conf = float(
            np.clip((best_cat_sim - CATEGORY_MIN_SIM) / (0.8 - CATEGORY_MIN_SIM), 0, 1) * 55 + 40
        )

    # ── Tactics ──
    tactic_scores = _tactic_clf.score(text)
    tactics = [t for t, s in tactic_scores.items() if s >= TACTIC_MIN_SIM]

    # ── Scam score: scam-similarity minus benign-similarity, + tactic boost ──
    benign_sim = float(np.max(cosine_sim(vec[None, :], _benign())[0]))
    contrast = best_cat_sim - benign_sim  # positive ⇒ looks more scammy than benign
    base = np.clip(contrast / 0.5, 0, 1) * 70  # contrast of ~0.5 ⇒ 70 pts
    scam_score = float(np.clip(base + len(tactics) * 8, 0, 100))

    return NlpClassificationOutput(
        category=category,
        categoryConfidence=round(category_conf, 1),
        scamScore=round(scam_score, 1),
        manipulationTactics=tactics,
        modelVersion=MODEL_VERSION,
    )
