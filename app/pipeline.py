"""The hybrid, model-agnostic decision pipeline.

Order, per the architecture:
  1. Self-hosted open model  (tier 1) — semantic category + scam score
  2. Rule-based detection    (tier 2) — obvious-signal reason codes, can floor risk
  3. Vector/embedding search (tier 3) — similarity to the scam prototype space
  4. External API fallback   (tier 4) — optional, low-confidence/high-risk only
  5. Human review            (tier 5) — flag when final confidence is low

Every run produces a `Decision` (model, version, confidence, reason codes,
risk score, evidence ref, tier, requiresHumanReview) attached to the response.
"""
from __future__ import annotations

import numpy as np

from .config import get_settings
from .decision import Decision, Tier
from .external import external_classify, should_escalate
from .labels import BENIGN_PROTOTYPES, SCAM_CATEGORIES, TACTIC_PROTOTYPES
from .models import EmbeddingBackbone, PrototypeClassifier, cosine_sim
from .rules import apply_rules

NLP_MODEL_VERSION = "nlp-hybrid-minilm-1.1.0"
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


def classify_nlp(
    text: str,
    hinted_category: str | None,
    high_risk: bool = False,
    evidence_ref: str | None = None,
) -> tuple[dict, Decision]:
    """Run the hybrid pipeline. Returns (nlp_output_dict, Decision)."""
    text = (text or "").strip()
    reason_codes: list[str] = []

    if not text:
        decision = Decision(
            model_used=NLP_MODEL_VERSION,
            model_version=NLP_MODEL_VERSION,
            confidence=0.0,
            risk_score=0.0,
            reason_codes=["EMPTY_INPUT"],
            tier=Tier.RULES,
            requires_human_review=False,
            evidence_ref=evidence_ref,
        )
        out = {
            "category": hinted_category,
            "categoryConfidence": 0.0,
            "scamScore": 0.0,
            "manipulationTactics": [],
            "modelVersion": NLP_MODEL_VERSION,
        }
        return out, decision

    backbone = EmbeddingBackbone.instance()
    vec = backbone.embed_one(text)
    tier = Tier.SELF_HOSTED

    # ── Tier 1: self-hosted model — category + tactics + base scam score ──
    cat_scores = _category_clf.score(text)
    best_cat = max(cat_scores, key=cat_scores.get)
    best_cat_sim = cat_scores[best_cat]
    if best_cat_sim < CATEGORY_MIN_SIM:
        category = hinted_category
        category_conf = 0.0
    else:
        category = best_cat
        category_conf = float(
            np.clip((best_cat_sim - CATEGORY_MIN_SIM) / (0.8 - CATEGORY_MIN_SIM), 0, 1) * 55 + 40
        )
        reason_codes.append(f"SEMANTIC_MATCH:{best_cat}")

    tactic_scores = _tactic_clf.score(text)
    tactics = [t for t, s in tactic_scores.items() if s >= TACTIC_MIN_SIM]
    reason_codes.extend(f"TACTIC:{t}" for t in tactics)

    # ── Tier 3: vector similarity — scam space vs benign anchor ──
    benign_sim = float(np.max(cosine_sim(vec[None, :], _benign())[0]))
    contrast = best_cat_sim - benign_sim
    model_risk = float(np.clip(np.clip(contrast / 0.5, 0, 1) * 70 + len(tactics) * 8, 0, 100))
    if contrast > 0.1:
        reason_codes.append("VECTOR_SCAM_SIMILARITY")

    # ── Tier 2: rules — obvious signals, can floor the risk ──
    rule_hits = apply_rules(text)
    rule_points = sum(h.points for h in rule_hits)
    hard_floor = max((h.points for h in rule_hits if h.hard), default=0)
    reason_codes.extend(f"RULE:{h.code}" for h in rule_hits)

    risk_score = float(np.clip(max(model_risk + min(rule_points, 40), hard_floor), 0, 100))
    if hard_floor > 0:
        tier = Tier.RULES  # an obvious-signal rule dominated the verdict

    # Confidence: how sure we are of the verdict (not the risk itself). High
    # when model + rules agree or a hard rule fired; lower when only weak
    # signals.
    confidence = float(
        np.clip(
            max(category_conf, hard_floor and 90, 30 + len(reason_codes) * 8),
            0,
            100,
        )
    )

    # ── Tier 4: external API fallback (optional, low-confidence/high-risk) ──
    if should_escalate(confidence, high_risk):
        ext = external_classify(text)
        if ext is not None:
            tier = Tier.EXTERNAL
            reason_codes.append("EXTERNAL_SECOND_OPINION")
            # Blend: trust the external scamScore, keep the higher risk.
            risk_score = float(np.clip(max(risk_score, ext["scamScore"]), 0, 100))
            if ext.get("category"):
                category = ext["category"]
            confidence = float(np.clip(max(confidence, 75), 0, 100))
            if ext.get("reason"):
                reason_codes.append(f"EXTERNAL_REASON:{ext['reason'][:80]}")

    # ── Tier 5: human review flag ──
    requires_review = confidence < get_settings().review_confidence_threshold
    if requires_review:
        reason_codes.append("LOW_CONFIDENCE_REVIEW")

    decision = Decision(
        model_used=NLP_MODEL_VERSION,
        model_version=NLP_MODEL_VERSION,
        confidence=confidence,
        risk_score=risk_score,
        reason_codes=reason_codes,
        tier=tier,
        requires_human_review=requires_review,
        evidence_ref=evidence_ref,
    )
    out = {
        "category": category,
        "categoryConfidence": round(category_conf, 1),
        "scamScore": round(risk_score, 1),
        "manipulationTactics": tactics,
        "modelVersion": NLP_MODEL_VERSION,
    }
    return out, decision
