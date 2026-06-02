"""Shared model backbone.

A single SentenceTransformer (MiniLM) is loaded once and reused by every
semantic capability: embeddings, NLP category classification, manipulation-
tactic detection, and the journey/victim-state insight engines all run as
zero-shot prototype-similarity over the same embedding space. This keeps the
service light enough to run on CPU while being genuinely model-driven rather
than keyword heuristics.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from .config import get_settings

logger = logging.getLogger("vigiscam-ai.models")


class EmbeddingBackbone:
    """Lazy, thread-safe singleton around the sentence-transformer."""

    _instance: "EmbeddingBackbone | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_name = get_settings().embedding_model
        self._device = get_settings().device

    @classmethod
    def instance(cls) -> "EmbeddingBackbone":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Imported lazily so the module imports even where the
                    # heavy deps aren't installed (e.g. contract tests).
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s on %s", self._model_name, self._device)
                    self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings, shape (n, dim)."""
        model = self._ensure_loaded()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def warm(self) -> None:
        """Force-load + a tiny encode so the first real request is fast."""
        self.embed(["warmup"])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row(s) a and rows b. Inputs assumed
    L2-normalized, so this is just a dot product. Returns shape (len(a), len(b))
    or (len(b),) when a is 1-D."""
    return a @ b.T


class PrototypeClassifier:
    """Zero-shot classifier: embed a set of label prototype phrases once, then
    score any input text by max cosine similarity to each label's prototypes.

    This is the real-model replacement for the backend's keyword stubs — it
    generalizes to paraphrases the keyword lists would miss.
    """

    def __init__(self, label_prototypes: dict[str, list[str]]) -> None:
        self._labels = list(label_prototypes.keys())
        self._prototype_phrases = label_prototypes
        self._proto_embeds: dict[str, np.ndarray] | None = None

    def _ensure_embedded(self) -> dict[str, np.ndarray]:
        if self._proto_embeds is None:
            backbone = EmbeddingBackbone.instance()
            self._proto_embeds = {
                label: backbone.embed(phrases) for label, phrases in self._prototype_phrases.items()
            }
        return self._proto_embeds

    def score(self, text: str) -> dict[str, float]:
        """Per-label score in [0, 1] = max cosine over that label's prototypes."""
        protos = self._ensure_embedded()
        vec = EmbeddingBackbone.instance().embed_one(text)
        out: dict[str, float] = {}
        for label, emb in protos.items():
            sims = cosine_sim(vec[None, :], emb)[0]
            out[label] = float(np.max(sims))
        return out

    def best(self, text: str) -> tuple[str, float]:
        scores = self.score(text)
        label = max(scores, key=scores.get)
        return label, scores[label]
