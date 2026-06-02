"""Runtime configuration for the VIGISCAM AI worker.

All values are environment-overridable so the same image runs CPU-only in
dev/CI and on a GPU node in production. The backend points `AI_SERVICE_URL`
at this service; when that env is unset the backend falls back to its
in-process deterministic stubs (so this service is always optional).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIGISCAM_AI_", env_file=".env")

    # ── Serving ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Models (open-source, self-hosted) ─────────────────────────────────────
    # MiniLM is the shared backbone: 384-dim sentence embeddings, ~80 MB,
    # fast on CPU. It powers /embeddings AND the semantic classifiers for
    # /nlp/classify and /insights/* (zero-shot via prototype similarity).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # "cuda" on a GPU node

    # Warm the model on startup so the first request isn't slow. Disable in
    # tests where the model isn't available.
    warm_on_startup: bool = True

    # Optional heavier models for the vision/voice authenticity tier. Left
    # unset by default — those checks return INCONCLUSIVE until configured.
    deepfake_image_model: str | None = None
    voice_spoof_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
