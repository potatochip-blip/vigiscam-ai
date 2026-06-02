"""VIGISCAM AI worker — FastAPI entry point.

Implements the exact HTTP contract the NestJS backend's AI clients call
(/nlp/classify, /embeddings, /insights/*, /osint/*, /authenticity/*). Point
the backend's AI_SERVICE_URL at this service to flip every AI engine from its
in-process stub to these real models (source=EXTERNAL in the backend audit).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .models import EmbeddingBackbone
from .routers import authenticity, embeddings, insights, nlp, osint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vigiscam-ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.warm_on_startup:
        try:
            EmbeddingBackbone.instance().warm()
            logger.info("Embedding backbone warmed and ready")
        except Exception as exc:  # noqa: BLE001 — never block startup on model load
            logger.warning("Could not warm embedding model at startup: %s", exc)
    yield


app = FastAPI(title="VIGISCAM AI Worker", version="1.0.0", lifespan=lifespan)

app.include_router(nlp.router, tags=["nlp"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(insights.router, tags=["insights"])
app.include_router(osint.router, tags=["osint"])
app.include_router(authenticity.router, tags=["authenticity"])


@app.get("/health", tags=["health"])
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": "vigiscam-ai",
        "embeddingModel": settings.embedding_model,
        "device": settings.device,
    }
