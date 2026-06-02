"""POST /embeddings — real sentence embeddings (MiniLM, 384-dim)."""
from fastapi import APIRouter

from ..models import EmbeddingBackbone
from ..schemas import EmbeddingInput, EmbeddingOutput

router = APIRouter()

MODEL_VERSION = "embedding-minilm-l6-v2-1.0.0"


@router.post("/embeddings", response_model=EmbeddingOutput)
def embeddings(body: EmbeddingInput) -> EmbeddingOutput:
    vec = EmbeddingBackbone.instance().embed_one(body.text)
    return EmbeddingOutput(vector=vec.tolist(), modelVersion=MODEL_VERSION)
