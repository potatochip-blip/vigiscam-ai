# VIGISCAM AI worker — CPU image for Azure Container Apps.
# For a GPU node, base on an nvidia/cuda image, install the CUDA torch wheels,
# and set VIGISCAM_AI_DEVICE=cuda.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /srv

# System libs for the heavy authenticity tier: ffmpeg + libsndfile for audio
# decoding (voice-spoof model), libGL for image decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels for torch + torchaudio first (smaller, no CUDA).
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the always-on embedding backbone so the first request is fast
# and no outbound HF access is needed for the semantic tier at runtime. The
# heavy authenticity models lazy-load on first use (kept out of the image to
# keep the build lean; they download to HF_HOME on first authenticity call).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
