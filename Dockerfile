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
# and no outbound HF access is needed for the semantic tier at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Bake the deepfake-image ENSEMBLE into the image so authenticity calls need no
# runtime HF download (deterministic cold starts, no outbound egress, no
# first-call latency spike). Kept BEFORE `COPY app` so this ~700MB layer is
# cached across app-only code changes. Must stay in sync with the
# `deepfake_image_models` default in app/config.py.
ARG DEEPFAKE_IMAGE_MODELS="dima806/deepfake_vs_real_image_detection,prithivMLmods/Deep-Fake-Detector-Model"
RUN python -c "import os; from transformers import pipeline; \
    [pipeline('image-classification', model=m, device=-1) \
     for m in os.environ['DEEPFAKE_IMAGE_MODELS'].split(',') if m]"
# Bake the voice-spoof model too (authenticity VOICE_MATCH_SEAL).
ARG VOICE_SPOOF_MODEL="MelodyMachine/Deepfake-audio-detection-V2"
RUN python -c "import os; from transformers import pipeline; \
    pipeline('audio-classification', model=os.environ['VOICE_SPOOF_MODEL'], device=-1)" \
    || echo "voice model bake skipped (lazy-loads at runtime)"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
