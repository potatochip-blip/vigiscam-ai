# VIGISCAM AI Worker

Self-hosted, open-model implementation of the AI engine the NestJS backend
calls. It exposes the **exact HTTP contract** the backend's AI clients expect,
so pointing the backend's `AI_SERVICE_URL` at this service flips every AI
engine from its in-process deterministic stub to a real model
(`source=EXTERNAL` in the backend's AI-decision audit). The backend keeps its
stubs as the permanent fallback — this service is always optional.

> **Status:** deployed to Azure Container Apps and verified live end-to-end —
> NLP, embeddings, insights, OSINT, and the deepfake-image authenticity check
> all confirmed returning `source=EXTERNAL` with the full decision envelope.
> Runs on a 2 vCPU / 4 GiB Consumption profile (the image model loads alongside
> MiniLM without needing a dedicated node).

## What's real today

A single open embedding model — **`all-MiniLM-L6-v2`** (384-dim, ~80 MB,
CPU-fast) — is the shared backbone. Every semantic capability runs as
zero-shot prototype similarity over that embedding space, which generalizes to
paraphrases that the backend's keyword stubs miss.

| Endpoint | Capability | Implementation |
|---|---|---|
| `POST /embeddings` | Vector similarity (4) | MiniLM sentence embeddings |
| `POST /nlp/classify` | NLP scam classification (1), message analysis (2), manipulation-tactic detection (3) | Semantic category + tactic scoring + calibrated scam score vs. a benign anchor set |
| `POST /insights/journey` | Fraud journey prediction (8) | Zero-shot stage classification |
| `POST /insights/victim-state` | Victim state assessment (9) | Zero-shot state classification |
| `POST /insights/predicted-move` | Predicted next move (10) | State-machine over inferred stage |
| `POST /osint/{type}` | OSINT enrichment (6) | Safe structural signals (TLD, punycode, disposable-email, digit patterns) — no PII, no network lookup |
| `POST /authenticity/{type}` | Authenticity checks (7) | Deterministic (DUAL_AUTH, CAM_VIGUARD) + **heavy ML**: deepfake image (LIVE_FACE_SEAL, SCENE_SEAL, ANTI_FAKE_VIDEO) + voice-spoof (VOICE_MATCH_SEAL) |

Confidence + explainability (11) are emitted on every response (scores +
top-label breakdown), which the backend persists on each AIDecision row.
Clustering (5) consumes `/embeddings` (the backend clusters on the vectors).

## Heavy authenticity models (open, self-hosted, swappable)

The model ids are **configuration, not hardcoded** (env-overridable), so the
layer is model-agnostic — swap a checkpoint without touching the contract.
Defaults are vetted Hugging Face models chosen for the task:

| Check | Default model | Why it fits |
|---|---|---|
| `LIVE_FACE_SEAL`, `SCENE_SEAL`, `ANTI_FAKE_VIDEO` | **`dima806/deepfake_vs_real_image_detection`** | ViT-base fine-tuned for real-vs-fake faces/images; the most-downloaded general deepfake-image classifier on the HF Hub; runs on CPU; binary `Real`/`Fake` labels the router maps directly. Video uses one sampled frame. |
| `VOICE_MATCH_SEAL` | **`MelodyMachine/Deepfake-audio-detection-V2`** | wav2vec2-based real-vs-spoof speech classifier via the audio-classification pipeline; token-based label mapping handles `fake/real` and `spoof/bonafide`. |

**Engineering honesty:** the label mapping is robust (case-insensitive token
match) and every path is graceful — a bad model id, missing media, or
inference error returns `INCONCLUSIVE`, never a crash. The image model is the
strongest *general* deepfake fit; a dedicated face-anti-spoofing checkpoint
(e.g. MiniFASNet/Silent-Face) can be swapped into `LIVE_FACE_SEAL` later via
the same env var. **Verify the exact audio repo id + its label names against
your data before trusting production verdicts** — the id is swappable for any
ASVspoof/anti-spoofing wav2vec2 checkpoint.

### Media: how the backend must send it

The heavy checks need the raw media in the `payload`:
`imageBase64`/`imageUrl` (or `frameBase64`/`frameUrl` for video) and
`audioBase64`/`audioUrl`. The current backend authenticity flow is
contract-only and does **not** yet capture/forward media — wiring that
(capture image/audio on the session → include in the authenticity payload)
is a small backend follow-up. Until then these checks return `INCONCLUSIVE`
(no media) even with the models live.

Models **lazy-load on first use** (not baked into the image, to keep the build
lean) and download to `HF_HOME`. First authenticity call is slow (~30–60 s
download + load); set `--min-replicas 1` so the container stays warm. On CPU
inference is a few seconds/image; a GPU workload profile speeds it up.

## Still the next tier

- **Network-backed OSINT** (passive DNS, whois age, cert transparency) —
  requires allow-listed outbound calls; gated off by default.
- **Feedback loop / retraining (12)** — the backend already audits every AI
  decision and has the reviewer-correction queue; turning that into a
  fine-tuning/active-learning loop is a dedicated pipeline.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# health: GET http://localhost:8000/health
```

Then point the backend at it:

```
AI_SERVICE_URL=http://localhost:8000
```

## Hybrid, model-agnostic pipeline

Every classification runs a layered pipeline (see `app/pipeline.py`):

1. **Self-hosted open model** (tier 1) — MiniLM semantic category + scam score.
2. **Rule-based detection** (tier 2, `app/rules.py`) — high-precision obvious
   signals ("read me the gift card code", "install AnyDesk") that emit explicit
   reason codes and can floor the risk score.
3. **Vector/embedding similarity** (tier 3) — scam-space vs. a benign anchor set.
4. **External API fallback** (tier 4, `app/external.py`) — OFF by default;
   optional second opinion from any OpenAI-compatible model for low-confidence
   or caller-flagged high-risk/enterprise cases. Never a hard dependency.
5. **Human review** (tier 5) — verdicts below the confidence threshold are
   flagged `requiresHumanReview: true` for the backend reviewer queue.

Every response carries a **decision envelope** — `modelUsed`, `modelVersion`,
`confidence`, `riskScore`, `reasonCodes`, `tier`, `requiresHumanReview`,
`evidenceRef` — which the backend persists verbatim on the AIDecision row.

## Deploy (Azure Container Apps, alongside the backend)

Push-to-deploy via `.github/workflows/deploy.yml` (mirrors the backend).
One-time setup:

1. Create a GitHub repo (e.g. `vigiscam-ai`) and push this directory.
2. Add the same OIDC **secrets** the backend repo uses: `AZURE_CLIENT_ID`,
   `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`.
3. Add repo **variables**: `ACR_NAME=acrvigiscamdevj37c5cbp6o2tq`,
   `RESOURCE_GROUP=rg-vigiscam-dev`, `AI_CONTAINER_APP_NAME=ca-vigiscam-dev-ai`.
4. Create the container app once (internal ingress so only the backend reaches
   it) in Cloud Shell, then point the backend at it:

```bash
az acr build --registry acrvigiscamdevj37c5cbp6o2tq --image vigiscam-ai:latest .
az containerapp create -n ca-vigiscam-dev-ai -g rg-vigiscam-dev \
  --environment <your-container-app-env> \
  --image acrvigiscamdevj37c5cbp6o2tq.azurecr.io/vigiscam-ai:latest \
  --target-port 8000 --ingress internal --cpu 2 --memory 4Gi --min-replicas 1
az containerapp update -n ca-vigiscam-dev-backend -g rg-vigiscam-dev \
  --set-env-vars "AI_SERVICE_URL=https://ca-vigiscam-dev-ai.internal.<env-domain>"
```

After that, every `git push` to main rebuilds + rolls out automatically. Once
`AI_SERVICE_URL` is set, `GET /api/v1/intelligence/ai-status` on the backend
flips the engines to `EXTERNAL`.

## Contract source of truth

The request/response shapes in `app/schemas.py` mirror the backend's
`*.types.ts` files exactly. If the backend contract changes, update schemas
here in lockstep.
