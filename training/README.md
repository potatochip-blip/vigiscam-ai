# Training the VIGISCAM deepfake detector

This pipeline fine-tunes an open Vision Transformer on a labelled real-vs-fake
face dataset and produces a checkpoint the worker uses via
`VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL`. **It requires a GPU and a dataset** — it is
not something the running service does on its own.

## Why fine-tune (and the honest limits)

The default off-the-shelf models miss high-quality face-swap deepfakes
(see `../MODEL-EVALUATION.md`). Fine-tuning on data that matches **your threat
distribution** — i.e. the kind of impersonation deepfakes used in live video
scam calls — is the only reliable way to lift detection on that distribution.
Even then, top-tier deepfakes remain hard; treat the model as one signal in the
multi-signal authenticity design (liveness challenges + behaviour + review).

## 1. Get a dataset (licensed)

Pick one (or combine), each needs a signed agreement / form:

| Dataset | Content | Access |
|---|---|---|
| **FaceForensics++** | face-swap/reenactment deepfakes | github.com/ondyari/FaceForensics (EULA) |
| **DFDC** | Deepfake Detection Challenge | kaggle.com/c/deepfake-detection-challenge |
| **Celeb-DF v2** | celebrity face-swaps | github.com/yuezunli/celeb-deepfakeforensics (form) |

Extract **face-cropped frames** (one or a few per video) into the ImageFolder
layout. For video datasets, sample frames with `ffmpeg`/`opencv` and crop faces
(e.g. with `facenet-pytorch` MTCNN) before training — face crops train far
better than full frames.

## 2. Arrange the data

```
data/
  train/{real,fake}/*.jpg
  val/{real,fake}/*.jpg
```
Keep classes balanced and put *different source identities/videos* in val than
train (so you measure generalisation, not memorisation).

## 3. Train (GPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python train.py --data-dir ./data --base google/vit-base-patch16-224 \
  --output ./vigiscam-deepfake-vit --epochs 5 --batch-size 32
```

The trainer selects the checkpoint with the **lowest false-negative rate on the
fake class** (missed fakes is the metric that matters here), not just accuracy.

## 4. Use the checkpoint

- **Local path** (bake into the worker image or mount a volume):
  ```
  VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL=/models/vigiscam-deepfake-vit
  ```
- **Or upload to Hugging Face** (`huggingface-cli upload <you>/vigiscam-deepfake-vit ./vigiscam-deepfake-vit`) and set the env var to the HF id.

Set it on the AI container app and restart; the worker loads it on the next
authenticity call. Re-run the evaluation in `../MODEL-EVALUATION.md` to confirm
the false-negative rate dropped.

## 5. Iterate

- Add more/varied fakes to train; re-train.
- Consider an **ensemble** — the worker already supports multiple checkpoints
  via `VIGISCAM_AI_DEEPFAKE_IMAGE_MODELS` (comma-separated); add your fine-tuned
  model alongside the open ones and the worker aggregates (max fake-prob).
- For video, sample several frames per call and aggregate.
