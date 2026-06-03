# Deepfake Image Model — Evaluation

Evaluation of the default authenticity image model
(`dima806/deepfake_vs_real_image_detection`) run live through the deployed
worker via the backend `POST /api/v1/intelligence/authenticity`
(`LIVE_FACE_SEAL`), 2026-06-02.

## Method

Three labelled samples, each fetched by the worker from a public URL and run
through the real model (`source=EXTERNAL`):

| Sample | Label | Source |
|---|---|---|
| Portrait A | REAL | Wikimedia Commons (`Special:FilePath`) |
| Portrait B | REAL | Wikimedia Commons (`Special:FilePath`) |
| "Deep fake Tom Cruise" | FAKE | armywarcollege.edu (the @deeptomcruise face-swap) |

(Two further candidate links were not usable: a Facebook **reel** is video, not
an image, and a `grok.com/imagine` link is an HTML page — image checks need a
direct image-file URL or a video frame.)

## Results

| Sample | Expected | Result | score | fakeProbability | topLabel |
|---|---|---|---|---|---|
| Portrait A (real) | PASS | **PASS** ✅ | 94 | 0.053 | Real |
| Portrait B (real) | PASS | **PASS** ✅ | 99 | 0.004 | Real |
| Deepfake Tom Cruise | FAIL | **PASS** ❌ | 99 | 0.007 | Real |

## Findings

1. **The pipeline is correct end-to-end.** Real photos are authenticated
   (`PASS`), the call runs on the real model (`source=EXTERNAL`), and the full
   decision envelope (model, version, confidence, reason codes, risk score,
   evidenceRef) is returned and persisted.
2. **The model has a critical false-negative on a high-quality face-swap
   deepfake** — it classified the @deeptomcruise deepfake as "Real" at 0.99.
   For an anti-scam authenticity check, a false negative (calling a deepfake
   genuine) is the most dangerous error.

## Why this is expected (not a bug)

Deepfake-detector **generalisation across generation methods and datasets is
an open research problem.** A classifier trained on one family of fakes
routinely fails on another. The @deeptomcruise videos are exceptionally
high-quality (VFX artist + impersonator + face-swap) and defeat most
open-source still-image detectors. No single open image model reliably catches
top-tier deepfakes from one frame.

## Update (2026-06-02) — ENSEMBLE re-eval ✅ catches the deepfake

The worker now runs an **ensemble** (`vision.py`): `dima806` **+**
`prithivMLmods/Deep-Fake-Detector-Model`, aggregating by **MAX fake-probability**
(if any model confidently flags a fake, the verdict reflects it). Re-running the
exact same 3 samples through the real ensemble code (`training/eval_ensemble.py`):

| Sample | Expected | Result | ensemble fakeProb | dima806 | prithivMLmods |
|---|---|---|---|---|---|
| Portrait A (real) | PASS | **PASS** ✅ | 0.053 | 0.053 | 0.008 |
| Portrait B (real) | PASS | **PASS** ✅ | 0.395 | 0.004 | 0.395 |
| Deepfake Tom Cruise | FAIL | **FAIL** ✅ | **0.922** | 0.007 | **0.922** |

**Misclassified: 0 / 3.** The @deeptomcruise face-swap that the single
`dima806` model called "Real" at 0.99 is now correctly flagged **FAKE at 0.922**
— `prithivMLmods` catches it and the MAX-aggregation surfaces it. Both real
portraits still PASS (Portrait B rose to 0.395 but stayed under the 0.5 FAIL
threshold — the cost of higher sensitivity is a smaller margin on reals, worth
watching but not a misclassification).

**Caveat:** 3 samples is a smoke test, not a benchmark. The fine-tuning pipeline
in `training/` (scored on a held-out FaceForensics++/DFDC/Celeb-DF set, by
false-negative rate) remains the path to a defensible production number. But the
ensemble demonstrably closes the specific blind spot found in the first eval.

## Live Azure verification (2026-06-03) — end-to-end, EXTERNAL worker ✅

Verified through the **deployed backend** (`POST /api/v1/intelligence/authenticity`,
`LIVE_FACE_SEAL`) calling the **deployed AI worker**, sending the image as
base64 bytes (`scripts/smoke-authenticity.ps1` in the backend repo):

| Sample | Expected | Result | score | riskScore | reasonCodes | source |
|---|---|---|---|---|---|---|
| Deepfake Tom Cruise | FAIL | **FAIL** ✅ | 92 | 92.2 | DEEPFAKE_IMAGE_MODEL, TOP_LABEL:Fake:0.922, **FAKE_DOMINANT** | EXTERNAL |
| Real portrait | PASS | **PASS** ✅ | 94 | 5.3 | DEEPFAKE_IMAGE_MODEL, TOP_LABEL:Real:0.992, REAL_DOMINANT | EXTERNAL |

Two bugs were found and fixed during this verification (both would have hit any
real client, not just the test):

1. **Backend body limit** — Express's 100 KB default rejected every base64 image
   with `PayloadTooLargeError` before it reached the worker. Raised to 40 MB
   (covers the worker's 25 MB media cap + base64 inflation).
2. **Ensemble aggregation** — the ensemble returned `fake_prob` = max-fake from
   one model alongside `real_prob` = max-real from a *different* model, so the
   verdict comparison let a high "real" reading cancel a high "fake" reading (a
   confirmed Fake@0.922 came back `PASS`/`REAL_DOMINANT`). Fixed so `real_prob`
   is the **complement** of the sensitivity (max-fake) belief: `fake>0.5 ⟺ FAIL`.

The decision envelope (model, version, confidence, risk score, reason codes,
evidence ref, requiresHumanReview) is returned and persisted on every call.

## Recommendation — treat the image model as one signal, not an authority

The robust, already-built mitigation is **multi-signal**, with the image model
as a *contributing input*:

| Signal | Why a deepfake can't beat it |
|---|---|
| **CAM_VIGUARD** (device-fingerprint consistency) | Tied to the enrolled device, not the pixels |
| **DUAL_AUTH** (live challenge–response) | Requires a real-time correct response |
| **Behavioural / script signals** (NLP, tactics, urgency, payment pressure) | Detects the *scam*, independent of face realism |
| **Human-review flag** on low confidence | A reviewer adjudicates ambiguous cases |
| **Image deepfake model** | Catches low/mid-quality fakes; weights into the score |

The system already composes these — the authenticity verdict is never the sole
gate.

## If/when a production image model is chosen

- The model id is **configuration, not code** (`VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL`),
  so swapping a checkpoint is one env var + a restart — no code or backend
  change. Candidates to evaluate: `prithivMLmods/Deep-Fake-Detector-Model`,
  `Wvolf/ViT_Deepfake_Detection`, and any FaceForensics++/DFDC/Celeb-DF-trained
  checkpoint.
- **Choose it against a labelled test set** (FaceForensics++ / DFDC / Celeb-DF),
  not ad-hoc samples — report accuracy, and especially the **false-negative
  rate** on the deepfake class, which is the metric that matters here.
- Consider an **ensemble** (average several checkpoints) and per-frame
  aggregation for video, to reduce single-model blind spots.
