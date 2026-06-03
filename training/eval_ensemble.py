"""Re-run the deepfake eval against the ENSEMBLE (app.vision.analyze_image).

Runs the *actual* worker ensemble code locally on the same 3 labelled samples
used in MODEL-EVALUATION.md, so we can measure whether the two-model ensemble
(dima806 + prithivMLmods, aggregated by MAX fake-prob) beats the single model.
"""
from __future__ import annotations

import sys
import urllib.request

# Force the ensemble config the deployed worker uses.
import os
os.environ.setdefault(
    "VIGISCAM_AI_DEEPFAKE_IMAGE_MODELS",
    "dima806/deepfake_vs_real_image_detection,prithivMLmods/Deep-Fake-Detector-Model",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.vision import analyze_image  # noqa: E402

SAMPLES = [
    ("Portrait A", "REAL", "https://commons.wikimedia.org/wiki/Special:FilePath/03alcob.jpg"),
    ("Portrait B", "REAL", "https://commons.wikimedia.org/wiki/Special:FilePath/054A3180_%282%29.jpg"),
    ("Deepfake Tom Cruise", "FAKE", "https://warroom.armywarcollege.edu/wp-content/uploads/21-057-Deep_fake_Tom_Cruise.jpeg"),
]

UA = {"User-Agent": "Mozilla/5.0 (VIGISCAM-eval)"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def verdict(fake_prob: float) -> str:
    # Same threshold semantics as the authenticity service: >0.5 fake → FAIL.
    return "FAIL" if fake_prob > 0.5 else "PASS"


def main() -> int:
    print("Loading ensemble (downloads models on first run)...\n")
    rows = []
    misses = 0
    for name, label, url in SAMPLES:
        try:
            img = fetch(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not fetch {name}: {exc}")
            continue
        res = analyze_image(img)
        if not res:
            print(f"  ! {name}: ensemble returned None (no model recognized labels)")
            continue
        fp = res["fake_prob"]
        v = verdict(fp)
        expected = "FAIL" if label == "FAKE" else "PASS"
        ok = "OK" if v == expected else "MISS"
        if ok == "MISS":
            misses += 1
        rows.append((name, label, v, expected, ok, fp, res.get("ensemble", {})))
        print(f"[{ok:4}] {name:22} label={label:4} verdict={v:4} (expected {expected})")
        print(f"        ensemble fake_prob={fp:.3f}  per-model={res.get('ensemble')}")
        print(f"        top_label={res.get('top_label')} driver_model={res.get('model')}\n")

    print("=" * 64)
    print(f"Samples scored: {len(rows)}   Misclassified: {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
