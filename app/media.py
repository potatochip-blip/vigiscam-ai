"""Media loading for the heavy authenticity tier.

The backend's AuthenticityRequest carries the media in `payload`. We accept it
two ways so the backend can choose whichever is cheaper:

  - inline base64:  payload["imageBase64"] / payload["audioBase64"]
  - by reference:   payload["imageUrl"]    / payload["audioUrl"]  (fetched)

Returns decoded bytes / a (waveform, sample_rate) pair, or None when no usable
media is present (→ the check returns INCONCLUSIVE rather than guessing).
"""
from __future__ import annotations

import base64
import binascii
import logging
import urllib.request
from typing import Any

logger = logging.getLogger("vigiscam-ai.media")

_MAX_FETCH_BYTES = 25 * 1024 * 1024  # 25 MB ceiling on fetched media


def _strip_data_uri(b64: str) -> str:
    # "data:image/png;base64,...." → "...."
    if b64.startswith("data:") and "," in b64:
        return b64.split(",", 1)[1]
    return b64


def _from_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode(_strip_data_uri(value), validate=False)
    except (binascii.Error, ValueError) as exc:
        logger.warning("base64 decode failed: %s", exc)
        return None


def _from_url(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vigiscam-ai/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — backend-supplied URL
            data = resp.read(_MAX_FETCH_BYTES + 1)
        if len(data) > _MAX_FETCH_BYTES:
            logger.warning("media at %s exceeds size ceiling", url)
            return None
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("media fetch failed for %s: %s", url, exc)
        return None


def load_image_bytes(payload: dict[str, Any]) -> bytes | None:
    if not payload:
        return None
    if payload.get("imageBase64"):
        return _from_base64(str(payload["imageBase64"]))
    if payload.get("imageUrl"):
        return _from_url(str(payload["imageUrl"]))
    # A single video frame can be supplied the same way.
    if payload.get("frameBase64"):
        return _from_base64(str(payload["frameBase64"]))
    if payload.get("frameUrl"):
        return _from_url(str(payload["frameUrl"]))
    return None


def load_audio_bytes(payload: dict[str, Any]) -> bytes | None:
    if not payload:
        return None
    if payload.get("audioBase64"):
        return _from_base64(str(payload["audioBase64"]))
    if payload.get("audioUrl"):
        return _from_url(str(payload["audioUrl"]))
    return None
