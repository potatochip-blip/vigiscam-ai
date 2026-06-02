"""Tier 2 — rule-based detection for *obvious* scam signals.

Runs before the model. These are high-precision patterns that, when present,
are strong enough on their own to push the verdict and emit an explicit reason
code — cheaper and more certain than the model for the unambiguous cases
(e.g. "read me the gift card code", "install AnyDesk and log into your bank").

Each rule returns a reason code + a score contribution. The pipeline combines
these with the model score; a rule with `hard=True` floors the risk score so
an obvious scam can't be talked down by a soft model reading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleHit:
    code: str
    points: int
    hard: bool  # if True, risk score is floored at `points`


# (reason_code, compiled_pattern, points, hard)
_RULES: list[tuple[str, re.Pattern[str], int, bool]] = [
    ("GIFT_CARD_CODE_REQUEST", re.compile(r"\b(read|give|tell|send).{0,20}(gift\s*card|card)\s*(code|number|pin)", re.I), 80, True),
    ("REMOTE_ACCESS_TOOL", re.compile(r"\b(anydesk|teamviewer|ultraviewer|remote\s*desktop)\b", re.I), 70, True),
    ("MONEY_TO_SAFE_ACCOUNT", re.compile(r"\b(move|transfer|send).{0,25}(safe|secure|protected)\s*account\b", re.I), 75, True),
    ("CRYPTO_TRANSFER_DEMAND", re.compile(r"\b(send|transfer|deposit).{0,20}(bitcoin|btc|ethereum|eth|usdt|crypto)\b", re.I), 60, False),
    ("CREDENTIAL_REQUEST", re.compile(r"\b(your|the)\s+(password|otp|one[-\s]?time\s*code|pin|ssn|social\s*security)\b", re.I), 55, False),
    ("ARREST_THREAT", re.compile(r"\b(arrest|warrant|jail|prosecut|deport)\w*\b", re.I), 50, False),
    ("URGENCY_NOW", re.compile(r"\b(right\s*now|immediately|within\s*the\s*next|expires?\s*(today|now)|act\s*now)\b", re.I), 25, False),
    ("SECRECY", re.compile(r"\b(don'?t|do\s*not)\s*(tell|discuss|mention).{0,20}(anyone|family|bank|police)\b", re.I), 30, False),
]


def apply_rules(text: str) -> list[RuleHit]:
    if not text:
        return []
    hits: list[RuleHit] = []
    for code, pattern, points, hard in _RULES:
        if pattern.search(text):
            hits.append(RuleHit(code=code, points=points, hard=hard))
    return hits
