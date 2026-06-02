"""POST /osint/{indicatorType} — safe, structural OSINT enrichment.

"Safe" = public-source / structural signals only, never PII (per the backend
contract and LR-5). This tier derives risk hints from the *shape* of an
indicator (TLD, length, punycode/homoglyph, digit ratio, free-email host,
disposable-domain patterns) without any external lookup. A network-backed
tier (passive DNS, whois age, cert transparency) can slot in later behind the
same contract; those require allow-listed outbound calls and are gated off by
default.
"""
from __future__ import annotations

import re

from fastapi import APIRouter

from ..schemas import OsintEnrichmentInput, OsintEnrichmentOutput

router = APIRouter()

MODEL_VERSION = "osint-structural-1.0.0"

SUSPICIOUS_TLDS = {"zip", "mov", "xyz", "top", "click", "country", "gq", "tk", "ml", "cf"}
FREE_EMAIL_HOSTS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me"}
DISPOSABLE_HINTS = ("mailinator", "guerrilla", "10minute", "tempmail", "trashmail")


def _domain_hints(value: str) -> list[str]:
    hints: list[str] = []
    host = value.lower().strip().rstrip(".")
    if host.startswith("xn--") or "xn--" in host:
        hints.append("punycode-domain")
    if re.search(r"[0-9]", host.split(".")[0]):
        hints.append("digits-in-label")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        hints.append(f"suspicious-tld:{tld}")
    label = host.split(".")[0]
    if len(label) >= 25:
        hints.append("unusually-long-label")
    if label.count("-") >= 3:
        hints.append("many-hyphens")
    return hints


def _email_hints(value: str) -> list[str]:
    hints: list[str] = []
    value = value.lower().strip()
    host = value.split("@")[-1] if "@" in value else ""
    if any(d in host for d in DISPOSABLE_HINTS):
        hints.append("disposable-email-host")
    if host in FREE_EMAIL_HOSTS:
        hints.append("free-email-host")
    hints.extend(_domain_hints(host))
    return hints


def _phone_hints(value: str) -> list[str]:
    digits = re.sub(r"\D", "", value)
    hints: list[str] = []
    if len(digits) < 7:
        hints.append("implausibly-short-number")
    if len(set(digits)) <= 2 and digits:
        hints.append("repeated-digit-pattern")
    return hints


@router.post("/osint/{indicator_type}", response_model=OsintEnrichmentOutput)
def osint(indicator_type: str, body: OsintEnrichmentInput) -> OsintEnrichmentOutput:
    value = body.normalizedIndicator
    kind = (body.indicatorType or indicator_type or "").upper()

    if kind in ("DOMAIN", "URL"):
        hints = _domain_hints(value)
    elif kind == "EMAIL":
        hints = _email_hints(value)
    elif kind == "PHONE":
        hints = _phone_hints(value)
    else:
        hints = []

    return OsintEnrichmentOutput(
        provider="vigiscam-structural",
        modelVersion=MODEL_VERSION,
        data={"indicatorType": kind, "structuralSignals": hints, "networkLookup": False},
        riskHints=hints,
    )
