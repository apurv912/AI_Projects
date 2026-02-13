from __future__ import annotations

import re
from dataclasses import dataclass


# Basic patterns (intentionally lightweight; not perfect)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Phone: catches common international + local formats with separators
_PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})"
)

# Avoid masking very short numeric fragments (reduce false positives)
_MIN_PHONE_DIGITS = 8


@dataclass
class RedactionResult:
    text: str
    redacted: bool
    emails_masked: int = 0
    phones_masked: int = 0


def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def redact_pii(text: str) -> RedactionResult:
    src = text or ""
    emails = _EMAIL_RE.findall(src)
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", src)

    phones_found = []
    for m in _PHONE_RE.finditer(out):
        val = m.group(0)
        if _count_digits(val) >= _MIN_PHONE_DIGITS:
            phones_found.append(val)

    # Replace phones conservatively (only those with enough digits)
    def _phone_sub(match: re.Match) -> str:
        val = match.group(0)
        if _count_digits(val) >= _MIN_PHONE_DIGITS:
            return "[REDACTED_PHONE]"
        return val

    out2 = _PHONE_RE.sub(_phone_sub, out)

    return RedactionResult(
        text=out2,
        redacted=(out2 != src),
        emails_masked=len(emails),
        phones_masked=len(phones_found),
    )
