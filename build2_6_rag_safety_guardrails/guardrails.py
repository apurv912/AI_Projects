from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GuardrailDecision:
    injection_detected: bool
    risk_domain: str  # "none" | "medical" | "legal" | "financial"
    strict_mode: bool
    reason: str = ""


# --- Prompt injection / jailbreak patterns (lightweight + generic) ---
# Goal: catch common attempts to override system/developer instructions or grounding rules.
_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    # classic jailbreak / prompt exfil
    ("ignore_instructions", r"\bignore (all|any|previous) (instructions|rules)\b"),
    ("system_prompt", r"\b(system prompt|developer message|hidden prompt)\b"),
    ("reveal_secrets", r"\b(reveal|show|leak).*(secrets|keys|api key|credentials)\b"),
    ("do_anything_now", r"\b(do anything now|dan mode|jailbreak)\b"),
    ("override_policy", r"\b(disregard|bypass|override) (policy|rules|safety)\b"),

    # grounding override / citation suppression (very common in RAG attacks)
    ("ignore_context", r"\bignore (the )?(chunks|chunk|context|retrieved|evidence|sources)\b"),
    ("ignore_retrieval", r"\b(ignore|skip) (retrieval|citations|sources)\b"),
    ("answer_general_knowledge", r"\banswer from (your )?(general knowledge|training data|memory)\b"),
    ("no_citations", r"\b(do not|don't)\s+(cite|include citations|provide citations)\b"),
    ("without_citations", r"\b(without citations|no citations|don’t cite anything|don't cite anything)\b"),
]


_INJ_RE = re.compile("|".join(f"(?:{pat})" for _, pat in _INJECTION_PATTERNS), re.IGNORECASE)


# --- Risk domain detection (simple heuristic) ---
_MEDICAL_RE = re.compile(r"\b(chest pain|dose|dosage|medication|symptom|diagnos|treatment|prescription|mg|doctor)\b", re.I)
_LEGAL_RE = re.compile(r"\b(legal advice|lawsuit|sue|contract|visa|immigration|bypass|evade|illegal)\b", re.I)
_FIN_RE = re.compile(r"\b(invest|investment|crypto|stocks|options|portfolio|buy|sell|returns|profit)\b", re.I)


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, ""
    if _INJ_RE.search(t):
        return True, "prompt_injection_pattern_match"
    return False, ""


def detect_risk_domain(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "none"
    if _MEDICAL_RE.search(t):
        return "medical"
    if _LEGAL_RE.search(t):
        return "legal"
    if _FIN_RE.search(t):
        return "financial"
    return "none"


def decide_guardrails(question: str) -> GuardrailDecision:
    inj, inj_reason = detect_prompt_injection(question)
    risk = detect_risk_domain(question)

    # strict_mode is engaged when we detect injection or risky domain
    strict = bool(inj or (risk != "none"))

    reason = ""
    if inj:
        reason = inj_reason
    elif risk != "none":
        reason = f"risk_domain_{risk}"

    return GuardrailDecision(
        injection_detected=bool(inj),
        risk_domain=risk,
        strict_mode=bool(strict),
        reason=reason,
    )


def evidence_strength_ok(retrieved: List[Tuple[str, float, str]], min_score: float) -> bool:
    """
    Lightweight evidence strength check: require at least one chunk at/above threshold.
    """
    if not retrieved:
        return False
    try:
        best = max(float(score) for _, score, _ in retrieved)
        return best >= float(min_score)
    except Exception:
        return False


def build_refusal_message(reason: str = "") -> str:
    msg = "I can’t help with that request."
    if reason:
        msg += f" (Reason: {reason})"
    msg += "\n\nIf you can rephrase your question to focus on the provided document, I can try again."
    return msg


def build_not_enough_info_message() -> str:
    return "Not enough information in the provided document to answer that safely."


def build_risky_domain_safe_message(domain: str) -> str:
    domain = (domain or "none").lower()

    if domain == "medical":
        return (
            "I can’t provide medical diagnosis or dosage advice.\n\n"
            "If this is urgent (e.g., chest pain, difficulty breathing, fainting), seek emergency help immediately.\n"
            "If not urgent, consult a qualified clinician. If you share what the document says, I can summarize it with citations."
        )

    if domain == "legal":
        return (
            "I can’t provide legal advice or guidance to bypass laws.\n\n"
            "For legal questions, consult an accredited professional or official government resources. "
            "If the document contains relevant policy text, I can summarize it with citations."
        )

    if domain == "financial":
        return (
            "I can’t provide definitive financial or investment advice.\n\n"
            "Consider risk tolerance, diversification, and professional guidance. "
            "If the document includes financial context, I can summarize it with citations."
        )

    return build_not_enough_info_message()
