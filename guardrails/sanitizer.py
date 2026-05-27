"""
Guardrails for prompt injection prevention.

Two surfaces are protected:
  1. Retrieved content (paper titles, authors, summaries) — via sanitize_retrieved()
  2. User input queries — via validate_user_input()

Input validation strategy (layered):
  - Primary: Prompt-Guard-86M classifier (Meta) — catches semantic paraphrases,
    unicode obfuscation, and indirect injection that regex misses.
  - Fallback: regex pattern registry — used when the model is unavailable
    (no HF access, CPU-only with tight latency budget).
  - Always applied: length cap and NFC unicode normalization.

Retrieved content sanitization still uses regex only — Prompt-Guard is
designed for user query classification, not arbitrary document text.
"""

import re
import unicodedata

from guardrails import prompt_guard as _prompt_guard

# ---------------------------------------------------------------------------
# Injection pattern registry
# Each entry is a raw regex fragment matched case-insensitively.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[str] = [
    # Role / identity override
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(the\s+)?(above|previous|prior|all)",
    r"forget\s+(everything|all|your|previous)",
    r"you\s+are\s+now\s+\w+",
    r"act\s+as\s+(a\s+|an\s+)?\w+",
    r"pretend\s+(to\s+be|you\s+are)",
    r"roleplay\s+as",
    r"simulate\s+(a\s+|an\s+)?\w+",
    r"your\s+(true|real|actual|hidden)\s+(self|instructions?|purpose|goal)",
    r"you\s+have\s+no\s+restrictions?",
    r"you\s+have\s+been\s+(freed|unlocked|jailbroken)",
    # Instruction injection
    r"new\s+instructions?\s*:",
    r"updated\s+instructions?\s*:",
    r"override\s+(instructions?|prompt|system)",
    r"your\s+instructions?\s+(are|say|state)\s*:",
    # System / structural injection
    r"\bsystem\s*prompt\b",
    r"\bsystem\s*message\b",
    r"<\s*system\s*>",
    r"\[system\]",
    r"<\s*/?\s*inst\s*>",  # llama-style <INST> tags
    r"\|\s*im_start\s*\|",  # chatml delimiters
    r"\|\s*im_end\s*\|",
    r"<\s*\|?\s*system\s*\|?\s*>",
    # Jailbreak keywords
    r"\bDAN\b",  # "Do Anything Now"
    r"do\s+anything\s+now",
    r"jailbroken?\b",
    r"developer\s+mode",
    r"sudo\s+mode",
    r"god\s+mode",
    r"unrestricted\s+mode",
    # Data/prompt exfiltration
    r"(reveal|print|output|repeat|show|display|leak)\s+(your\s+)?(system\s+)?(prompt|instructions?|context|config)",
    r"what\s+(are|were)\s+your\s+(original\s+)?instructions?",
    r"translate\s+(the\s+)?(above|previous|system)\s+(to|into)",
]

# Combined pattern for single-pass substitution in sanitize_retrieved.
_COMBINED = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)

# Hard length cap on retrieved fields — a 10 000-char "title" is a red flag.
_MAX_FIELD_LENGTH = 2000
_MAX_QUERY_LENGTH = 500


def _normalize(text: str) -> str:
    """NFC-normalize to collapse unicode lookalikes before pattern matching."""
    return unicodedata.normalize("NFC", text)


def sanitize_retrieved(text: str) -> str:
    """
    Sanitize text retrieved from the vector DB (titles, authors, summaries)
    before it is embedded in a prompt.

    - Truncates oversized fields
    - NFC-normalizes unicode
    - Replaces matched injection patterns with [blocked]
    """
    if not text:
        return text

    text = text[:_MAX_FIELD_LENGTH]
    text = _normalize(text)

    text = _COMBINED.sub("[blocked]", text)

    return text


class InputRejected(ValueError):
    """Raised when a user query is rejected by the input guardrail."""

    pass


def validate_user_input(query: str) -> str:
    """
    Validate and sanitize a user query before passing it to the agent.

    Raises InputRejected if the query looks like a prompt injection attempt.
    Returns the (possibly truncated + normalized) query otherwise.

    Detection strategy:
      1. Prompt-Guard-86M classifier (primary) — catches semantic paraphrases.
      2. Regex pattern registry (fallback) — used if model is unavailable.
    """
    if not query or not query.strip():
        raise InputRejected("Query must not be empty.")

    if len(query) > _MAX_QUERY_LENGTH:
        raise InputRejected(f"Query too long ({len(query)} chars). Max is {_MAX_QUERY_LENGTH}.")

    normalized = _normalize(query)

    # Primary: Prompt-Guard-86M model-based classifier.
    score = _prompt_guard.classify_injection(normalized)
    if score is not None:
        if score >= _prompt_guard.INJECTION_THRESHOLD:
            raise InputRejected(
                f"Query flagged as likely injection by Prompt-Guard "
                f"(score={score:.2f}, threshold={_prompt_guard.INJECTION_THRESHOLD})."
            )
        # Model ran successfully — skip regex to avoid false positives.
        return normalized

    # Fallback: regex heuristic when model is unavailable.
    if _COMBINED.search(normalized):
        raise InputRejected("Query contains a disallowed pattern and was rejected.")

    return normalized
