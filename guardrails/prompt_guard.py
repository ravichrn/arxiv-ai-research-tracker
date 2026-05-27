"""
Prompt-Guard-86M based injection classifier (Option 3).

Replaces the regex heuristic in validate_user_input() for jailbreak/injection
detection. Meta's Prompt-Guard-86M is a DeBERTa-based classifier trained
specifically on prompt injection and jailbreak examples — it catches semantic
paraphrases and obfuscated variants that regex misses.

The model is loaded lazily on first call and cached for the process lifetime.
Falls back to the regex heuristic if the model is unavailable (no HF access,
no disk space, CPU-only with tight latency budget).

Usage:
    from guardrails.prompt_guard import classify_injection
    score = classify_injection("ignore previous instructions")
    # score is float in [0, 1] — probability of injection/jailbreak
    # threshold: INJECTION_THRESHOLD (default 0.85)
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# Probability threshold above which a query is treated as injection.
# 0.85 gives high precision with low false-positive rate on arxiv queries.
INJECTION_THRESHOLD: float = float(os.getenv("PROMPT_GUARD_THRESHOLD", "0.85"))

_MODEL_ID = "meta-llama/Prompt-Guard-86M"

# Module-level cache — loaded once per process.
_pipeline = None
_load_attempted = False


def _load_pipeline():
    """Load the Prompt-Guard pipeline, caching the result."""
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline

    _load_attempted = True
    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(
            "text-classification",
            model=_MODEL_ID,
            device_map="auto",
            truncation=True,
            max_length=512,
        )
        _log.info("Prompt-Guard-86M loaded (device_map=auto).")
    except Exception as exc:
        _log.warning("Prompt-Guard-86M unavailable (%s). Falling back to regex guardrail.", exc)
        _pipeline = None

    return _pipeline


def classify_injection(text: str) -> float | None:
    """
    Return the injection/jailbreak probability score for text, or None if the
    model is unavailable.

    Prompt-Guard-86M outputs two labels: BENIGN and INJECTION (which covers
    both direct injection and jailbreak). We return the INJECTION score.
    """
    pipe = _load_pipeline()
    if pipe is None:
        return None

    try:
        result = pipe(text)
        # result is a list of dicts: [{"label": "INJECTION", "score": 0.97}]
        label = result[0]["label"].upper()
        score = result[0]["score"]
        # If the top label is BENIGN, the injection probability is 1 - score.
        return score if label == "INJECTION" else 1.0 - score
    except Exception as exc:
        _log.warning("Prompt-Guard inference failed (%s). Falling back to regex.", exc)
        return None


def is_injection(text: str) -> bool:
    """
    Return True if Prompt-Guard classifies text as injection above threshold.
    Returns False (not None) if the model is unavailable — callers should
    separately apply the regex fallback.
    """
    score = classify_injection(text)
    if score is None:
        return False
    return score >= INJECTION_THRESHOLD
