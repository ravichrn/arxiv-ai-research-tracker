"""
Output validation for RAG responses (Option 1 — Guardrails AI pattern).

The guardrails-ai package imports as `guardrails`, which collides with this
project's own `guardrails/` directory. We implement the same validation
contract directly, without the library dependency.

Two validators are applied to every RAG response before it reaches the user:

1. ArxivCitationValidator
   Checks that any arxiv ID mentioned in the response (e.g. 2312.01234 or
   arxiv:2312.01234) actually exists in the local papers database. Flags
   IDs that look fabricated.

2. ToxicLanguageValidator
   Uses a lightweight classifier (unitary/toxic-bert, 110M params) to score
   the response for toxic content. Raises OutputRejected above threshold.
   Falls back to an allow if the model is unavailable.

Both validators follow the Guardrails AI validator contract:
    validate(value: str, metadata: dict) -> ValidationResult

where ValidationResult is either PassResult or FailResult.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation result types  (mirror guardrails-ai's PassResult / FailResult)
# ---------------------------------------------------------------------------


@dataclass
class PassResult:
    value: str


@dataclass
class FailResult:
    error_message: str
    fix_value: str | None = None


ValidationResult = PassResult | FailResult


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class OutputRejected(ValueError):
    """Raised when an LLM response fails output validation."""


# ---------------------------------------------------------------------------
# Validator 1 — ArxivCitationValidator
# ---------------------------------------------------------------------------

# Matches bare IDs like 2312.01234 or 2312.01234v2
# and prefixed forms: arxiv:2312.01234, arXiv:2312.01234v2
_ARXIV_ID_RE = re.compile(r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


class ArxivCitationValidator:
    """
    Validates that arxiv IDs cited in a response exist in the local database.

    Known IDs are passed in via metadata["known_arxiv_ids"] (a set of strings).
    IDs that are mentioned but not found are collected into FailResult.
    """

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        known_ids: set[str] = metadata.get("known_arxiv_ids", set())
        if not known_ids:
            # No DB context provided — skip citation check.
            return PassResult(value=value)

        mentioned = _ARXIV_ID_RE.findall(value)
        # Normalize: strip version suffix for lookup (2312.01234v2 → 2312.01234)
        hallucinated = []
        for raw_id in mentioned:
            base_id = raw_id.split("v")[0]
            if base_id not in known_ids and raw_id not in known_ids:
                hallucinated.append(raw_id)

        if hallucinated:
            return FailResult(
                error_message=(
                    f"Response cites arxiv IDs not found in the database: "
                    f"{', '.join(hallucinated)}. These may be hallucinated."
                ),
                fix_value=None,
            )

        return PassResult(value=value)


# ---------------------------------------------------------------------------
# Validator 2 — ToxicLanguageValidator
# ---------------------------------------------------------------------------

_TOXIC_THRESHOLD: float = 0.80
_TOXIC_MODEL_ID = "unitary/toxic-bert"

_toxic_pipeline = None
_toxic_load_attempted = False


def _load_toxic_pipeline():
    global _toxic_pipeline, _toxic_load_attempted
    if _toxic_load_attempted:
        return _toxic_pipeline
    _toxic_load_attempted = True
    try:
        from transformers import pipeline as hf_pipeline

        _toxic_pipeline = hf_pipeline(
            "text-classification",
            model=_TOXIC_MODEL_ID,
            device_map="auto",
            truncation=True,
            max_length=512,
            top_k=None,
        )
        _log.info("toxic-bert loaded for output validation.")
    except Exception as exc:
        _log.warning("toxic-bert unavailable (%s). Toxicity check will be skipped.", exc)
        _toxic_pipeline = None
    return _toxic_pipeline


class ToxicLanguageValidator:
    """
    Validates that LLM output doesn't contain toxic language.

    Uses unitary/toxic-bert (110M params). Falls back to PassResult if the
    model is unavailable so the application stays usable without GPU/internet.
    """

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        pipe = _load_toxic_pipeline()
        if pipe is None:
            return PassResult(value=value)

        try:
            results = pipe(value[:512])
            # results is a list of label/score dicts when top_k=None
            scores = {r["label"].lower(): r["score"] for r in results[0]}
            toxic_score = scores.get("toxic", 0.0)
            if toxic_score >= _TOXIC_THRESHOLD:
                return FailResult(
                    error_message=(
                        f"Response flagged as potentially toxic "
                        f"(score={toxic_score:.2f}, threshold={_TOXIC_THRESHOLD})."
                    )
                )
        except Exception as exc:
            _log.warning("Toxicity check failed (%s). Allowing response.", exc)

        return PassResult(value=value)


# ---------------------------------------------------------------------------
# Guard — runs all validators in sequence
# ---------------------------------------------------------------------------

_DEFAULT_VALIDATORS = [ArxivCitationValidator(), ToxicLanguageValidator()]


def validate_output(response: str, metadata: dict | None = None) -> str:
    """
    Run all output validators on a RAG response.

    Args:
        response: The LLM-generated response string.
        metadata: Optional dict; pass {"known_arxiv_ids": set(...)} for
                  citation checking.

    Returns:
        The original response if all validators pass.

    Raises:
        OutputRejected: if any validator returns a FailResult.
    """
    meta = metadata or {}
    for validator in _DEFAULT_VALIDATORS:
        result = validator.validate(response, meta)
        if isinstance(result, FailResult):
            raise OutputRejected(result.error_message)
    return response
