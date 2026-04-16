"""
Deterministic interest-aware re-ranking for retrieved docs.

This is a lightweight, no-LLM heuristic:
- Load saved tags/notes keyed by normalized title.
- Extract query keywords (simple tokenization + tiny stopword filter).
- Compute a bonus for docs whose saved tags overlap the query keywords.
- If no tag overlap is found, return the original order unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from databases.saved_metadata import get_tags_for_titles, normalize_title_key

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "last",
    "days",
    "day",
    "recent",
    "paper",
    "papers",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _extract_keywords(query: str) -> set[str]:
    tokens = _TOKEN_RE.findall(query.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}


def interest_aware_rerank(query: str, docs: list[Any]) -> list[Any]:
    """Return docs in interest-biased order, preserving original order if no matches."""
    if not docs:
        return docs

    keywords = _extract_keywords(query)
    if not keywords:
        return docs

    titles: list[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        t = meta.get("title") or ""
        titles.append(str(t))

    unique_titles: list[str] = []
    seen = set()
    for t in titles:
        if t and t not in seen:
            unique_titles.append(t)
            seen.add(t)

    tags_map = get_tags_for_titles(unique_titles)
    if not tags_map:
        return docs

    bonuses: list[int] = []
    for _d, title in zip(docs, titles, strict=True):
        title_key = normalize_title_key(title)
        tags = tags_map.get(title_key, [])
        if not tags:
            bonuses.append(0)
            continue
        bonus = 0
        for tag in tags:
            tag_l = str(tag).lower()
            if any(kw in tag_l for kw in keywords) or any(tag_l in kw for kw in keywords):
                bonus += 1
        bonuses.append(bonus)

    if max(bonuses) <= 0:
        return docs

    with_bonus = [d for d, b in zip(docs, bonuses, strict=True) if b > 0]
    without_bonus = [d for d, b in zip(docs, bonuses, strict=True) if b <= 0]
    return with_bonus + without_bonus
