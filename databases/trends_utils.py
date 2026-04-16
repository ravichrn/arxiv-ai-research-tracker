"""
Deterministic trend analysis for recently published papers.

Given paper dicts with `published` (ISO 8601) and `categories` (comma-separated),
compute category counts over a "recent" and "previous" window and rank rising
categories by absolute delta (and secondarily relative growth).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


def _parse_published(published: str) -> datetime | None:
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(published)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _primary_category(categories: str) -> str:
    # Keep consistent with other parts of the codebase (digest/tag).
    if not categories:
        return "unknown"
    return categories.split(",")[0].strip() or "unknown"


@dataclass(frozen=True)
class TrendRow:
    category: str
    recent_count: int
    prev_count: int
    delta: int
    growth_ratio: float
    example_titles: list[str]


def compute_category_trends(
    papers: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    recent_days: int = 14,
    previous_days: int = 14,
    top_k: int = 5,
    examples_per_category: int = 3,
) -> list[TrendRow]:
    """Compute rising categories over two adjacent time windows."""
    now = now or datetime.now(UTC)
    recent_start = now - timedelta(days=recent_days)
    prev_start = now - timedelta(days=recent_days + previous_days)
    prev_end = recent_start

    recent_counts: dict[str, int] = {}
    prev_counts: dict[str, int] = {}
    recent_titles: dict[str, list[str]] = {}

    for p in papers:
        published_raw = str(p.get("published", "")).strip()
        dt = _parse_published(published_raw)
        if dt is None:
            continue

        cat = _primary_category(str(p.get("categories", "")).strip())
        title = str(p.get("title", "")).strip()

        if dt >= recent_start and dt <= now:
            recent_counts[cat] = recent_counts.get(cat, 0) + 1
            if title:
                recent_titles.setdefault(cat, []).append(title)
        elif dt >= prev_start and dt < prev_end:
            prev_counts[cat] = prev_counts.get(cat, 0) + 1

    candidates = set(recent_counts.keys()) | set(prev_counts.keys())
    rows: list[TrendRow] = []
    for cat in candidates:
        r = recent_counts.get(cat, 0)
        prev = prev_counts.get(cat, 0)
        delta = r - prev
        growth_ratio = r / max(1, prev)
        if delta <= 0 and r == 0:
            # Only surface categories that appear (or rise) in the recent window.
            continue

        examples = recent_titles.get(cat, [])[:examples_per_category]
        rows.append(
            TrendRow(
                category=cat,
                recent_count=r,
                prev_count=prev,
                delta=delta,
                growth_ratio=growth_ratio,
                example_titles=examples,
            )
        )

    # Sort: highest delta first, then highest relative growth.
    rows.sort(key=lambda x: (x.delta, x.growth_ratio), reverse=True)
    return rows[:top_k]


def render_trends_report(rows: list[TrendRow], *, recent_days: int, previous_days: int) -> str:
    if not rows:
        return f"No rising categories found in the last {recent_days} days."

    lines: list[str] = [
        f"Trends (last {recent_days} days vs previous {previous_days} days):",
    ]
    for r in rows:
        growth_pct = int((r.growth_ratio - 1) * 100) if r.prev_count > 0 else 100
        lines.append(
            f"- {r.category}: {r.recent_count} recent / {r.prev_count} previous "
            f"(delta {r.delta}, growth ~{growth_pct}%)"
        )
        if r.example_titles:
            titles_str = "; ".join(t[:80] for t in r.example_titles)
            lines.append(f"  Examples: {titles_str}")

    return "\n".join(lines)
