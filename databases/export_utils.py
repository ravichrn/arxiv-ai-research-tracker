"""
Deterministic export helpers for turning paper metadata into BibTeX/CSV.

These helpers are intentionally stdlib-only and contain no LLM calls.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

_ARXIV_ID_RE = re.compile(r"/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\D|$)")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _escape_bibtex_value(value: str) -> str:
    # Minimal escaping for BibTeX.
    v = value.replace("{", r"\{").replace("}", r"\}").replace("&", r"\&").replace("%", r"\%")
    v = v.replace('"', "").strip()
    return _normalize_whitespace(v)


def _authors_to_bibtex(authors: str) -> str:
    authors = _normalize_whitespace(authors)
    if not authors:
        return ""
    # In this project, author strings are typically "A B, C D" (comma+space separated).
    # BibTeX prefers "A B and C D".
    parts = [p.strip() for p in authors.split(",") if p.strip()]
    return " and ".join(parts) if len(parts) > 1 else authors


def _infer_arxiv_id(paper: dict[str, Any]) -> str:
    arxiv_id = str(paper.get("arxiv_id", "")).strip()
    if arxiv_id:
        return arxiv_id
    url = str(paper.get("url", "")).strip()
    if url:
        m = _ARXIV_ID_RE.search(url)
        if m:
            return m.group(1)
    return ""


def _infer_year(paper: dict[str, Any]) -> str:
    published = str(paper.get("published", "")).strip()
    # Expect ISO 8601 (e.g. 2024-04-12T...).
    return published[:4] if len(published) >= 4 and published[:4].isdigit() else ""


def render_bibtex(papers: list[dict[str, Any]]) -> str:
    """Render BibTeX entries for a list of paper dicts."""
    entries: list[str] = []
    for paper in papers:
        arxiv_id = _infer_arxiv_id(paper)
        if not arxiv_id:
            # BibTeX entry key still needs something stable.
            arxiv_id = "unknown"

        key = "arxiv_" + arxiv_id.replace(".", "_").replace("/", "_").replace("-", "_")

        title = _escape_bibtex_value(str(paper.get("title", "")).strip())
        authors = _authors_to_bibtex(str(paper.get("authors", "")).strip())
        url = _escape_bibtex_value(str(paper.get("url", "")).strip())
        categories = _escape_bibtex_value(str(paper.get("categories", "")).strip())
        year = _infer_year(paper)

        primary_class = categories.split(",")[0].strip() if categories else ""

        # Note: we keep "abstract" out for deterministic metadata-only export.
        entry = [
            f"@article{{{key},",
            f"  title={{ {title} }},",
            f"  author={{ {authors} }},",
            f"  year={{ {year} }},",
            f"  url={{ {url} }},",
            f"  eprint={{ {arxiv_id} }},",
            "  archivePrefix={arXiv},",
            f"  primaryClass={{ {primary_class} }},",
            f"  note={{ {categories} }},",
            "}",
        ]
        entries.append("\n".join(entry))

    return "\n\n".join(entries) + "\n"


def render_csv(papers: list[dict[str, Any]]) -> str:
    """Render CSV for a list of paper dicts."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["arxiv_id", "url", "title", "authors", "categories", "published"])
    for paper in papers:
        writer.writerow(
            [
                _infer_arxiv_id(paper),
                str(paper.get("url", "")).strip(),
                str(paper.get("title", "")).strip(),
                str(paper.get("authors", "")).strip(),
                str(paper.get("categories", "")).strip(),
                str(paper.get("published", "")).strip(),
            ]
        )
    return buf.getvalue()
