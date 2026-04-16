import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from databases.export_utils import render_bibtex, render_csv
from databases.saved_metadata import (
    get_tags_and_note_for_title,
    get_tags_for_titles,
    init_db,
    set_note,
    set_tags,
)
from databases.trends_utils import compute_category_trends, render_trends_report


def test_render_csv_contains_expected_fields() -> None:
    papers = [
        {
            "arxiv_id": "2301.12345v2",
            "url": "https://arxiv.org/abs/2301.12345v2",
            "title": "Attention & Transformers",
            "authors": "Alice Smith, Bob Jones",
            "categories": "cs.CL, cs.LG",
            "published": "2024-04-12T00:00:00+00:00",
        }
    ]
    csv_text = render_csv(papers)
    assert csv_text.startswith("arxiv_id,url,title,authors,categories,published")
    assert "2301.12345v2" in csv_text
    assert "Attention & Transformers" in csv_text


def test_render_bibtex_contains_eprint_and_escaped_title() -> None:
    papers = [
        {
            "arxiv_id": "2301.12345v2",
            "url": "https://arxiv.org/abs/2301.12345v2",
            "title": "Attention & Transformers",
            "authors": "Alice Smith, Bob Jones",
            "categories": "cs.CL, cs.LG",
            "published": "2024-04-12T00:00:00+00:00",
        }
    ]
    bib = render_bibtex(papers)
    assert "@article{" in bib
    assert "eprint={ 2301.12345v2 }" in bib
    # We escape `&` for BibTeX.
    assert "Attention \\& Transformers" in bib


def test_saved_metadata_roundtrip_tags_and_note() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "meta.db"
        init_db(db_path)

        set_tags("My Paper Title", ["LLM", "NLP", "llm"], db_path=db_path)
        set_note("My Paper Title", "Important note", db_path=db_path)

        tags, note = get_tags_and_note_for_title("My Paper Title", db_path=db_path)
        assert note == "Important note"
        # Tags are lowercased and deduped, preserving first-seen order.
        assert tags == ["llm", "nlp"]

        tags_map = get_tags_for_titles(["My Paper Title"], db_path=db_path)
        assert tags_map  # not empty


def test_compute_category_trends_rising_categories() -> None:
    now = datetime(2026, 4, 16, tzinfo=UTC)
    recent_days = 7
    previous_days = 7

    papers = [
        # Category A: rises from 1 -> 3
        {"published": (now - timedelta(days=12)).isoformat(), "categories": "cs.AI", "title": "A1"},
        {"published": (now - timedelta(days=6)).isoformat(), "categories": "cs.AI", "title": "A2"},
        {"published": (now - timedelta(days=2)).isoformat(), "categories": "cs.AI", "title": "A3"},
        # Category B: rises from 0 -> 1 (should still show up)
        {"published": (now - timedelta(days=3)).isoformat(), "categories": "cs.LG", "title": "B1"},
    ]

    rows = compute_category_trends(
        papers,
        now=now,
        recent_days=recent_days,
        previous_days=previous_days,
        top_k=10,
        examples_per_category=2,
    )

    categories = {r.category for r in rows}
    assert "cs.AI" in categories
    assert "cs.LG" in categories

    # Ensure delta for cs.AI is positive.
    ai_row = next(r for r in rows if r.category == "cs.AI")
    assert ai_row.delta >= 1

    report = render_trends_report(rows, recent_days=recent_days, previous_days=previous_days)
    assert "Trends (last" in report
