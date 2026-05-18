"""Unit tests for databases/citation_graph.py.

All tests use a temporary SQLite database (tmp_path) so they never touch the
real citation_graph.db and can run fully offline without any API keys.
"""

from __future__ import annotations

from pathlib import Path

from databases.citation_graph import (
    _base_id,
    get_edges,
    has_edges,
    init_db,
    upsert_edges,
)

# ── _base_id ──────────────────────────────────────────────────────────────────


def test_base_id_strips_version():
    assert _base_id("2301.12345v2") == "2301.12345"


def test_base_id_no_version_unchanged():
    assert _base_id("2301.12345") == "2301.12345"


def test_base_id_strips_high_version():
    assert _base_id("2301.12345v10") == "2301.12345"


# ── init_db ───────────────────────────────────────────────────────────────────


def test_init_db_creates_table(tmp_path: Path):
    db = tmp_path / "cg.db"
    init_db(db)
    import sqlite3

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='citation_edges'"
        ).fetchone()
    assert row is not None


# ── upsert_edges / get_edges / has_edges ─────────────────────────────────────

_EDGES = [
    {"cited_arxiv_id": "2101.00001", "title": "Paper A"},
    {"cited_arxiv_id": "2101.00002", "title": "Paper B"},
]


def test_has_edges_false_on_empty_db(tmp_path: Path):
    db = tmp_path / "cg.db"
    init_db(db)
    assert has_edges("2301.12345", db) is False


def test_upsert_and_has_edges(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", _EDGES, "references", db)
    assert has_edges("2301.12345", db) is True


def test_get_edges_returns_inserted_rows(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", _EDGES, "references", db)
    rows = get_edges("2301.12345", "references", db_path=db)
    assert len(rows) == 2
    ids = {r["cited_arxiv_id"] for r in rows}
    assert ids == {"2101.00001", "2101.00002"}


def test_get_edges_empty_for_unknown_paper(tmp_path: Path):
    db = tmp_path / "cg.db"
    init_db(db)
    assert get_edges("9999.99999", "references", db_path=db) == []


def test_upsert_edges_ignores_duplicates(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", _EDGES, "references", db)
    upsert_edges("2301.12345", _EDGES, "references", db)  # second call — INSERT OR IGNORE
    rows = get_edges("2301.12345", "references", db_path=db)
    assert len(rows) == 2  # no duplicates


def test_get_edges_direction_filter(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", _EDGES, "references", db)
    upsert_edges(
        "2301.12345", [{"cited_arxiv_id": "2101.00099", "title": "Citer"}], "citations", db
    )
    refs = get_edges("2301.12345", "references", db_path=db)
    cits = get_edges("2301.12345", "citations", db_path=db)
    assert len(refs) == 2
    assert len(cits) == 1


def test_get_edges_limit(tmp_path: Path):
    db = tmp_path / "cg.db"
    many = [{"cited_arxiv_id": f"2101.{i:05d}", "title": None} for i in range(10)]
    upsert_edges("2301.12345", many, "references", db)
    rows = get_edges("2301.12345", "references", limit=3, db_path=db)
    assert len(rows) == 3


# ── version-string normalization (public API entry points) ────────────────────


def test_has_edges_normalizes_versioned_id(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345v2", _EDGES, "references", db)
    # Both versioned and base ID should resolve to the same cached entry.
    assert has_edges("2301.12345", db) is True
    assert has_edges("2301.12345v3", db) is True


def test_upsert_with_base_id_found_by_versioned(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", _EDGES, "references", db)
    rows = get_edges("2301.12345v2", "references", db_path=db)
    assert len(rows) == 2


def test_upsert_edges_empty_list_is_noop(tmp_path: Path):
    db = tmp_path / "cg.db"
    upsert_edges("2301.12345", [], "references", db)
    assert has_edges("2301.12345", db) is False
