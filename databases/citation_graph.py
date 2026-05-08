"""SQLite store for paper citation edges (lightweight Graph RAG).

Stores 1-hop reference and citation edges fetched from Semantic Scholar.
Each edge is (source_arxiv_id, cited_arxiv_id, direction) where direction
is "references" (outbound, papers this work cites) or "citations" (inbound,
papers that cite this work).

The `has_edges` guard prevents redundant S2 API calls — once edges are
fetched for a paper they are cached here indefinitely.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

_DB_DIR = Path(__file__).parent
DEFAULT_DB_PATH = _DB_DIR / "citation_graph.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Ensure the citation_edges table exists."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_edges (
                source_arxiv_id  TEXT NOT NULL,
                cited_arxiv_id   TEXT NOT NULL,
                direction        TEXT NOT NULL,
                title            TEXT,
                fetched_at       TEXT NOT NULL,
                PRIMARY KEY (source_arxiv_id, cited_arxiv_id, direction)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_edges(
    source_id: str,
    edges: list[dict],
    direction: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Insert citation edges, ignoring duplicates.

    Args:
        source_id: Base arXiv ID of the paper whose edges we fetched.
        edges: List of {"cited_arxiv_id": str, "title": str | None} dicts.
        direction: "references" or "citations".
    """
    if not edges:
        return
    now = datetime.now(UTC).isoformat()
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO citation_edges
                (source_arxiv_id, cited_arxiv_id, direction, title, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (source_id, e["cited_arxiv_id"], direction, e.get("title"), now)
                for e in edges
                if e.get("cited_arxiv_id")
            ],
        )
        conn.commit()
        _log.debug("[citation_graph] upserted %d %s edges for %s", len(edges), direction, source_id)
    finally:
        conn.close()


def get_edges(
    arxiv_id: str,
    direction: str,
    limit: int = 20,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Return stored edges for a paper.

    Returns list of {"cited_arxiv_id": str, "title": str | None} dicts,
    ordered by insertion order (ROWID), up to `limit` results.
    """
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT cited_arxiv_id, title
            FROM citation_edges
            WHERE source_arxiv_id = ? AND direction = ?
            ORDER BY ROWID
            LIMIT ?
            """,
            (arxiv_id, direction, limit),
        ).fetchall()
        return [{"cited_arxiv_id": r[0], "title": r[1]} for r in rows]
    finally:
        conn.close()


def has_edges(arxiv_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Return True if edges have already been fetched for this paper."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM citation_edges WHERE source_arxiv_id = ? LIMIT 1",
            (arxiv_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()
