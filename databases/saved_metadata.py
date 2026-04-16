"""
SQLite side-table for user-controlled saved-paper metadata (tags and notes).

This is intentionally stdlib-only (sqlite3 + json) and keyed by a normalized
paper title so it stays stable across arXiv version strings.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

_DB_DIR = Path(__file__).parent
DEFAULT_DB_PATH = _DB_DIR / "saved_metadata.db"


_WS_RE = re.compile(r"\s+")


def normalize_title_key(title: str) -> str:
    """Normalize a paper title into a stable lookup key."""
    # Case-insensitive matching + collapse whitespace.
    return _WS_RE.sub(" ", title.strip().lower())


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Ensure the saved metadata tables exist."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_metadata (
                title_key TEXT PRIMARY KEY,
                tags_json TEXT NOT NULL DEFAULT '[]',
                note TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x2 = x.strip()
        if not x2:
            continue
        if x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out


def set_tags(title: str, tags: list[str], db_path: Path = DEFAULT_DB_PATH) -> None:
    """Replace tags for the given paper title."""
    init_db(db_path)
    title_key = normalize_title_key(title)
    tags_norm = [t.lower().strip() for t in tags if t and t.strip()]
    tags_norm = _dedupe_preserve_order(tags_norm)
    tags_json = json.dumps(tags_norm, ensure_ascii=False)

    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO paper_metadata (title_key, tags_json, note)
            VALUES (?, ?, COALESCE((SELECT note FROM paper_metadata WHERE title_key = ?), NULL))
            ON CONFLICT(title_key) DO UPDATE SET tags_json = excluded.tags_json
            """,
            (title_key, tags_json, title_key),
        )
        conn.commit()
    finally:
        conn.close()


def set_note(title: str, note: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Replace note for the given paper title."""
    init_db(db_path)
    title_key = normalize_title_key(title)
    note_str = note.strip() if note is not None else ""

    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO paper_metadata (title_key, tags_json, note)
            VALUES (
                ?,
                COALESCE(
                    (SELECT tags_json FROM paper_metadata WHERE title_key = ?),
                    '[]'
                ),
                ?
            )
            ON CONFLICT(title_key) DO UPDATE SET note = excluded.note
            """,
            (title_key, title_key, note_str),
        )
        conn.commit()
    finally:
        conn.close()


def get_tags_for_titles(titles: list[str], db_path: Path = DEFAULT_DB_PATH) -> dict[str, list[str]]:
    """Get tags for multiple titles (returned keys are normalized title keys)."""
    if not titles:
        return {}
    init_db(db_path)

    title_keys = [normalize_title_key(t) for t in titles if t and t.strip()]
    if not title_keys:
        return {}

    # sqlite IN clauses work with a tuple of params.
    placeholders = ",".join(["?"] * len(title_keys))
    conn = _connect(db_path)
    try:
        # `placeholders` is derived solely from `len(title_keys)`; it never contains
        # user input. Still, we construct the query string to build the correct
        # number of `?` placeholders, so we suppress the S608 warning here.
        query = (
            "SELECT title_key, tags_json FROM paper_metadata WHERE title_key IN ("  # noqa: S608
            + placeholders
            + ")"
        )
        rows = conn.execute(query, tuple(title_keys)).fetchall()
    finally:
        conn.close()

    out: dict[str, list[str]] = {}
    for title_key, tags_json in rows:
        try:
            tags = json.loads(tags_json) if tags_json else []
            if isinstance(tags, list):
                out[str(title_key)] = [str(x) for x in tags]
        except Exception:
            out[str(title_key)] = []
    return out


def get_tags_and_note_for_title(
    title: str, db_path: Path = DEFAULT_DB_PATH
) -> tuple[list[str], str]:
    """Return (tags, note) for a single title."""
    tags_map = get_tags_for_titles([title], db_path=db_path)
    title_key = normalize_title_key(title)
    tags = tags_map.get(title_key, [])

    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT note FROM paper_metadata WHERE title_key = ?",
            (title_key,),
        ).fetchone()
    finally:
        conn.close()

    note = str(row[0]) if row and row[0] is not None else ""
    return tags, note
