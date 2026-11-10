"""SQLite persistence for documents and scan results.

Uses the standard-library ``sqlite3`` driver with:
  * WAL journaling, so autosaves never block reads from other devices,
  * parameterised queries only (no string-built SQL),
  * optimistic concurrency via a per-document ``version`` counter, so two
    devices editing the same document can't silently overwrite each other.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL DEFAULT 'Untitled document',
    content     TEXT    NOT NULL DEFAULT '',   -- Quill Delta (JSON)
    plain_text  TEXT    NOT NULL DEFAULT '',
    word_count  INTEGER NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text_hash    TEXT    NOT NULL,
    result_json  TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_document ON scans(document_id, id DESC);
"""

DOC_SUMMARY_COLUMNS = "id, title, word_count, version, created_at, updated_at"
DOC_FULL_COLUMNS = DOC_SUMMARY_COLUMNS + ", content, plain_text"


class VersionConflict(Exception):
    """Raised when a save is based on an out-of-date document version."""

    def __init__(self, current: dict[str, Any]):
        super().__init__("Document was modified elsewhere")
        self.current = current


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def count_words(text: str) -> int:
    return len(text.split())


def text_fingerprint(text: str) -> str:
    """Whitespace-insensitive hash, used to tell whether a scan is stale."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    # -- connection handling -------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    # -- documents -------------------------------------------------------------
    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {DOC_SUMMARY_COLUMNS} FROM documents ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(f"SELECT {DOC_FULL_COLUMNS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def create_document(self, title: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            doc_id = cur.lastrowid
        doc = self.get_document(doc_id)
        assert doc is not None
        return doc

    def update_document(
        self,
        doc_id: int,
        *,
        title: str,
        content: str,
        plain_text: str,
        base_version: int,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Save a document. Returns the new summary, or None if it doesn't exist.

        Raises VersionConflict if ``base_version`` is stale and ``force`` is False.
        """
        now = utc_now()
        words = count_words(plain_text)
        with self.connect() as conn:
            if force:
                cur = conn.execute(
                    """UPDATE documents
                       SET title = ?, content = ?, plain_text = ?, word_count = ?,
                           version = version + 1, updated_at = ?
                       WHERE id = ?""",
                    (title, content, plain_text, words, now, doc_id),
                )
            else:
                cur = conn.execute(
                    """UPDATE documents
                       SET title = ?, content = ?, plain_text = ?, word_count = ?,
                           version = version + 1, updated_at = ?
                       WHERE id = ? AND version = ?""",
                    (title, content, plain_text, words, now, doc_id, base_version),
                )
            if cur.rowcount == 0:
                row = conn.execute(f"SELECT {DOC_FULL_COLUMNS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
                if row is None:
                    return None
                raise VersionConflict(dict(row))
            row = conn.execute(f"SELECT {DOC_SUMMARY_COLUMNS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row)

    def delete_document(self, doc_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0

    # -- scans -------------------------------------------------------------------
    def save_scan(self, doc_id: int, text: str, result: dict[str, Any]) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO scans (document_id, text_hash, result_json, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, text_fingerprint(text), json.dumps(result), utc_now()),
            )
        return True

    def latest_scan(self, doc_id: int, current_text: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT text_hash, result_json, created_at FROM scans WHERE document_id = ? ORDER BY id DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "result": json.loads(row["result_json"]),
            "scanned_at": row["created_at"],
            "is_stale": row["text_hash"] != text_fingerprint(current_text),
        }
