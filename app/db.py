from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DATA_DIR, DB_PATH, PAGE_DIR, UPLOAD_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                content_type TEXT,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                page_title TEXT,
                image_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ready',
                error TEXT,
                raw_output TEXT,
                kv_pairs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS page_extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER NOT NULL,
                query_key TEXT NOT NULL,
                value_text TEXT,
                bbox_json TEXT,
                ref_text TEXT,
                confidence REAL,
                raw_output TEXT,
                status TEXT NOT NULL DEFAULT 'processed',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            )
            """
        )
        _migrate_legacy_documents(db)
        _migrate_page_titles(db)
        db.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA foreign_keys = ON")
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def row_to_document(row: sqlite3.Row) -> dict[str, Any]:
    doc = dict(row)
    return doc


def row_to_page(row: sqlite3.Row) -> dict[str, Any]:
    page = dict(row)
    page["kv_pairs"] = json.loads(page.pop("kv_pairs_json") or "[]")
    return page


def row_to_extraction(row: sqlite3.Row) -> dict[str, Any]:
    extraction = dict(row)
    bbox = extraction.pop("bbox_json", None)
    extraction["bbox"] = json.loads(bbox) if bbox else None
    return extraction


def _migrate_legacy_documents(db: sqlite3.Connection) -> None:
    columns = [row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()]
    if "raw_output" not in columns and "kv_pairs_json" not in columns:
        return

    db.execute("ALTER TABLE documents RENAME TO documents_legacy")
    db.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            content_type TEXT,
            size_bytes INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        INSERT INTO documents
            (id, filename, stored_path, content_type, size_bytes, status, error, created_at, updated_at)
        SELECT id, filename, stored_path, content_type, size_bytes, status, error, created_at, updated_at
        FROM documents_legacy
        """
    )
    legacy_rows = db.execute("SELECT * FROM documents_legacy").fetchall()
    for row in legacy_rows:
        db.execute(
            """
            INSERT INTO pages
                (document_id, page_number, page_title, image_path, status, error, raw_output, kv_pairs_json, created_at, updated_at)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                Path(str(row["stored_path"])).name,
                row["stored_path"],
                row["status"],
                row["error"],
                row["raw_output"] if "raw_output" in row.keys() else None,
                row["kv_pairs_json"] if "kv_pairs_json" in row.keys() else "[]",
                row["created_at"],
                row["updated_at"],
            ),
        )
    db.execute("DROP TABLE documents_legacy")


def _migrate_page_titles(db: sqlite3.Connection) -> None:
    columns = [row["name"] for row in db.execute("PRAGMA table_info(pages)").fetchall()]
    if "page_title" not in columns:
        db.execute("ALTER TABLE pages ADD COLUMN page_title TEXT")
    db.execute(
        """
        UPDATE pages
        SET page_title = 'Page ' || page_number
        WHERE page_title IS NULL OR trim(page_title) = ''
        """
    )
