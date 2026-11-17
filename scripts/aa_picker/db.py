"""SQLite schema and helpers for Artificial Analysis model picker."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "aa_picker.db"

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    summary TEXT NOT NULL,
    skills_json TEXT NOT NULL,
    knowledge_json TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    featured INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    release_slug TEXT,
    release_name TEXT,
    name TEXT NOT NULL,
    short_name TEXT,
    effort_suffix TEXT,
    deprecated INTEGER NOT NULL DEFAULT 0,
    is_reasoning INTEGER NOT NULL DEFAULT 0,
    is_open_weights INTEGER NOT NULL DEFAULT 0,
    param_class TEXT,
    price_class TEXT,
    creator_name TEXT,
    context_window_tokens INTEGER,
    intelligence_index REAL,
    omniscience REAL,
    price_1m_input REAL,
    price_1m_output REAL,
    cache_hit_price REAL,
    cache_write_price REAL,
    output_speed REAL,
    time_to_first_token REAL,
    time_to_first_answer_token REAL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_release ON models(release_slug);

CREATE TABLE IF NOT EXISTS model_scores (
    model_id TEXT NOT NULL,
    benchmark_slug TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (model_id, benchmark_slug),
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_scores_bench ON model_scores(benchmark_slug);

CREATE TABLE IF NOT EXISTS provider_endpoints (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    host_api_id TEXT,
    host_name TEXT NOT NULL,
    host_slug TEXT NOT NULL,
    release_slug TEXT,
    matched_model_id TEXT,
    is_open_weights INTEGER NOT NULL DEFAULT 0,
    deprecated INTEGER NOT NULL DEFAULT 0,
    reasoning_model INTEGER NOT NULL DEFAULT 0,
    size_class TEXT,
    creator_name TEXT,
    context_window_tokens INTEGER,
    function_calling INTEGER,
    json_mode INTEGER,
    openai_compatible INTEGER,
    price_class TEXT,
    price_1m_input REAL,
    price_1m_output REAL,
    cache_hit_price REAL,
    cache_write_price REAL,
    output_speed REAL,
    time_to_first_token REAL,
    time_to_first_answer_token REAL,
    endpoint_error_rate REAL,
    intelligence_index REAL,
    omniscience REAL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_endpoints_release ON provider_endpoints(release_slug);
CREATE INDEX IF NOT EXISTS idx_endpoints_host ON provider_endpoints(host_slug);

CREATE TABLE IF NOT EXISTS endpoint_scores (
    endpoint_id TEXT NOT NULL,
    benchmark_slug TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (endpoint_id, benchmark_slug),
    FOREIGN KEY (endpoint_id) REFERENCES provider_endpoints(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_endpoint_scores_bench ON endpoint_scores(benchmark_slug);
"""


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    if db_path:
        return Path(db_path)
    return Path(os.environ.get("AA_PICKER_DB", DEFAULT_DB_PATH))


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> Path:
    path = resolve_db_path(db_path)
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO meta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, now),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def get_db_age_hours(conn: sqlite3.Connection) -> float | None:
    val = get_meta(conn, "last_updated_utc")
    if not val:
        row = conn.execute("SELECT MAX(updated_at) AS mx FROM models").fetchone()
        val = str(row["mx"]) if row and row["mx"] else None
    if not val:
        return None
    try:
        s = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return round(max(0.0, delta.total_seconds() / 3600.0), 2)
    except Exception:
        return None
