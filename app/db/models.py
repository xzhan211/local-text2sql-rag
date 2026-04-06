"""
Database models for the text-to-SQL RAG system.

Two concerns are handled here:

1. Pydantic models — used in Python code for type-safe data passing.
   These are NOT ORM models; they don't talk to the DB directly.
   Think of them as typed dicts with validation.

2. CREATE_TABLES_SQL — the raw DDL that initializes the SQLite schema.
   Run once at startup via sqlite_client.init_db().

The two-index design maps to two tables here:
  - nlq_sql_pairs  → backs FAISS Index 1 (kb_nlq_sql_pairs)
  - lessons        → backs FAISS Index 2 (kb_lessons_learned)

The `faiss_index_id` column is the bridge: after a vector search returns
an integer ID from FAISS, you use that ID to fetch the full record from SQLite.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Pydantic models ──────────────────────────────────────────────────────────

class NLQSQLPair(BaseModel):
    """One training example: a natural language question paired with gold SQL."""
    id: Optional[int] = None
    nlq: str                               # "How many customers are in Germany?"
    sql: str                               # "SELECT COUNT(*) FROM customers WHERE country = 'Germany'"
    faiss_index_id: Optional[int] = None   # position in FAISS Index 1
    created_at: Optional[datetime] = None


class Lesson(BaseModel):
    """
    A structured, reusable lesson learned from a SQL generation failure.

    Design principle: lessons must be GENERALIZABLE, not query-specific.
    A good lesson captures a pattern ("when filtering by month, use DATE_TRUNC")
    not a fact ("query #47 failed because of X").

    The `example` field holds a JSON string with the failure that triggered this lesson.
    """
    id: Optional[int] = None
    title: str           = Field(..., description="Short name for the lesson, e.g. 'Use DATE_TRUNC for month filtering'")
    trigger: str         = Field(..., description="What NLQ pattern triggers this lesson")
    diagnosis: str       = Field(..., description="Why the SQL was wrong")
    fix_rule: str        = Field(..., description="The actionable rule to apply next time")
    error_category: str  = Field(..., description="Category: date_handling | join_logic | aggregation | filter | other")
    example: str         = Field(..., description="JSON: {nlq, pred_sql, gold_sql}")
    faiss_index_id: Optional[int] = None   # position in FAISS Index 2
    created_at: Optional[datetime] = None


class TrainingRun(BaseModel):
    """Tracks one full training pipeline execution (CSV upload → eval → metrics)."""
    id: Optional[int] = None
    status: str = "running"         # "running" | "done" | "failed"
    csv_path: str = ""
    metrics: Optional[str] = None   # JSON-encoded MetricsReport
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class EvalResult(BaseModel):
    """
    One row per evaluation item. Records both attempts (pass@1, pass@2).

    pass1: did the first SQL attempt match gold?
    pass2: did the lesson-augmented retry match gold? (only set if pass1=False)
    hard_failure: both attempts failed
    """
    id: Optional[int] = None
    training_run_id: int
    nlq: str
    gold_sql: str
    pred_sql_1: Optional[str] = None
    pred_sql_2: Optional[str] = None
    pass1: bool = False
    pass2: bool = False
    hard_failure: bool = False
    used_lesson_id: Optional[int] = None   # which lesson (if any) was used on retry
    created_at: Optional[datetime] = None


# ── DDL ──────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS nlq_sql_pairs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nlq             TEXT    NOT NULL,
    sql             TEXT    NOT NULL,
    faiss_index_id  INTEGER,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    trigger         TEXT    NOT NULL,
    diagnosis       TEXT    NOT NULL,
    fix_rule        TEXT    NOT NULL,
    error_category  TEXT    NOT NULL,
    example         TEXT    NOT NULL,
    faiss_index_id  INTEGER,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    status       TEXT    NOT NULL DEFAULT 'running',
    csv_path     TEXT    NOT NULL DEFAULT '',
    metrics      TEXT,
    started_at   TEXT    DEFAULT (datetime('now')),
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id  INTEGER NOT NULL,
    nlq              TEXT    NOT NULL,
    gold_sql         TEXT    NOT NULL,
    pred_sql_1       TEXT,
    pred_sql_2       TEXT,
    pass1            INTEGER NOT NULL DEFAULT 0,
    pass2            INTEGER NOT NULL DEFAULT 0,
    hard_failure     INTEGER NOT NULL DEFAULT 0,
    used_lesson_id   INTEGER,
    created_at       TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id),
    FOREIGN KEY (used_lesson_id)  REFERENCES lessons(id)
);
"""
