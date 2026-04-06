"""
SQLite connection management and CRUD helpers.

Design decisions:
  - No ORM (SQLAlchemy). Raw sqlite3 keeps the dependency list short and
    makes the SQL transparent for learning purposes.
  - Context manager `db_conn()` ensures connections are always closed and
    transactions are committed or rolled back — no leaking handles.
  - `conn.row_factory = sqlite3.Row` lets you access columns by name:
      row["nlq"]  instead of  row[0]
  - Foreign keys are disabled by default in SQLite. We enable them per-connection
    with PRAGMA foreign_keys = ON.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator

from app.core.config import settings
from app.db.models import CREATE_TABLES_SQL


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with row dict access and FK enforcement."""
    conn = sqlite3.connect(str(settings.sqlite_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for a SQLite connection.

    Usage:
        with db_conn() as conn:
            conn.execute("INSERT INTO ...")

    Commits on success, rolls back on any exception, always closes.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create all tables if they don't exist.

    Safe to call multiple times (all statements use CREATE TABLE IF NOT EXISTS).
    Also creates the data/ and data/indexes/ directories if they don't exist.

    Call this once at application startup (in app/main.py).
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    with db_conn() as conn:
        conn.executescript(CREATE_TABLES_SQL)


# ── CRUD helpers ─────────────────────────────────────────────────────────────
# These are thin wrappers. Business logic (what to insert/query) lives in
# the pipeline and evaluation modules, not here.

def insert_nlq_sql_pair(nlq: str, sql: str, faiss_index_id: int | None = None) -> int:
    """Insert one NLQ+SQL pair. Returns the new row's id."""
    with db_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO nlq_sql_pairs (nlq, sql, faiss_index_id) VALUES (?, ?, ?)",
            (nlq, sql, faiss_index_id),
        )
        return cursor.lastrowid


def update_pair_faiss_id(row_id: int, faiss_index_id: int) -> None:
    """Set the FAISS index ID for an existing pair (called after vector is added to FAISS)."""
    with db_conn() as conn:
        conn.execute(
            "UPDATE nlq_sql_pairs SET faiss_index_id = ? WHERE id = ?",
            (faiss_index_id, row_id),
        )


def get_pair_by_faiss_id(faiss_index_id: int) -> sqlite3.Row | None:
    """Fetch a pair by its FAISS vector position. Used after a similarity search."""
    with db_conn() as conn:
        return conn.execute(
            "SELECT * FROM nlq_sql_pairs WHERE faiss_index_id = ?",
            (faiss_index_id,),
        ).fetchone()


def insert_lesson(
    title: str,
    trigger: str,
    diagnosis: str,
    fix_rule: str,
    error_category: str,
    example: str,
    faiss_index_id: int | None = None,
) -> int:
    """Insert one lesson. Returns the new row's id."""
    with db_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO lessons (title, trigger, diagnosis, fix_rule, error_category, example, faiss_index_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (title, trigger, diagnosis, fix_rule, error_category, example, faiss_index_id),
        )
        return cursor.lastrowid


def update_lesson_faiss_id(row_id: int, faiss_index_id: int) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE lessons SET faiss_index_id = ? WHERE id = ?",
            (faiss_index_id, row_id),
        )


def get_lesson_by_faiss_id(faiss_index_id: int) -> sqlite3.Row | None:
    with db_conn() as conn:
        return conn.execute(
            "SELECT * FROM lessons WHERE faiss_index_id = ?",
            (faiss_index_id,),
        ).fetchone()


def insert_training_run(csv_path: str) -> int:
    """Create a new training run record in 'running' status. Returns run id."""
    with db_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO training_runs (status, csv_path) VALUES ('running', ?)",
            (csv_path,),
        )
        return cursor.lastrowid


def finish_training_run(run_id: int, metrics_json: str, status: str = "done") -> None:
    with db_conn() as conn:
        conn.execute(
            """UPDATE training_runs
               SET status = ?, metrics = ?, finished_at = datetime('now')
               WHERE id = ?""",
            (status, metrics_json, run_id),
        )


def insert_eval_result(
    training_run_id: int,
    nlq: str,
    gold_sql: str,
    pred_sql_1: str | None,
    pred_sql_2: str | None,
    pass1: bool,
    pass2: bool,
    hard_failure: bool,
    used_lesson_id: int | None = None,
) -> int:
    with db_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO eval_results
               (training_run_id, nlq, gold_sql, pred_sql_1, pred_sql_2,
                pass1, pass2, hard_failure, used_lesson_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                training_run_id, nlq, gold_sql, pred_sql_1, pred_sql_2,
                int(pass1), int(pass2), int(hard_failure), used_lesson_id,
            ),
        )
        return cursor.lastrowid


def get_training_run(run_id: int) -> sqlite3.Row | None:
    with db_conn() as conn:
        return conn.execute(
            "SELECT * FROM training_runs WHERE id = ?", (run_id,)
        ).fetchone()
