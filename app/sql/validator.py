"""
SQL validation using sqlglot AST parsing.

Responsibility: answer one question — "is this SQL safe to execute?"
It never fixes or rewrites SQL. It only accepts or rejects.

Two checks are performed in order:
  1. Parse check  — does sqlglot parse the string without error?
  2. Safety check — is the top-level statement a SELECT or WITH?
                    Rejects any DDL (CREATE, DROP, ALTER) or DML (INSERT, UPDATE, DELETE).

Why sqlglot instead of regex?
  A regex like r"^SELECT" would miss:
    - WITH cte AS (...) SELECT ...   (valid — starts with WITH)
    - /* comment */ SELECT ...       (valid — comment before SELECT)
    - DROP TABLE; SELECT ...         (dangerous — multi-statement)
  sqlglot parses the full AST, so all these are handled correctly.

Why not execute and catch errors instead?
  DuckDB execution is slow (disk I/O) and has side effects.
  Validation should be a fast, pure in-memory check before any I/O.

Design decisions:
  - validate_sql() returns (bool, str | None) — the string is an error message
    when validation fails, None when it passes. This gives the pipeline a human-
    readable reason for rejection (useful for logging and lesson generation).
  - Multi-statement SQL (e.g., "DROP TABLE x; SELECT 1") is rejected — sqlglot
    parses all statements and we check only the first, but we also reject if more
    than one statement is present.
  - sqlglot dialect is left as default (generic). DuckDB SQL is close enough to
    ANSI SQL that generic parsing handles all our cases correctly.
"""

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError


# Statement types that are allowed through validation.
_ALLOWED_TYPES = (exp.Select, exp.With)


def validate_sql(sql: str) -> tuple[bool, str | None]:
    """
    Check whether a SQL string is safe to execute against DuckDB.

    Args:
        sql: The SQL string to validate (may contain markdown fences — these
             should be stripped by LLMClient before calling this function).

    Returns:
        (True, None)          if the SQL is valid and safe.
        (False, error_message) if the SQL is invalid or unsafe.

    Examples:
        >>> validate_sql("SELECT COUNT(*) FROM customers")
        (True, None)

        >>> validate_sql("DROP TABLE customers")
        (False, "Unsafe statement type: Drop. Only SELECT and WITH are allowed.")

        >>> validate_sql("SELET 1")
        (False, "SQL parse error: ...")

        >>> validate_sql("")
        (False, "SQL is empty.")
    """
    stripped = sql.strip()
    if not stripped:
        return False, "SQL is empty."

    # ── Parse check ───────────────────────────────────────────────────────────
    try:
        statements = sqlglot.parse(stripped)
    except ParseError as e:
        return False, f"SQL parse error: {e}"

    if not statements:
        return False, "SQL produced no parse result."

    # ── Multi-statement check ─────────────────────────────────────────────────
    # Reject "SELECT 1; DROP TABLE x" — the LLM should never produce this,
    # but we guard against it explicitly.
    if len(statements) > 1:
        return False, f"Multiple statements detected ({len(statements)}). Only a single SELECT or WITH is allowed."

    # ── Safety check ──────────────────────────────────────────────────────────
    top = statements[0]
    if not isinstance(top, _ALLOWED_TYPES):
        statement_type = type(top).__name__
        return False, (
            f"Unsafe statement type: {statement_type}. "
            f"Only SELECT and WITH are allowed."
        )

    return True, None
