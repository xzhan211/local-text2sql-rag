"""
SQL generation orchestrator.

Responsibility: given an NLQ + retrieved context, build the right prompt and
call the LLM client to get SQL back. This is the only module that knows about
both prompts.py and claude_client.py.

Two call patterns:

  First attempt (no lessons, temperature=0.0):
      sql = generate_sql(client, nlq, examples)

  Retry attempt (with lessons, temperature=0.3):
      sql = generate_sql(client, nlq, examples, lessons=lessons)

Design decisions:
  - generate_sql() is a plain function, not a method. It takes an LLMClient as
    an argument (dependency injection) so tests can pass a mock client without
    patching module-level state.
  - The caller decides whether to retry and what lessons to pass. generate_sql()
    has no internal retry loop — it executes exactly one LLM call per invocation.
    Retry logic lives in the eval loop (Phase 4) and inference pipeline (Phase 6).
  - temperature is derived from whether lessons are present, not passed by the
    caller. This enforces the convention: first attempt = 0.0, retry = 0.3.
    Callers cannot accidentally pass the wrong temperature.
  - LLMError propagates upward unchanged. The pipeline layer decides how to handle
    API failures (log, mark hard_failure, etc.).
"""

from app.llm.claude_client import LLMClient
from app.llm.prompts import build_sql_gen_prompt, build_nlq_gen_prompt
from app.core.config import settings


def generate_sql(
    client: LLMClient,
    nlq: str,
    examples: list[dict],
    lessons: list[dict] | None = None,
) -> str:
    """
    Generate a SQL query for a natural language question.

    Builds the appropriate prompt (with or without lessons), calls the LLM,
    and returns the raw SQL string. The SQL is not validated or executed here —
    that is the responsibility of validator.py and the pipeline.

    Args:
        client:   An instantiated LLMClient. Shared across calls in a pipeline run.
        nlq:      The natural language question to answer with SQL.
        examples: Retrieved few-shot examples from KBManager.search_examples().
                  List of dicts: [{nlq, sql, score}, ...].
        lessons:  Optional lessons from KBManager.search_lessons().
                  List of dicts: [{title, trigger, fix_rule, error_category, ...}, ...].
                  Pass None (default) for first attempt.
                  Pass a list (even empty) to signal retry — raises temperature to 0.3.

    Returns:
        SQL string, stripped of markdown fences. Not yet validated.

    Raises:
        LLMError: if the Claude API call fails.
    """
    # Temperature convention: deterministic on first attempt, slight variation on retry.
    # This is intentional — retry needs creative variation to escape the first attempt's
    # failure mode, but not so much randomness that it ignores the lesson guidance.
    temperature = settings.llm_retry_temperature if lessons is not None else settings.llm_temperature

    system, human = build_sql_gen_prompt(nlq, examples, lessons)
    return client.complete(system, human, temperature=temperature)


def generate_nlq(client: LLMClient, sql: str) -> str:
    """
    Generate a natural language question from a SQL query.

    Used by the training pipeline when the input CSV has a SQL column but
    no NLQ column (or empty NLQ). The generated question is stored in SQLite
    alongside the SQL and used as the retrieval key in Index 1.

    Args:
        client: An instantiated LLMClient.
        sql:    The SQL query to reverse into a natural language question.

    Returns:
        A plain English question string, stripped of fences/whitespace.

    Raises:
        LLMError: if the Claude API call fails.
    """
    system, human = build_nlq_gen_prompt(sql)
    return client.complete(system, human, temperature=0.0)
