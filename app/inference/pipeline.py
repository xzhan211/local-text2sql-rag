"""
Inference pipeline.

Takes a natural language question, retrieves relevant examples and lessons,
generates SQL, and returns the best result with a confidence score.

Pipeline steps:
  1. Retrieve top-k examples from Index 1 (kb.search_examples)
  2. Compute confidence = avg cosine similarity of retrieved examples
  3. Generate SQL (attempt 1, temperature=0.0)
  4. Validate SQL
  5. If confidence < threshold OR invalid SQL:
       retrieve lessons from Index 2 (kb.search_lessons)
       retry with lesson-augmented prompt (temperature=0.3)
  6. Return {sql, confidence, used_lesson}

Design decisions:
  - query(nlq, client=None, kb=None): None creates defaults. FastAPI route
    passes None; tests inject mocks. Same pattern as training pipeline.
  - Confidence is computed from retrieval similarity only — inference never
    knows the gold SQL, so similarity is the only proxy for "will this work?"
  - Retry trigger: low confidence OR invalid SQL (not just low confidence).
    An invalid SQL is always worth retrying regardless of confidence score.
  - Inference never raises on SQL generation failure — LLMError returns an
    empty string with confidence=0.0 and used_lesson=False. The API layer
    decides how to handle it.
  - used_lesson=True only when lessons were actually retrieved (non-empty)
    and used in the retry prompt. A retry with no lessons still sets
    used_lesson=False — the flag means "a lesson influenced this result."
  - The returned SQL is always the most recent attempt: attempt 2 if a retry
    happened, otherwise attempt 1. Even if attempt 2 is also invalid, it is
    returned — the pipeline does not compare attempts.
"""

from app.core.config import settings
from app.llm.claude_client import LLMClient, LLMError
from app.sql.generator import generate_sql
from app.sql.validator import validate_sql
from app.vectorstore.kb_manager import KBManager


def query(
    nlq: str,
    client: LLMClient | None = None,
    kb: KBManager | None = None,
) -> dict:
    """
    Run the inference pipeline for a natural language question.

    Args:
        nlq:    The natural language question to answer with SQL.
        client: LLMClient instance. If None, a new one is created from settings.
        kb:     KBManager instance. If None, a new one is created and indexes loaded.

    Returns:
        dict with keys:
          sql         (str)  — generated SQL, empty string on total failure
          confidence  (float)— avg cosine similarity of top-k retrieved examples
          used_lesson (bool) — True if a lesson from Index 2 influenced the result
    """
    if client is None:
        client = LLMClient()
    if kb is None:
        kb = KBManager()
        kb.load_indexes()

    # ── Step 1: Retrieve examples + compute confidence ─────────────────────────
    examples = kb.search_examples(nlq)
    confidence = _avg_score(examples)

    # ── Step 2: Attempt 1 ──────────────────────────────────────────────────────
    try:
        sql = generate_sql(client, nlq, examples)
    except LLMError:
        return {"sql": "", "confidence": 0.0, "used_lesson": False}

    valid, _ = validate_sql(sql)

    # ── Step 3: Retry if low confidence or invalid SQL ─────────────────────────
    if confidence < settings.confidence_threshold or not valid:
        lessons = kb.search_lessons(nlq)
        used_lesson = len(lessons) > 0

        try:
            sql = generate_sql(client, nlq, examples, lessons=lessons)
        except LLMError:
            pass  # keep attempt 1 result

        return {"sql": sql, "confidence": confidence, "used_lesson": used_lesson}

    return {"sql": sql, "confidence": confidence, "used_lesson": False}


# ── Private helpers ────────────────────────────────────────────────────────────

def _avg_score(results: list[dict]) -> float:
    """Return the average 'score' from a list of search results. 0.0 if empty."""
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)
