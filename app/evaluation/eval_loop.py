"""
Evaluation loop — the core of the self-learning system.

For each item in the eval set, runs two attempts:
  Attempt 1: retrieve examples → generate SQL → compare vs gold
  Attempt 2: (if pass@1 failed) retrieve lessons → retry → compare vs gold
             then: critic → generate lesson → store in Index 2

Every result is written to SQLite (eval_results table) so the training
pipeline can report metrics and the lessons persist across runs.

Design decisions:
  - run_eval() takes client and kb as arguments (dependency injection).
    Tests pass mocks; the training pipeline passes real instances.
  - EvalRunStats carries auxiliary data (sim scores, latencies, per-item flags)
    that doesn't live in SQLite but is needed by metrics.py.
  - Lesson generation is attempted for ALL pass@1 failures, regardless of
    pass@2 outcome. Even when pass@2 succeeds, the pass@1 failure pattern
    is worth capturing — future queries benefit from having the lesson in Index 2.
  - LessonGenerationError and LLMError in lesson gen are caught silently.
    A lesson gen failure is not a reason to mark the eval item differently —
    the pass/fail outcome is already determined. We log and move on.
  - validate_sql failure → treat pred_sql as wrong (pass=False) immediately.
    An invalid SQL is a failed answer. The raw string is still stored so the
    critic can analyze what the model produced.
  - _avg_score() computes average cosine similarity across retrieved results.
    This feeds into avg_sim1 / avg_sim2 metrics and the confidence signal.
"""

import time
from dataclasses import dataclass, field

from app.db.models import Lesson
from app.db.sqlite_client import insert_eval_result, insert_lesson
from app.lessons.critic import analyze
from app.lessons.generator import LessonGenerationError, generate_lesson
from app.llm.claude_client import LLMClient, LLMError
from app.sql.comparator import compare
from app.sql.generator import generate_sql
from app.sql.validator import validate_sql
from app.vectorstore.kb_manager import KBManager


@dataclass
class EvalRunStats:
    """
    All data produced by one eval run, in memory.

    items:          One dict per eval item — pass/fail flags and retry metadata.
    sim1_scores:    Per-item avg cosine similarity from example retrieval (Index 1).
    sim2_scores:    Per-item avg cosine similarity from lesson retrieval (Index 2).
                    Only populated for items that went to attempt 2.
    latencies_ms:   End-to-end wall-clock time per item in milliseconds.
    """

    items: list[dict] = field(default_factory=list)
    sim1_scores: list[float] = field(default_factory=list)
    sim2_scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)


def run_eval(
    client: LLMClient,
    kb: KBManager,
    eval_pairs: list[dict],
    training_run_id: int,
) -> EvalRunStats:
    """
    Run the evaluation loop over a set of NLQ+SQL pairs.

    Args:
        client:          Shared LLMClient instance for all LLM calls.
        kb:              KBManager with loaded indexes (both Index 1 and Index 2).
        eval_pairs:      List of dicts: [{nlq, sql, db_row_id}, ...].
                         These are the 20% holdout items from the training pipeline.
        training_run_id: SQLite ID of the current training run, for linking eval results.

    Returns:
        EvalRunStats with per-item results, sim scores, and latencies.
        All results are also persisted to SQLite via insert_eval_result().
    """
    stats = EvalRunStats()

    for pair in eval_pairs:
        start_time = time.time()

        nlq = pair["nlq"]
        gold_sql = pair["sql"]

        # ── Attempt 1 ─────────────────────────────────────────────────────────
        examples = kb.search_examples(nlq)
        avg_sim1 = _avg_score(examples)

        pred_sql_1: str | None = None
        pass1 = False

        try:
            pred_sql_1 = generate_sql(client, nlq, examples)
            valid_1, _ = validate_sql(pred_sql_1)
            if valid_1:
                pass1 = compare(pred_sql_1, gold_sql)["ast_match"]
        except LLMError:
            pass  # pred_sql_1 may be None; pass1 stays False

        # ── Attempt 2 (only on pass@1 failure) ───────────────────────────────
        pred_sql_2: str | None = None
        pass2 = False
        used_lesson_id: int | None = None
        retried = False
        had_lessons = False
        avg_sim2: float | None = None

        if not pass1:
            retried = True
            lessons = kb.search_lessons(nlq)
            had_lessons = len(lessons) > 0
            avg_sim2 = _avg_score(lessons)

            try:
                pred_sql_2 = generate_sql(client, nlq, examples, lessons=lessons)
                valid_2, _ = validate_sql(pred_sql_2)
                if valid_2:
                    pass2 = compare(pred_sql_2, gold_sql)["ast_match"]
            except LLMError:
                pass  # pred_sql_2 may be None; pass2 stays False

            # ── Lesson generation from pass@1 failure ─────────────────────────
            # Attempt regardless of pass@2 outcome — the failure pattern is worth
            # capturing even if the retry succeeded.
            source_sql = pred_sql_1 or ""
            try:
                errors = analyze(client, nlq, source_sql, gold_sql)
                lesson = generate_lesson(client, nlq, source_sql, gold_sql, errors)
                lesson_row_id = insert_lesson(
                    title=lesson.title,
                    trigger=lesson.trigger,
                    diagnosis=lesson.diagnosis,
                    fix_rule=lesson.fix_rule,
                    error_category=lesson.error_category,
                    example=lesson.example,
                )
                kb.add_lesson(lesson, lesson_row_id)
                used_lesson_id = lesson_row_id
            except (LessonGenerationError, LLMError):
                pass  # lesson gen failure does not affect pass/fail outcome

        hard_failure = not pass1 and not pass2
        latency_ms = (time.time() - start_time) * 1000

        # ── Persist to SQLite ──────────────────────────────────────────────────
        insert_eval_result(
            training_run_id=training_run_id,
            nlq=nlq,
            gold_sql=gold_sql,
            pred_sql_1=pred_sql_1,
            pred_sql_2=pred_sql_2,
            pass1=pass1,
            pass2=pass2,
            hard_failure=hard_failure,
            used_lesson_id=used_lesson_id,
        )

        # ── Accumulate stats ───────────────────────────────────────────────────
        stats.items.append({
            "nlq":         nlq,
            "pass1":       pass1,
            "pass2":       pass2,
            "hard_failure":hard_failure,
            "retried":     retried,
            "had_lessons": had_lessons,
        })
        stats.sim1_scores.append(avg_sim1)
        if avg_sim2 is not None:
            stats.sim2_scores.append(avg_sim2)
        stats.latencies_ms.append(latency_ms)

    return stats


# ── Private helpers ────────────────────────────────────────────────────────────

def _avg_score(results: list[dict]) -> float:
    """
    Average cosine similarity score across a list of search results.

    Works for both example results ({nlq, sql, score}) and lesson results
    ({title, trigger, ..., score}) — both include a 'score' key.

    Returns 0.0 for empty lists (no retrieval results = zero confidence).
    """
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)
