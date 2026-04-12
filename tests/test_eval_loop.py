"""
Unit tests for app/evaluation/eval_loop.py.

LLMClient and KBManager are mocked. SQLite calls (insert_eval_result,
insert_lesson) are patched at the module level so tests don't need a DB.

Test strategy:
  - Verify pass@1 / pass@2 / hard_failure logic under different compare() outcomes
  - Verify that retry only happens on pass@1 failure
  - Verify that lesson gen is attempted on all pass@1 failures
  - Verify that LLMError / LessonGenerationError in lesson gen doesn't abort the loop
  - Verify EvalRunStats shape and sim score accumulation
"""

from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.eval_loop import EvalRunStats, _avg_score, run_eval
from app.lessons.generator import LessonGenerationError
from app.llm.claude_client import LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(sql: str = "SELECT 1") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = sql
    return client


def _make_kb(
    examples: list[dict] | None = None,
    lessons: list[dict] | None = None,
) -> MagicMock:
    kb = MagicMock()
    kb.search_examples.return_value = examples or [
        {"nlq": "ex", "sql": "SELECT 1", "score": 0.85}
    ]
    kb.search_lessons.return_value = lessons or []
    kb.add_lesson.return_value = None
    return kb


_PAIR = {"nlq": "how many customers", "sql": "SELECT COUNT(*) FROM customers", "db_row_id": 1}


def _run(client, kb, pairs=None, training_run_id=1):
    """Run eval with all SQLite calls patched out."""
    with patch("app.evaluation.eval_loop.insert_eval_result", return_value=1), \
         patch("app.evaluation.eval_loop.insert_lesson", return_value=99):
        return run_eval(client, kb, pairs or [_PAIR], training_run_id)


# ── pass@1 logic ──────────────────────────────────────────────────────────────

class TestPass1:
    def test_pass1_when_ast_match(self):
        """generate_sql returns SQL that matches gold → pass@1=True."""
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": True, "token_sim": 1.0}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)):
            stats = _run(_make_client(), _make_kb())
        assert stats.items[0]["pass1"] is True
        assert stats.items[0]["retried"] is False

    def test_no_retry_when_pass1(self):
        """If pass@1, attempt 2 must not happen — kb.search_lessons not called."""
        kb = _make_kb()
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": True, "token_sim": 1.0}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)):
            _run(_make_client(), kb)
        kb.search_lessons.assert_not_called()

    def test_pass1_false_when_ast_no_match(self):
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.5}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)):
            stats = _run(_make_client(), _make_kb())
        assert stats.items[0]["pass1"] is False

    def test_pass1_false_when_invalid_sql(self):
        """validate_sql returns False → pass1=False without calling compare."""
        with patch("app.evaluation.eval_loop.validate_sql", return_value=(False, "bad SQL")), \
             patch("app.evaluation.eval_loop.compare") as mock_compare:
            stats = _run(_make_client(), _make_kb())
        assert stats.items[0]["pass1"] is False
        mock_compare.assert_not_called()

    def test_pass1_false_when_llm_error(self):
        """LLMError on attempt 1 → pass1=False, no crash."""
        client = _make_client()
        client.complete.side_effect = LLMError("API down")
        stats = _run(client, _make_kb())
        assert stats.items[0]["pass1"] is False


# ── pass@2 logic ──────────────────────────────────────────────────────────────

class TestPass2:
    def _fail_then_pass(self):
        """compare returns False first, then True (simulating pass@2)."""
        compare_mock = MagicMock(side_effect=[
            {"ast_match": False, "token_sim": 0.5},  # attempt 1
            {"ast_match": True,  "token_sim": 1.0},  # attempt 2
        ])
        return compare_mock

    def test_pass2_when_retry_matches(self):
        with patch("app.evaluation.eval_loop.compare", self._fail_then_pass()), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="error list"), \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            stats = _run(_make_client(), _make_kb())
        assert stats.items[0]["pass2"] is True
        assert stats.items[0]["hard_failure"] is False

    def test_hard_failure_when_both_fail(self):
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors"), \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            stats = _run(_make_client(), _make_kb())
        assert stats.items[0]["hard_failure"] is True

    def test_retry_uses_lessons_from_kb(self):
        """kb.search_lessons is called exactly once on pass@1 failure."""
        kb = _make_kb(lessons=[{"title": "L", "trigger": "T", "fix_rule": "F",
                                 "error_category": "other", "score": 0.7}])
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors"), \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            _run(_make_client(), kb)
        kb.search_lessons.assert_called_once()

    def test_had_lessons_flag_set_when_lessons_exist(self):
        lessons = [{"title": "L", "trigger": "T", "fix_rule": "F",
                    "error_category": "other", "score": 0.7}]
        kb = _make_kb(lessons=lessons)
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors"), \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            stats = _run(_make_client(), kb)
        assert stats.items[0]["had_lessons"] is True


# ── Lesson generation ─────────────────────────────────────────────────────────

class TestLessonGeneration:
    def test_lesson_generated_on_pass1_failure(self):
        """analyze() and generate_lesson() called when pass@1 fails."""
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors") as mock_analyze, \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            _run(_make_client(), _make_kb())
        mock_analyze.assert_called_once()

    def test_lesson_not_generated_on_pass1(self):
        """analyze() must NOT be called when pass@1 succeeds."""
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": True, "token_sim": 1.0}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze") as mock_analyze:
            _run(_make_client(), _make_kb())
        mock_analyze.assert_not_called()

    def test_lesson_gen_error_does_not_abort(self):
        """LessonGenerationError during lesson gen must not raise — loop continues."""
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors"), \
             patch("app.evaluation.eval_loop.generate_lesson",
                   side_effect=LessonGenerationError("bad JSON")):
            stats = _run(_make_client(), _make_kb())
        assert len(stats.items) == 1  # loop completed

    def test_llm_error_in_lesson_gen_does_not_abort(self):
        """LLMError during lesson gen must not raise — loop continues."""
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": False, "token_sim": 0.3}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", side_effect=LLMError("API down")):
            stats = _run(_make_client(), _make_kb())
        assert len(stats.items) == 1


# ── EvalRunStats shape ────────────────────────────────────────────────────────

class TestStatsShape:
    def test_one_item_per_pair(self):
        pairs = [_PAIR, _PAIR, _PAIR]
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": True, "token_sim": 1.0}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)):
            stats = _run(_make_client(), _make_kb(), pairs=pairs)
        assert len(stats.items) == 3
        assert len(stats.sim1_scores) == 3
        assert len(stats.latencies_ms) == 3

    def test_sim2_only_populated_for_retries(self):
        """sim2_scores only has entries for items that went to attempt 2."""
        pairs = [_PAIR, _PAIR]
        compare_results = [
            {"ast_match": True,  "token_sim": 1.0},  # pair 1: pass@1
            {"ast_match": False, "token_sim": 0.3},  # pair 2: fail → retry
            {"ast_match": False, "token_sim": 0.3},  # pair 2: retry also fails
        ]
        with patch("app.evaluation.eval_loop.compare", side_effect=compare_results), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)), \
             patch("app.evaluation.eval_loop.analyze", return_value="errors"), \
             patch("app.evaluation.eval_loop.generate_lesson", return_value=MagicMock(
                 title="t", trigger="tr", diagnosis="d", fix_rule="f",
                 error_category="other", example="{}")):
            stats = _run(_make_client(), _make_kb(), pairs=pairs)
        assert len(stats.sim2_scores) == 1  # only the retry item

    def test_returns_eval_run_stats_instance(self):
        with patch("app.evaluation.eval_loop.compare", return_value={"ast_match": True, "token_sim": 1.0}), \
             patch("app.evaluation.eval_loop.validate_sql", return_value=(True, None)):
            result = _run(_make_client(), _make_kb())
        assert isinstance(result, EvalRunStats)


# ── _avg_score ────────────────────────────────────────────────────────────────

class TestAvgScore:
    def test_empty_returns_zero(self):
        assert _avg_score([]) == 0.0

    def test_single_item(self):
        assert _avg_score([{"score": 0.8}]) == pytest.approx(0.8)

    def test_multiple_items(self):
        results = [{"score": 0.9}, {"score": 0.7}, {"score": 0.8}]
        assert _avg_score(results) == pytest.approx(0.8)
