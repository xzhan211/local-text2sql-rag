"""
Unit tests for app/training/pipeline.py.

All external calls are mocked: LLMClient, KBManager, SQLite helpers,
run_eval, and compute_metrics. CSV I/O uses real temporary files.

Test strategy:
  - _load_csv: correct parsing, missing nlq column, missing sql column, empty rows
  - _fill_missing_nlqs: NLQ generated when empty, passthrough when present, skip on error
  - _split: correct ratio, reproducibility, edge cases (tiny datasets)
  - run(): correct orchestration — verify call order and that each step happens
"""

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.evaluation.eval_loop import EvalRunStats
from app.evaluation.metrics import MetricsReport
from app.llm.claude_client import LLMError
from app.training.pipeline import _fill_missing_nlqs, _load_csv, _split, run


# ── _load_csv ─────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def _write_csv(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_loads_nlq_and_sql(self):
        path = self._write_csv("nlq,sql\nhow many,SELECT COUNT(*) FROM customers\n")
        pairs = _load_csv(path)
        assert len(pairs) == 1
        assert pairs[0]["nlq"] == "how many"
        assert pairs[0]["sql"] == "SELECT COUNT(*) FROM customers"

    def test_missing_nlq_column_defaults_to_empty(self):
        path = self._write_csv("sql\nSELECT 1\n")
        pairs = _load_csv(path)
        assert pairs[0]["nlq"] == ""

    def test_raises_when_no_sql_column(self):
        path = self._write_csv("nlq\nhow many\n")
        with pytest.raises(ValueError, match="'sql' column"):
            _load_csv(path)

    def test_drops_rows_with_empty_sql(self):
        path = self._write_csv("nlq,sql\nhow many,SELECT 1\nempty row,\n")
        pairs = _load_csv(path)
        assert len(pairs) == 1

    def test_strips_whitespace(self):
        path = self._write_csv("nlq,sql\n  how many  ,  SELECT 1  \n")
        pairs = _load_csv(path)
        assert pairs[0]["nlq"] == "how many"
        assert pairs[0]["sql"] == "SELECT 1"

    def test_ignores_extra_columns(self):
        path = self._write_csv("nlq,sql,source\nhow many,SELECT 1,manual\n")
        pairs = _load_csv(path)
        assert "source" not in pairs[0]

    def test_returns_empty_list_for_all_empty_sql(self):
        path = self._write_csv("nlq,sql\nhow many,\n")
        pairs = _load_csv(path)
        assert pairs == []


# ── _fill_missing_nlqs ────────────────────────────────────────────────────────

class TestFillMissingNlqs:
    def test_passthrough_when_nlq_present(self):
        client = MagicMock()
        pairs = [{"nlq": "existing question", "sql": "SELECT 1"}]
        result = _fill_missing_nlqs(client, pairs)
        assert result[0]["nlq"] == "existing question"
        client.complete.assert_not_called()

    def test_generates_nlq_when_empty(self):
        client = MagicMock()
        client.complete.return_value = "how many rows?"
        pairs = [{"nlq": "", "sql": "SELECT COUNT(*) FROM customers"}]
        result = _fill_missing_nlqs(client, pairs)
        assert result[0]["nlq"] == "how many rows?"

    def test_skips_row_on_llm_error(self):
        client = MagicMock()
        client.complete.side_effect = LLMError("API down")
        pairs = [{"nlq": "", "sql": "SELECT 1"}]
        with pytest.warns(UserWarning):
            result = _fill_missing_nlqs(client, pairs)
        assert result == []

    def test_partial_fill(self):
        """Mix of rows with and without NLQ — only empty ones are filled."""
        client = MagicMock()
        client.complete.return_value = "generated question"
        pairs = [
            {"nlq": "existing", "sql": "SELECT 1"},
            {"nlq": "",         "sql": "SELECT 2"},
        ]
        result = _fill_missing_nlqs(client, pairs)
        assert len(result) == 2
        assert result[0]["nlq"] == "existing"
        assert result[1]["nlq"] == "generated question"
        assert client.complete.call_count == 1

    def test_sql_unchanged_after_fill(self):
        client = MagicMock()
        client.complete.return_value = "generated"
        pairs = [{"nlq": "", "sql": "SELECT COUNT(*) FROM orders"}]
        result = _fill_missing_nlqs(client, pairs)
        assert result[0]["sql"] == "SELECT COUNT(*) FROM orders"


# ── _split ────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_correct_ratio(self):
        pairs = [{"nlq": str(i), "sql": f"SELECT {i}"} for i in range(100)]
        train, eval_ = _split(pairs)
        assert len(train) == 80
        assert len(eval_) == 20

    def test_no_overlap(self):
        pairs = [{"nlq": str(i), "sql": f"SELECT {i}"} for i in range(20)]
        train, eval_ = _split(pairs)
        train_nlqs = {p["nlq"] for p in train}
        eval_nlqs  = {p["nlq"] for p in eval_}
        assert train_nlqs.isdisjoint(eval_nlqs)

    def test_reproducible_with_seed(self):
        pairs = [{"nlq": str(i), "sql": f"SELECT {i}"} for i in range(50)]
        train1, eval1 = _split(pairs)
        train2, eval2 = _split(pairs)
        assert [p["nlq"] for p in train1] == [p["nlq"] for p in train2]

    def test_minimum_one_eval_item(self):
        pairs = [{"nlq": "a", "sql": "SELECT 1"}, {"nlq": "b", "sql": "SELECT 2"}]
        train, eval_ = _split(pairs)
        assert len(eval_) >= 1
        assert len(train) >= 1

    def test_total_preserved(self):
        pairs = [{"nlq": str(i), "sql": f"SELECT {i}"} for i in range(25)]
        train, eval_ = _split(pairs)
        assert len(train) + len(eval_) == 25

    def test_raises_on_single_item(self):
        pairs = [{"nlq": "a", "sql": "SELECT 1"}]
        with pytest.raises(ValueError, match="at least 2"):
            _split(pairs)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            _split([])


# ── run() — orchestration ─────────────────────────────────────────────────────

class TestRun:
    def _write_csv(self, n: int = 10) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write("nlq,sql\n")
        for i in range(n):
            f.write(f"question {i},SELECT {i}\n")
        f.close()
        return Path(f.name)

    def _mock_kb(self):
        kb = MagicMock()
        kb.search_examples.return_value = [{"nlq": "ex", "sql": "SELECT 1", "score": 0.9}]
        kb.search_lessons.return_value = []
        return kb

    def _mock_eval_stats(self, n: int = 2) -> EvalRunStats:
        items = [{"pass1": True, "pass2": False, "hard_failure": False,
                  "retried": False, "had_lessons": False}] * n
        return EvalRunStats(
            items=items,
            sim1_scores=[0.8] * n,
            sim2_scores=[],
            latencies_ms=[100.0] * n,
        )

    def test_returns_metrics_report(self):
        csv_path = self._write_csv(10)
        kb = self._mock_kb()
        client = MagicMock()

        with patch("app.training.pipeline.init_db"), \
             patch("app.training.pipeline.insert_training_run", return_value=1), \
             patch("app.training.pipeline.insert_nlq_sql_pair", return_value=1), \
             patch("app.training.pipeline.finish_training_run"), \
             patch("app.training.pipeline.run_eval", return_value=self._mock_eval_stats()):
            result = run(csv_path, client=client, kb=kb)

        assert isinstance(result, MetricsReport)

    def test_indexes_saved_twice(self):
        """save_indexes() must be called exactly twice: checkpoint + final."""
        csv_path = self._write_csv(10)
        kb = self._mock_kb()

        with patch("app.training.pipeline.init_db"), \
             patch("app.training.pipeline.insert_training_run", return_value=1), \
             patch("app.training.pipeline.insert_nlq_sql_pair", return_value=1), \
             patch("app.training.pipeline.finish_training_run"), \
             patch("app.training.pipeline.run_eval", return_value=self._mock_eval_stats()):
            run(csv_path, client=MagicMock(), kb=kb)

        assert kb.save_indexes.call_count == 2

    def test_train_pairs_added_before_eval(self):
        """kb.add_example must be called for train pairs before run_eval is called."""
        csv_path = self._write_csv(10)
        kb = self._mock_kb()
        call_order = []

        kb.add_example.side_effect = lambda *a, **kw: call_order.append("add_example")

        def fake_run_eval(*a, **kw):
            call_order.append("run_eval")
            return self._mock_eval_stats()

        with patch("app.training.pipeline.init_db"), \
             patch("app.training.pipeline.insert_training_run", return_value=1), \
             patch("app.training.pipeline.insert_nlq_sql_pair", return_value=1), \
             patch("app.training.pipeline.finish_training_run"), \
             patch("app.training.pipeline.run_eval", side_effect=fake_run_eval):
            run(csv_path, client=MagicMock(), kb=kb)

        first_run_eval = call_order.index("run_eval")
        last_add_before_eval = max(
            i for i, v in enumerate(call_order[:first_run_eval]) if v == "add_example"
        )
        assert last_add_before_eval < first_run_eval

    def test_backfill_adds_eval_pairs_after_eval(self):
        """add_example must be called for eval pairs after run_eval completes."""
        csv_path = self._write_csv(10)
        kb = self._mock_kb()
        call_order = []

        kb.add_example.side_effect = lambda *a, **kw: call_order.append("add_example")

        def fake_run_eval(*a, **kw):
            call_order.append("run_eval")
            return self._mock_eval_stats()

        with patch("app.training.pipeline.init_db"), \
             patch("app.training.pipeline.insert_training_run", return_value=1), \
             patch("app.training.pipeline.insert_nlq_sql_pair", return_value=1), \
             patch("app.training.pipeline.finish_training_run"), \
             patch("app.training.pipeline.run_eval", side_effect=fake_run_eval):
            run(csv_path, client=MagicMock(), kb=kb)

        run_eval_idx = call_order.index("run_eval")
        adds_after_eval = [v for v in call_order[run_eval_idx + 1:] if v == "add_example"]
        assert len(adds_after_eval) > 0  # backfill happened

    def test_finish_training_run_called_with_metrics_json(self):
        csv_path = self._write_csv(10)

        with patch("app.training.pipeline.init_db"), \
             patch("app.training.pipeline.insert_training_run", return_value=42), \
             patch("app.training.pipeline.insert_nlq_sql_pair", return_value=1), \
             patch("app.training.pipeline.finish_training_run") as mock_finish, \
             patch("app.training.pipeline.run_eval", return_value=self._mock_eval_stats()):
            run(csv_path, client=MagicMock(), kb=self._mock_kb())

        args = mock_finish.call_args[0]
        assert args[0] == 42                    # correct training_run_id
        assert isinstance(json.loads(args[1]), dict)  # valid JSON metrics

    def test_raises_on_missing_sql_column(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write("nlq\nsome question\n")
        f.close()
        with pytest.raises(ValueError, match="'sql' column"):
            run(f.name, client=MagicMock(), kb=self._mock_kb())

    def test_raises_on_empty_csv(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write("nlq,sql\n")
        f.close()
        with pytest.raises(ValueError, match="empty"):
            with patch("app.training.pipeline.init_db"):
                run(f.name, client=MagicMock(), kb=self._mock_kb())
