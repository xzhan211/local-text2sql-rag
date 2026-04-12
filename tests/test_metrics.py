"""
Unit tests for app/evaluation/metrics.py.

Pure math — no I/O, no mocks. Tests use hand-crafted EvalRunStats with known
values so every metric can be verified against expected results.
"""

import pytest

from app.evaluation.eval_loop import EvalRunStats
from app.evaluation.metrics import MetricsReport, compute_metrics, format_report, report_to_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_item(pass1: bool, pass2: bool = False, retried: bool = False, had_lessons: bool = False) -> dict:
    return {
        "nlq":          "some query",
        "pass1":        pass1,
        "pass2":        pass2,
        "hard_failure": not pass1 and not pass2,
        "retried":      retried,
        "had_lessons":  had_lessons,
    }


def _make_stats(items, sim1=None, sim2=None, latencies=None) -> EvalRunStats:
    return EvalRunStats(
        items=items,
        sim1_scores=sim1 or [0.8] * len(items),
        sim2_scores=sim2 or [],
        latencies_ms=latencies or [100.0] * len(items),
    )


# ── compute_metrics — empty stats ─────────────────────────────────────────────

class TestComputeMetricsEmpty:
    def test_empty_stats_returns_zeros(self):
        report = compute_metrics(EvalRunStats())
        assert report.total == 0
        assert report.pass1_count == 0
        assert report.overall_pass_rate == 0.0

    def test_empty_stats_returns_metrics_report(self):
        assert isinstance(compute_metrics(EvalRunStats()), MetricsReport)


# ── compute_metrics — counts ──────────────────────────────────────────────────

class TestCounts:
    def test_all_pass1(self):
        items = [_make_item(pass1=True) for _ in range(5)]
        report = compute_metrics(_make_stats(items))
        assert report.total == 5
        assert report.pass1_count == 5
        assert report.pass2_count == 0
        assert report.hard_failure_count == 0

    def test_mixed_results(self):
        items = [
            _make_item(pass1=True),                              # pass@1
            _make_item(pass1=True),                              # pass@1
            _make_item(pass1=False, pass2=True, retried=True),   # pass@2
            _make_item(pass1=False, pass2=False, retried=True),  # hard failure
        ]
        report = compute_metrics(_make_stats(items))
        assert report.total == 4
        assert report.pass1_count == 2
        assert report.pass2_count == 1
        assert report.hard_failure_count == 1

    def test_all_hard_failures(self):
        items = [_make_item(pass1=False, pass2=False, retried=True) for _ in range(3)]
        report = compute_metrics(_make_stats(items))
        assert report.hard_failure_count == 3
        assert report.pass1_count == 0
        assert report.pass2_count == 0


# ── compute_metrics — rates ───────────────────────────────────────────────────

class TestRates:
    def test_overall_pass_rate_all_pass1(self):
        items = [_make_item(pass1=True)] * 10
        report = compute_metrics(_make_stats(items))
        assert report.overall_pass_rate == pytest.approx(1.0)

    def test_overall_pass_rate_mixed(self):
        # 7 pass@1, 2 pass@2, 1 hard_failure → (7+2)/10 = 0.9
        items = (
            [_make_item(pass1=True)] * 7 +
            [_make_item(pass1=False, pass2=True, retried=True)] * 2 +
            [_make_item(pass1=False, pass2=False, retried=True)] * 1
        )
        report = compute_metrics(_make_stats(items))
        assert report.overall_pass_rate == pytest.approx(0.9)

    def test_kb_recovery_rate(self):
        # 3 failures, 2 recovered by pass@2 → 2/3
        items = (
            [_make_item(pass1=True)] * 7 +
            [_make_item(pass1=False, pass2=True, retried=True)] * 2 +
            [_make_item(pass1=False, pass2=False, retried=True)] * 1
        )
        report = compute_metrics(_make_stats(items))
        assert report.kb_recovery_rate == pytest.approx(2 / 3)

    def test_kb_recovery_rate_no_failures(self):
        # All pass@1 — no failures → kb_recovery_rate = 1.0 (vacuously)
        items = [_make_item(pass1=True)] * 5
        report = compute_metrics(_make_stats(items))
        assert report.kb_recovery_rate == pytest.approx(1.0)

    def test_lesson_utilization_all_had_lessons(self):
        items = [_make_item(pass1=False, retried=True, had_lessons=True)] * 4
        report = compute_metrics(_make_stats(items))
        assert report.lesson_utilization == pytest.approx(1.0)

    def test_lesson_utilization_no_lessons(self):
        # Index 2 was empty — retried but no lessons available
        items = [_make_item(pass1=False, retried=True, had_lessons=False)] * 3
        report = compute_metrics(_make_stats(items))
        assert report.lesson_utilization == pytest.approx(0.0)

    def test_lesson_utilization_no_retries(self):
        # Everything passed on attempt 1 — no retries at all
        items = [_make_item(pass1=True)] * 5
        report = compute_metrics(_make_stats(items))
        assert report.lesson_utilization == pytest.approx(0.0)

    def test_lesson_utilization_partial(self):
        # 2 retries had lessons, 2 did not → 0.5
        items = (
            [_make_item(pass1=False, retried=True, had_lessons=True)] * 2 +
            [_make_item(pass1=False, retried=True, had_lessons=False)] * 2
        )
        report = compute_metrics(_make_stats(items))
        assert report.lesson_utilization == pytest.approx(0.5)


# ── compute_metrics — similarity and latency ──────────────────────────────────

class TestSimAndLatency:
    def test_avg_sim1(self):
        items = [_make_item(pass1=True)] * 3
        stats = _make_stats(items, sim1=[0.9, 0.8, 0.7])
        report = compute_metrics(stats)
        assert report.avg_sim1 == pytest.approx(0.8)

    def test_avg_sim2_no_retries(self):
        items = [_make_item(pass1=True)] * 3
        stats = _make_stats(items, sim2=[])
        report = compute_metrics(stats)
        assert report.avg_sim2 == pytest.approx(0.0)

    def test_avg_sim2_with_retries(self):
        items = [_make_item(pass1=False, retried=True)] * 2
        stats = _make_stats(items, sim2=[0.6, 0.4])
        report = compute_metrics(stats)
        assert report.avg_sim2 == pytest.approx(0.5)

    def test_avg_latency_ms(self):
        items = [_make_item(pass1=True)] * 4
        stats = _make_stats(items, latencies=[100.0, 200.0, 300.0, 400.0])
        report = compute_metrics(stats)
        assert report.avg_latency_ms == pytest.approx(250.0)


# ── format_report ─────────────────────────────────────────────────────────────

class TestFormatReport:
    def _base_report(self) -> MetricsReport:
        items = (
            [_make_item(pass1=True)] * 7 +
            [_make_item(pass1=False, pass2=True, retried=True, had_lessons=True)] * 2 +
            [_make_item(pass1=False, pass2=False, retried=True, had_lessons=True)] * 1
        )
        return compute_metrics(_make_stats(items, sim1=[0.8] * 10, sim2=[0.7, 0.6], latencies=[1200.0] * 10))

    def test_returns_string(self):
        assert isinstance(format_report(self._base_report()), str)

    def test_contains_total(self):
        assert "10" in format_report(self._base_report())

    def test_contains_overall_pass_rate(self):
        output = format_report(self._base_report())
        assert "90.0%" in output or "90%" in output

    def test_contains_section_headers(self):
        output = format_report(self._base_report())
        assert "Evaluation Report" in output
        assert "KB recovery rate" in output
        assert "Lesson utilization" in output


# ── report_to_dict ────────────────────────────────────────────────────────────

class TestReportToDict:
    def test_returns_dict(self):
        items = [_make_item(pass1=True)] * 3
        report = compute_metrics(_make_stats(items))
        result = report_to_dict(report)
        assert isinstance(result, dict)

    def test_contains_all_metric_keys(self):
        items = [_make_item(pass1=True)] * 3
        report = compute_metrics(_make_stats(items))
        d = report_to_dict(report)
        expected_keys = {
            "total", "pass1_count", "pass2_count", "hard_failure_count",
            "overall_pass_rate", "kb_recovery_rate", "lesson_utilization",
            "avg_sim1", "avg_sim2", "avg_latency_ms",
        }
        assert expected_keys.issubset(d.keys())
