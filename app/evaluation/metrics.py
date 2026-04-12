"""
Metrics computation and reporting for evaluation runs.

Takes an EvalRunStats (produced by eval_loop.run_eval) and computes all
9 metrics defined in the project spec. Also produces a human-readable
report string for logging and terminal output.

Metrics:
  pass1_count       — items correct on first attempt
  pass2_count       — items correct after lesson-augmented retry
  hard_failure_count— items that failed both attempts
  overall_pass_rate — (pass1 + pass2) / total
  kb_recovery_rate  — pass2 / total_failures  (how often retry rescued a failure)
  lesson_utilization— retries that had lessons available / total retries
  avg_sim1          — avg retrieval confidence (example search, all items)
  avg_sim2          — avg lesson retrieval confidence (retry items only)
  avg_latency_ms    — avg end-to-end wall time per item

Design decisions:
  - MetricsReport is a dataclass, not Pydantic. It's a pure computation result,
    not a DB model — no serialization or field validation needed here.
  - kb_recovery_rate returns 1.0 when there are no failures (everything passed
    on attempt 1). This represents "100% of failures were recovered" vacuously,
    which is the correct interpretation when the failure set is empty.
  - lesson_utilization returns 0.0 when there are no retries. This is honest:
    if nothing was retried, lessons were not utilized.
  - avg_sim2 is 0.0 when no retries occurred (sim2_scores is empty). The training
    pipeline stores MetricsReport as JSON — consumers should check pass2_count > 0
    before interpreting avg_sim2.
  - format_report() is for human consumption only. The training pipeline stores
    the full MetricsReport as JSON via dataclasses.asdict().
"""

import dataclasses
from dataclasses import dataclass

from app.evaluation.eval_loop import EvalRunStats


@dataclass
class MetricsReport:
    total: int
    pass1_count: int
    pass2_count: int
    hard_failure_count: int
    overall_pass_rate: float    # (pass1 + pass2) / total
    kb_recovery_rate: float     # pass2 / (total - pass1_count); 1.0 if no failures
    lesson_utilization: float   # retries with lessons / total retries; 0.0 if no retries
    avg_sim1: float
    avg_sim2: float             # over retry items only; 0.0 if no retries
    avg_latency_ms: float


def compute_metrics(stats: EvalRunStats) -> MetricsReport:
    """
    Compute all metrics from an EvalRunStats produced by run_eval().

    Args:
        stats: The EvalRunStats returned by eval_loop.run_eval().

    Returns:
        A populated MetricsReport. All rates are in [0.0, 1.0].
    """
    total = len(stats.items)

    if total == 0:
        return MetricsReport(
            total=0, pass1_count=0, pass2_count=0, hard_failure_count=0,
            overall_pass_rate=0.0, kb_recovery_rate=0.0, lesson_utilization=0.0,
            avg_sim1=0.0, avg_sim2=0.0, avg_latency_ms=0.0,
        )

    pass1_count        = sum(1 for item in stats.items if item["pass1"])
    pass2_count        = sum(1 for item in stats.items if item["pass2"])
    hard_failure_count = sum(1 for item in stats.items if item["hard_failure"])

    overall_pass_rate = (pass1_count + pass2_count) / total

    total_failures = total - pass1_count
    kb_recovery_rate = (pass2_count / total_failures) if total_failures > 0 else 1.0

    total_retried     = sum(1 for item in stats.items if item["retried"])
    total_had_lessons = sum(1 for item in stats.items if item["had_lessons"])
    lesson_utilization = (total_had_lessons / total_retried) if total_retried > 0 else 0.0

    avg_sim1 = _mean(stats.sim1_scores)
    avg_sim2 = _mean(stats.sim2_scores)
    avg_latency_ms = _mean(stats.latencies_ms)

    return MetricsReport(
        total=total,
        pass1_count=pass1_count,
        pass2_count=pass2_count,
        hard_failure_count=hard_failure_count,
        overall_pass_rate=overall_pass_rate,
        kb_recovery_rate=kb_recovery_rate,
        lesson_utilization=lesson_utilization,
        avg_sim1=avg_sim1,
        avg_sim2=avg_sim2,
        avg_latency_ms=avg_latency_ms,
    )


def format_report(report: MetricsReport) -> str:
    """
    Format a MetricsReport as a human-readable string for logging / terminal output.

    Example output:
        ══════════════════════════════════════════════════
          Evaluation Report
        ══════════════════════════════════════════════════
          Total items:          10
          pass@1:               7  (70.0%)
          pass@2:               2  (20.0%)
          Hard failures:        1  (10.0%)
          Overall pass rate:    90.0%
          ──────────────────────────────────────────────
          KB recovery rate:     66.7%   (2 of 3 failures recovered)
          Lesson utilization:   100.0%  (lessons available on all retries)
          ──────────────────────────────────────────────
          Avg retrieval sim:    0.812
          Avg lesson sim:       0.743
          Avg latency:          1243 ms
        ══════════════════════════════════════════════════
    """
    r = report
    total_failures = r.total - r.pass1_count

    lines = [
        "══════════════════════════════════════════════════",
        "  Evaluation Report",
        "══════════════════════════════════════════════════",
        f"  Total items:          {r.total}",
        f"  pass@1:               {r.pass1_count}  ({r.pass1_count / r.total:.1%})" if r.total else "  pass@1:               0",
        f"  pass@2:               {r.pass2_count}  ({r.pass2_count / r.total:.1%})" if r.total else "  pass@2:               0",
        f"  Hard failures:        {r.hard_failure_count}  ({r.hard_failure_count / r.total:.1%})" if r.total else "  Hard failures:        0",
        f"  Overall pass rate:    {r.overall_pass_rate:.1%}",
        "  ──────────────────────────────────────────────",
        f"  KB recovery rate:     {r.kb_recovery_rate:.1%}   ({r.pass2_count} of {total_failures} failures recovered)",
        f"  Lesson utilization:   {r.lesson_utilization:.1%}",
        "  ──────────────────────────────────────────────",
        f"  Avg retrieval sim:    {r.avg_sim1:.3f}",
        f"  Avg lesson sim:       {r.avg_sim2:.3f}",
        f"  Avg latency:          {r.avg_latency_ms:.0f} ms",
        "══════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


def report_to_dict(report: MetricsReport) -> dict:
    """
    Serialize a MetricsReport to a plain dict for JSON storage in SQLite.

    Used by the training pipeline to store metrics in the training_runs table.
    """
    return dataclasses.asdict(report)


# ── Private helpers ────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    """Return the mean of a list, or 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0
