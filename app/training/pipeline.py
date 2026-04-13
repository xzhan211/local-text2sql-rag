"""
End-to-end training pipeline.

Takes a CSV of NLQ+SQL pairs (NLQ optional), runs the full self-learning cycle,
and returns a MetricsReport.

Pipeline steps:
  1. Load CSV  →  list of {nlq, sql} rows
  2. Fill missing NLQs via LLM (generate_nlq)
  3. 80/20 random split (reproducible via settings.random_seed)
  4. Insert training run record in SQLite
  5. 80% → SQLite (insert_nlq_sql_pair) + Index 1 (kb.add_example)
  6. Save indexes (checkpoint before eval)
  7. Run eval loop on 20% → EvalRunStats
       (inside eval loop: pass@1 failure → critic → lesson → Index 2 via kb.add_lesson)
  8. Backfill 20% into SQLite + Index 1
  9. Save indexes (final state)
  10. Compute metrics → MetricsReport
  11. Persist metrics to SQLite (finish_training_run)
  12. Return MetricsReport

Design decisions:
  - run(csv_path, client=None, kb=None): None creates defaults. Tests inject
    mocks; the FastAPI route passes None. Avoids tight coupling to instantiation.
  - LLMError on NLQ generation → skip that row with a warning. One bad row
    should not abort the entire training run.
  - Backfill happens unconditionally after eval. The eval results are already
    written to SQLite before backfill — adding eval pairs to Index 1 afterward
    does not retroactively change any metrics.
  - kb.save_indexes() is called twice: once before eval (checkpoint) and once
    after backfill (final state). Both saves are intentional — eval can take
    minutes; the checkpoint prevents losing all Index 1 work if eval crashes.
  - CSV must have a 'sql' column. 'nlq' column is optional. Any other columns
    are silently ignored.
"""

import json
import random
import warnings
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.db.sqlite_client import (
    finish_training_run,
    init_db,
    insert_nlq_sql_pair,
    insert_training_run,
)
from app.evaluation.eval_loop import run_eval
from app.evaluation.metrics import MetricsReport, compute_metrics, report_to_dict
from app.llm.claude_client import LLMClient, LLMError
from app.sql.generator import generate_nlq
from app.vectorstore.kb_manager import KBManager


def run(
    csv_path: str | Path,
    client: LLMClient | None = None,
    kb: KBManager | None = None,
    run_id: int | None = None,
) -> MetricsReport:
    """
    Run the full training pipeline from a CSV file.

    Args:
        csv_path: Path to CSV with 'sql' column (required) and 'nlq' column (optional).
        client:   LLMClient instance. If None, a new one is created from settings.
        kb:       KBManager instance. If None, a new one is created and indexes loaded.
        run_id:   Existing training run id to use. If None, a new record is created.
                  Pass this when the API layer has pre-inserted the run (so the client
                  can poll status before training finishes).

    Returns:
        MetricsReport with all 9 metrics from the eval run.

    Raises:
        ValueError: if the CSV has no 'sql' column or is empty after loading.
        FileNotFoundError: if csv_path does not exist.
    """
    csv_path = Path(csv_path)

    # ── Initialize dependencies ────────────────────────────────────────────────
    if client is None:
        client = LLMClient()
    if kb is None:
        kb = KBManager()
        kb.load_indexes()

    init_db()

    # ── Step 1: Load CSV ───────────────────────────────────────────────────────
    pairs = _load_csv(csv_path)
    if not pairs:
        raise ValueError(f"CSV at {csv_path} is empty or has no valid rows.")

    # ── Step 2: Fill missing NLQs ──────────────────────────────────────────────
    pairs = _fill_missing_nlqs(client, pairs)
    if not pairs:
        raise ValueError("No valid rows remain after NLQ generation.")

    # ── Step 3: 80/20 split ────────────────────────────────────────────────────
    train_pairs, eval_pairs = _split(pairs)

    # ── Step 4: Create training run record ────────────────────────────────────
    training_run_id = run_id if run_id is not None else insert_training_run(str(csv_path))

    # ── Step 5: Index 80% into SQLite + Index 1 ───────────────────────────────
    for pair in train_pairs:
        row_id = insert_nlq_sql_pair(pair["nlq"], pair["sql"])
        kb.add_example(pair["nlq"], pair["sql"], row_id)

    # ── Step 6: Save checkpoint before eval ───────────────────────────────────
    kb.save_indexes()

    # ── Step 7: Run eval loop on 20% ──────────────────────────────────────────
    # Insert eval pairs into SQLite first so they have row IDs.
    # The db_row_id is passed to run_eval so KBManager can link FAISS IDs later
    # during backfill.
    eval_pairs_with_ids = []
    for pair in eval_pairs:
        row_id = insert_nlq_sql_pair(pair["nlq"], pair["sql"])
        eval_pairs_with_ids.append({**pair, "db_row_id": row_id})

    stats = run_eval(client, kb, eval_pairs_with_ids, training_run_id)

    # ── Step 8: Backfill eval pairs into Index 1 ──────────────────────────────
    for pair in eval_pairs_with_ids:
        kb.add_example(pair["nlq"], pair["sql"], pair["db_row_id"])

    # ── Step 9: Save final state ───────────────────────────────────────────────
    kb.save_indexes()

    # ── Step 10-11: Metrics + persist ─────────────────────────────────────────
    report = compute_metrics(stats)
    finish_training_run(training_run_id, json.dumps(report_to_dict(report)))

    return report


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_csv(csv_path: Path) -> list[dict]:
    """
    Load a CSV and return a list of {nlq, sql} dicts.

    'sql' column is required. 'nlq' column is optional — missing or empty
    values are stored as empty string for the NLQ fill step to handle.
    All other columns are ignored. Rows with empty 'sql' are dropped.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    if "sql" not in df.columns:
        raise ValueError(f"CSV must have a 'sql' column. Found: {list(df.columns)}")

    if "nlq" not in df.columns:
        df["nlq"] = ""

    pairs = []
    for _, row in df.iterrows():
        sql = row["sql"].strip()
        nlq = row["nlq"].strip()
        if sql:
            pairs.append({"nlq": nlq, "sql": sql})

    return pairs


def _fill_missing_nlqs(client: LLMClient, pairs: list[dict]) -> list[dict]:
    """
    For rows with an empty NLQ, generate one from the SQL using the LLM.

    Rows where NLQ generation fails (LLMError) are dropped with a warning.
    Rows that already have an NLQ are returned unchanged.
    """
    filled = []
    for pair in pairs:
        if pair["nlq"]:
            filled.append(pair)
            continue
        try:
            nlq = generate_nlq(client, pair["sql"])
            filled.append({**pair, "nlq": nlq})
        except LLMError as e:
            warnings.warn(
                f"NLQ generation failed for SQL: {pair['sql'][:60]}... — skipping row. ({e})"
            )
    return filled


def _split(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Randomly split pairs into 80% train / 20% eval.

    Uses settings.random_seed for reproducibility — same CSV always produces
    the same split. Guarantees at least 1 item in each split.

    Raises:
        ValueError: if fewer than 2 pairs are provided (can't satisfy ≥1 train + ≥1 eval).
    """
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 pairs to split into train/eval; got {len(pairs)}.")

    shuffled = list(pairs)
    random.seed(settings.random_seed)
    random.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - settings.eval_split)))
    split_idx = min(split_idx, len(shuffled) - 1)  # ensure at least 1 eval item

    return shuffled[:split_idx], shuffled[split_idx:]
