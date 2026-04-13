"""
Training routes:
  POST /api/v1/train/upload   — upload CSV, start training in background
  GET  /api/v1/train/{run_id} — poll training run status and metrics

Design decisions:
  - Training runs as a BackgroundTask: the endpoint returns {run_id, status}
    immediately so the client is not blocked waiting for minutes of LLM calls.
  - Status is tracked in SQLite (training_runs table). GET /train/{id} reads
    from there — no in-memory state needed.
  - The uploaded CSV is written to a NamedTemporaryFile. The path is passed to
    pipeline.run(). The temp file is cleaned up after training completes.
  - If pipeline.run() raises (e.g. missing 'sql' column), finish_training_run
    is called with status='failed' so the error is visible via GET /train/{id}.
"""

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.db.sqlite_client import finish_training_run, get_training_run, insert_training_run
from app.training.pipeline import run as run_pipeline

router = APIRouter(prefix="/api/v1")


# ── Response models ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    run_id: int
    status: str


class TrainStatusResponse(BaseModel):
    run_id: int
    status: str
    metrics: dict | None = None


# ── Background task ───────────────────────────────────────────────────────────

def _run_training(run_id: int, csv_path: Path) -> None:
    """
    Execute the training pipeline in the background.

    Called by BackgroundTasks after the upload response is sent.
    Passes run_id so pipeline.run() uses the pre-inserted record instead of
    creating a new one — ensures the client's run_id stays consistent.
    Marks the run as 'failed' on error. Cleans up the temp CSV regardless.
    """
    try:
        run_pipeline(csv_path, run_id=run_id)
    except Exception as e:
        finish_training_run(run_id, json.dumps({"error": str(e)}), status="failed")
    finally:
        csv_path.unlink(missing_ok=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/train/upload", response_model=UploadResponse, status_code=202)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a CSV file and start a training run in the background.

    The CSV must have a 'sql' column. The 'nlq' column is optional.
    Returns immediately with run_id and status='running'.
    Poll GET /api/v1/train/{run_id} to check progress.
    """
    # Write upload to a temp file — pipeline.run() expects a file path
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        csv_path = Path(tmp.name)
    finally:
        tmp.close()

    run_id = insert_training_run(file.filename or "upload.csv")
    background_tasks.add_task(_run_training, run_id, csv_path)

    return UploadResponse(run_id=run_id, status="running")


@router.get("/train/{run_id}", response_model=TrainStatusResponse)
def get_status(run_id: int) -> TrainStatusResponse:
    """
    Get the status and metrics of a training run.

    Status values:
      running — pipeline is still executing
      done    — pipeline completed successfully
      failed  — pipeline raised an exception (metrics contains error key)
    """
    row = get_training_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found.")

    metrics = json.loads(row["metrics"]) if row["metrics"] else None
    return TrainStatusResponse(run_id=row["id"], status=row["status"], metrics=metrics)
