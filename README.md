# Local Text-to-SQL RAG

A local, self-learning text-to-SQL system built with retrieval-augmented generation (RAG). Upload SQL examples, ask natural language questions, and get SQL back — with a feedback loop that learns from its own mistakes.

Built as a learning project to deeply understand the architecture of a production text-to-SQL + RAG + evaluation + self-learning system, using local/cheap substitutes for cloud infrastructure.

---

## How It Works

**Two FAISS indexes drive everything:**

- **Index 1** — NLQ+SQL example pairs. Retrieved as few-shot examples for SQL generation.
- **Index 2** — Lessons learned from failures. Retrieved on retry to guide the LLM away from past mistakes.

**Inference** (answering a query):
1. Embed the question → retrieve similar examples from Index 1
2. Compute confidence from retrieval similarity
3. Generate SQL (attempt 1)
4. If confidence is low or SQL is invalid → retrieve lessons from Index 2 → retry with lesson-augmented prompt

**Training** (uploading new examples):
1. Load a CSV of NLQ+SQL pairs (NLQ is optional — generated from SQL via LLM if missing)
2. 80/20 split: index 80% into Index 1
3. Eval loop on 20%: failures trigger critic → lesson generation → Index 2
4. Backfill eval pairs into Index 1
5. Return metrics report

Each training run makes the system smarter — lessons from failures are retrieved and injected into future retries.

---

## Stack

| Concern | Library |
|---|---|
| LLM | Claude API (`claude-sonnet-4-6`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector search | `faiss-cpu` (IndexFlatIP + L2 normalization = cosine similarity) |
| SQL validation | `sqlglot` (AST parsing, DuckDB dialect) |
| Database | `duckdb` (query execution), `sqlite3` (metadata) |
| API | `FastAPI` + `uvicorn` |
| Config | `pydantic-settings` |

---

## Setup

**Requirements:** Python 3.11+

```bash
# Clone and create virtual environment
git clone https://github.com/xzhan211/local-text2sql-rag.git
cd local-text2sql-rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

---

## Running the API

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health`

```json
curl -X GET http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `POST /api/v1/query`
Ask a natural language question.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"nlq": "how many customers signed up last month?"}'
```

```json
{
  "sql": "SELECT COUNT(*) FROM customers WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
  "confidence": 0.83,
  "used_lesson": false
}
```

### `POST /api/v1/train/upload`
Upload a CSV to train the system. Returns immediately — training runs in the background.

```bash
curl -X POST http://localhost:8000/api/v1/train/upload \
  -F "file=@data/sample_pairs.csv"
```

```json
{"run_id": 1, "status": "running"}
```

The CSV must have a `sql` column. The `nlq` column is optional — missing NLQs are generated from SQL via LLM.

### `GET /api/v1/train/{run_id}`
Poll training status.

```bash
curl http://localhost:8000/api/v1/train/1
```

```json
{
  "run_id": 1,
  "status": "done",
  "metrics": {
    "pass_at_1": 0.8,
    "pass_at_2": 0.1,
    "hard_failure_rate": 0.1,
    "overall_pass_rate": 0.9,
    "kb_recovery_rate": 0.5,
    "lesson_utilization": 0.6,
    "avg_sim1": 0.81,
    "avg_sim2": 0.74,
    "avg_latency_ms": 1240.0
  }
}
```

---

## Sample Data

`data/sample_pairs.csv` contains 25 NLQ+SQL pairs covering a simple e-commerce schema (customers, orders, order_items, products). Use it to seed the system:

```bash
curl -X POST http://localhost:8000/api/v1/train/upload \
  -F "file=@data/sample_pairs.csv"
```

---

## Running Tests

```bash
pytest
```

216 tests across all modules. No real API calls or disk I/O — all external dependencies are mocked.

---

## Project Structure

```
app/
  core/config.py          — all settings (API key, thresholds, paths)
  db/                     — SQLite client + schema
  embeddings/embedder.py  — sentence-transformers wrapper
  vectorstore/            — FAISS index + two-index KBManager
  llm/                    — Claude API client + all prompt builders
  sql/                    — generator, validator, comparator
  lessons/                — critic + lesson generator
  evaluation/             — eval loop + metrics
  training/pipeline.py    — training orchestrator
  inference/pipeline.py   — inference orchestrator
  api/                    — FastAPI routes
  main.py                 — app entrypoint
docs/
  architecture.md         — system design + module responsibilities
  lessons_schema.md       — Lesson JSON schema + field semantics
  prompt_design.md        — prompt design decisions
```

---

## Configuration

All settings live in `app/core/config.py` and can be overridden via `.env`:

```env
ANTHROPIC_API_KEY=your_key_here

# Optional overrides
LLM_MODEL=claude-sonnet-4-6
CONFIDENCE_THRESHOLD=0.75
TOP_K_EXAMPLES=5
TOP_K_LESSONS=3
EVAL_SPLIT=0.2
```
