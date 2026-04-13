# Project Context — Local Text-to-SQL RAG System

## What This File Is

This file exists so that Claude Code can fully reconstruct project context at the start of any new session. Read this before doing anything else.

---

## Project Goal

Build a **local, low-cost personal version** of a production self-learning text-to-SQL RAG system. The goal is dual:
1. Deeply understand the architecture of a production text-to-SQL + RAG + evaluation + self-learning system.
2. Practice structured AI-assisted development: plan → concepts → tradeoffs → code → review → debug → test.

**Local replacements for production infrastructure:**

| Production | Local substitute |
|---|---|
| AWS Bedrock | Claude API (anthropic SDK) |
| Amazon OpenSearch Serverless | FAISS (faiss-cpu) |
| Snowflake | DuckDB |
| AWS infra / Terraform | None — single process |

---

## How We Communicate (Critical — Follow This Every Phase)

**Never jump straight to code.** For every phase, the sequence is:

1. **Review plan** — show files to create, architecture diagram, design decisions
2. **Must-learn concepts** — teach the concepts required for this phase before coding
3. **Tradeoffs** — discuss key decisions and alternatives
4. **Start coding** — only after the user confirms they understand and approves the plan
5. **User reviews every file carefully** — do not batch-generate; go module by module
6. **User debugs locally** — wait for feedback before moving to next file
7. **Add tests** — after source files are reviewed and working

---

## System Architecture

```
User Query (NLQ)
     │
     ▼
┌──────────────┐
│  FastAPI API  │  ← POST /api/v1/query, POST /api/v1/train/upload, GET /health
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                 Inference Pipeline                    │
│  1. Embed NLQ (sentence-transformers)                │
│  2. Retrieve top-k from Index 1 (FAISS)              │
│  3. Compute confidence (avg cosine similarity)        │
│  4. Generate SQL (Claude API, constrained prompt)     │
│  5. If low confidence → retrieve lessons (Index 2)   │
│     → retry with lesson-augmented prompt             │
│  6. Return SQL + confidence score                    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│                 Training Pipeline                     │
│  1. Upload CSV (NLQ+SQL or SQL only)                 │
│  2. Generate missing NLQs via LLM                    │
│  3. 80/20 split                                      │
│  4. 80% → Index 1 (kb_nlq_sql_pairs)                │
│  5. Run eval loop on 20%                             │
│  6. Failures → Critic → Lesson JSON → Index 2       │
│  7. Backfill 20% into Index 1                        │
│  8. Report metrics                                   │
└──────────────────────────────────────────────────────┘

┌─────────────────────┐   ┌─────────────────────────┐
│  Index 1 (FAISS)    │   │  Index 2 (FAISS)         │
│  kb_nlq_sql_pairs   │   │  kb_lessons_learned      │
│  + SQLite metadata  │   │  + SQLite metadata       │
└─────────────────────┘   └─────────────────────────┘
```

---

## Two-Index Design (Core Architecture)

**Index 1 — kb_nlq_sql_pairs:**
- Stores NLQ + gold SQL few-shot examples
- Used for retrieval during inference and eval
- Embed: the NLQ string

**Index 2 — kb_lessons_learned:**
- Stores structured lessons learned from failures
- Used during retry / lesson-augmented generation
- Embed: the `trigger` field (NLQ pattern that activates the lesson)

**Lesson JSON schema:**
```json
{
  "title": "Use DATE_TRUNC for month filtering",
  "trigger": "NLQ asks about filtering by month or time period",
  "diagnosis": "Used MONTH() which is not supported in DuckDB",
  "fix_rule": "Always use DATE_TRUNC('month', date_col) for month-level filtering",
  "error_category": "date_handling",
  "example": "{\"nlq\": \"...\", \"pred_sql\": \"...\", \"gold_sql\": \"...\"}"
}
```

---

## Module / Folder Structure

```
local-text2sql-rag/
├── app/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes_query.py        # POST /api/v1/query
│   │   └── routes_train.py        # POST /api/v1/train/upload, GET /api/v1/train/{id}
│   ├── core/
│   │   └── config.py              # Settings (API keys, paths, thresholds)
│   ├── db/
│   │   ├── sqlite_client.py       # SQLite connection + CRUD helpers
│   │   └── models.py              # Pydantic models + DDL
│   ├── embeddings/
│   │   └── embedder.py            # sentence-transformers wrapper
│   ├── vectorstore/
│   │   ├── faiss_index.py         # FAISS IndexFlatIP wrapper
│   │   └── kb_manager.py          # Two-index manager
│   ├── llm/
│   │   ├── claude_client.py       # Claude API wrapper: complete(prompt, temperature) -> str
│   │   └── prompts.py             # All templates: SQL_GEN, NLQ_GEN, CRITIC, LESSON_GEN
│   ├── sql/
│   │   ├── generator.py           # generate_sql(nlq, schema, examples, lessons=None) -> str
│   │   ├── validator.py           # validate_sql(sql) -> bool (SELECT/WITH only)
│   │   └── comparator.py          # compare(pred, gold) -> {ast_match: bool, token_sim: float}
│   ├── lessons/
│   │   ├── critic.py              # analyze(nlq, pred_sql, gold_sql) -> Lesson
│   │   └── generator.py           # generate_lesson(nlq, pred_sql, gold_sql, errors) -> Lesson
│   ├── evaluation/
│   │   ├── eval_loop.py           # Main eval loop (pass@1, retry, pass@2, hard_failure)
│   │   └── metrics.py             # Compute + report all metrics
│   ├── training/
│   │   └── pipeline.py            # End-to-end: CSV → embed → eval → lessons → metrics
│   └── inference/
│       └── pipeline.py            # query(nlq) -> {sql, confidence, used_lesson}
├── tests/
├── data/
│   ├── sample_pairs.csv
│   ├── sample_db.duckdb
│   └── indexes/                   # FAISS binary index files (gitignored)
├── docs/
│   ├── project_context.md         # THIS FILE
│   ├── architecture.md
│   ├── lessons_schema.md
│   └── prompt_design.md
└── scripts/
    └── seed_data.py
```

---

## Implementation Phases

### Phase 0 — Foundation ✅ DONE (committed: 9667d66)
- `app/core/config.py`
- `app/db/sqlite_client.py` + `app/db/models.py`
- `scripts/seed_data.py`

### Phase 1 — Embeddings + Vector Store ✅ DONE (committed: f461731)
- `app/embeddings/embedder.py` — sentence-transformers, all-MiniLM-L6-v2, 384-dim, L2-normalized
- `app/vectorstore/faiss_index.py` — IndexFlatIP wrapper
- `app/vectorstore/kb_manager.py` — two-index manager
- `tests/test_embedder.py` + `tests/test_kb_manager.py`

### Phase 2 — LLM + SQL Generation ✅ DONE (committed: 5cd0212)
Files built:
1. `app/llm/prompts.py` — four builder functions: `build_sql_gen_prompt`, `build_nlq_gen_prompt`, `build_critic_prompt`, `build_lesson_gen_prompt`. Each returns `(system, human)` tuple. `SCHEMA` constant with DuckDB DDL + inline value hints.
2. `app/llm/claude_client.py` — `LLMClient.complete(system, human, temperature) -> str`, `LLMError`. Uses `anthropic.types.TextBlock` isinstance check for type-safe response parsing.
3. `app/sql/generator.py` — `generate_sql(client, nlq, examples, lessons=None) -> str`. Temperature derived from `lessons is not None` (0.0 first attempt, 0.3 retry).
4. `app/sql/validator.py` — `validate_sql(sql) -> (bool, str | None)`. sqlglot parse + SELECT/WITH safety check. Returns error message string on failure.
5. `app/sql/comparator.py` — `compare(pred, gold) -> {ast_match: bool, token_sim: float}`. sqlglot DuckDB dialect normalization + Jaccard token similarity.
- `tests/test_comparator.py`, `tests/test_claude_client.py`, `tests/test_generator.py` — 57 tests, all passing.

**Key implementation notes:**
- `_make_client()` in tests returns `(LLMClient, MagicMock)` tuple — mock is explicit so Pylance doesn't complain about attribute access on `anthropic.Anthropic` type
- `validator.py` returns `(bool, str | None)` not just `bool` — error message used by lesson generator
- `comparator.py` uses `dialect="duckdb"` for normalization; parse failure → `ast_match=False`, token_sim still computed

### Phase 3 — Lesson Schema + Critic ✅ DONE (committed: 9b9920c)
Files built:
1. `app/lessons/critic.py` — `analyze(client, nlq, pred_sql, gold_sql) -> str`. Calls `build_critic_prompt`, returns raw numbered error list. Temperature=0.0.
2. `app/lessons/generator.py` — `generate_lesson(client, nlq, pred_sql, gold_sql, errors) -> Lesson`. Two-step: `_extract_json` (regex finds JSON in prose) → `_parse_lesson` (json.loads + Pydantic). `LessonGenerationError` carries `raw_output` for debugging.
- `tests/test_critic.py`, `tests/test_lesson_generator.py` — 28 tests, all passing.

**Key implementation notes:**
- Two-step critic → lesson: critic extracts specific errors first; lesson generator abstracts them into a generalizable rule. One-shot produces over-specific lessons.
- `_extract_json` and `_parse_lesson` are separate private helpers so tests can unit-test each step independently.
- `LessonGenerationError` (not `LLMError`) signals parse/validation failure — eval loop uses this distinction to mark hard failures without crashing.

### Phase 4 — Evaluation Loop + Metrics ✅ DONE (committed: a6500c3)
Files built:
1. `app/evaluation/eval_loop.py` — `run_eval(client, kb, eval_pairs, training_run_id) -> EvalRunStats`. Full two-attempt loop: attempt 1 → validate → compare → if fail: attempt 2 with lessons → critic → lesson gen → store in Index 2. `EvalRunStats` carries per-item flags (`pass1`, `pass2`, `hard_failure`, `retried`, `had_lessons`), `sim1_scores`, `sim2_scores`, `latencies_ms`.
2. `app/evaluation/metrics.py` — `compute_metrics(stats) -> MetricsReport`, `format_report(report) -> str`, `report_to_dict(report) -> dict`. All 9 metrics: pass@1/2, hard_failure, overall_pass_rate, kb_recovery_rate, lesson_utilization, avg_sim1/2, avg_latency_ms.
- `tests/test_eval_loop.py`, `tests/test_metrics.py` — 42 tests, all passing.

**Key implementation notes:**
- Lesson gen attempted for ALL pass@1 failures (not just hard failures) — maximizes self-learning
- `LessonGenerationError` and `LLMError` in lesson gen caught silently — never aborts the loop
- `sim2_scores` only populated for retry items — `avg_sim2=0.0` when no retries occurred
- `kb_recovery_rate=1.0` when no failures (vacuously correct)
- `report_to_dict()` uses `dataclasses.asdict()` for JSON serialization into SQLite `training_runs.metrics`

### Phase 5 — Training Pipeline ✅ DONE
Files built:
1. `app/sql/generator.py` — added `generate_nlq(client, sql) -> str` (temperature=0.0)
2. `app/training/pipeline.py` — `run(csv_path, client=None, kb=None) -> MetricsReport`. 12-step orchestration: load CSV → fill NLQs → 80/20 split → insert_training_run → index 80% → save checkpoint → run_eval on 20% → backfill → save final → compute_metrics → finish_training_run → return.
3. `tests/test_training_pipeline.py` — 26 tests, all passing.

**Key implementation notes:**
- `pipeline.py` is the orchestrator — it owns call order, delegates all domain logic to other modules.
- `run(csv_path, client=None, kb=None)`: None creates defaults. Tests inject mocks.
- `_load_csv`: 'sql' required, 'nlq' optional (defaults to ""), extra columns ignored, empty-sql rows dropped.
- `_fill_missing_nlqs`: LLMError → skip row with `warnings.warn`. One bad row does not abort.
- `_split`: raises `ValueError` if fewer than 2 pairs (can't satisfy ≥1 train + ≥1 eval). Uses `settings.random_seed` for reproducibility.
- Index 2 is populated inside `run_eval` (step 7) — pass@1 failures trigger critic → lesson → `kb.add_lesson`.
- `save_indexes()` called twice: before eval (checkpoint) and after backfill (final). Checkpoint protects Index 1 work if eval crashes mid-run.

### Phase 6 — Inference Pipeline ✅ DONE
Files built:
1. `app/inference/pipeline.py` — `query(nlq, client=None, kb=None) -> dict`. Returns `{sql, confidence, used_lesson}`.
2. `tests/test_inference_pipeline.py` — 17 tests, all passing.

**Key implementation notes:**
- `confidence = avg(top-k cosine similarity scores)` from Index 1 retrieval — the only proxy for correctness at inference time (no gold SQL available).
- Retry trigger: `confidence < settings.confidence_threshold OR invalid SQL`. Invalid SQL always worth retrying regardless of confidence.
- `used_lesson=True` only when Index 2 returned at least one lesson and it was passed to the retry prompt.
- `LLMError` on attempt 1 → return `{sql: "", confidence: 0.0, used_lesson: False}`. Never raises.
- `LLMError` on attempt 2 → keep attempt 1 SQL. Better to return a valid low-confidence result than empty string.
- `_avg_score` duplicated from eval_loop intentionally — sharing would create invisible coupling between two independent pipelines.

### Phase 7 — REST API 🔲
- `app/api/routes_query.py`, `app/api/routes_train.py`, `app/main.py`
- **Must-learn:** FastAPI request/response models, background tasks

### Phase 8 — Tests + Docs 🔲
- Full pytest suite, docs/

---

## Key Design Decisions

### What Must NOT Be Simplified (Soul of the System)
1. **Two-index design** — examples and lessons are separate indexes with separate lifecycles
2. **Structured Lesson JSON** — title/trigger/diagnosis/fix_rule/error_category/example; flattening loses reusability
3. **pass@2 with lesson-augmented retry** — retry must use a real lesson from Index 2, not just "try again"
4. **AST-based SQL comparison** — string equality is not enough; sqlglot structural equivalence
5. **Confidence from retrieval similarity** — computed from avg top-k cosine scores, not a random threshold
6. **Critic → Lesson generation loop** — real LLM call analyzing the failure; this IS the self-learning
7. **Prompt constraints** — SELECT/WITH only, no markdown, exact column names

### What Is Simplified (Safe for Local Version)
- Single-process synchronous (no Celery, no async workers)
- FAISS Flat (exact search, fine for <10k vectors)
- No auth / no multi-tenancy
- DuckDB instead of Snowflake
- Training = blocking function call (no job queue)
- Single-user SQLite

---

## Key Settings (app/core/config.py)

```python
embedding_model      = "all-MiniLM-L6-v2"
embedding_dim        = 384
top_k_examples       = 5
top_k_lessons        = 3
confidence_threshold = 0.75   # below this → trigger lesson retry
llm_model            = "claude-sonnet-4-6"
llm_temperature      = 0.0    # first attempt
llm_retry_temperature= 0.3    # retry with lessons
```

---

## Metrics to Track (Phase 4+)

| Metric | Description |
|---|---|
| pass@1 | % correct on first attempt |
| pass@2 | % correct on retry (after lesson injection) |
| hard_failure | % failed both attempts |
| overall_pass_rate | (pass@1 + pass@2) / total |
| lesson_utilization | % of retries that used a lesson from Index 2 |
| kb_recovery_rate | % of pass@1 failures recovered by pass@2 |
| avg_sim1 | avg cosine similarity of top-k examples retrieved |
| avg_sim2 | avg cosine similarity of lessons retrieved on retry |
| llm_cost_usd | estimated API cost |
| avg_latency_ms | avg end-to-end latency |

---

## API Shape

```
POST /api/v1/query              → {sql, confidence, used_lesson}
POST /api/v1/train/upload       → {run_id, status}
GET  /api/v1/train/{id}         → {run_id, status, metrics}
GET  /health                    → {status: "ok"}
```

---

## Evaluation Loop Logic

```
For each item in eval set:
  1. embed NLQ → retrieve top-k examples from Index 1
  2. generate SQL (attempt 1, temperature=0.0)
  3. compare pred vs gold → ast_match + token_sim
  4. if pass@1: store result, continue
  5. if fail:
       a. retrieve lessons from Index 2 (quick retry with existing lessons)
       b. generate SQL (attempt 2, temperature=0.3, lesson-augmented prompt)
       c. compare again
       d. if pass@2: store lesson in Index 2, mark pass@2
       e. if still fail: run critic → generate new lesson → store in Index 2
                         mark hard_failure
  6. after all evals: backfill eval set into Index 1
```

---

## Self-Learning Feedback Loop

```
retrieve examples → generate SQL → fail → critic → lesson → store lesson
       ↑                                                          │
       └──────────────── future queries benefit ─────────────────┘
```

The lesson is stored in Index 2 keyed by its `trigger` embedding. Next time a semantically similar NLQ arrives, the lesson is retrieved and injected into the retry prompt. This is the self-learning mechanism.
