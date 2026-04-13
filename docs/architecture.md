# Architecture — Local Text-to-SQL RAG System

## System Overview

```
HTTP Client
     │
     ▼
┌──────────────────────────────────┐
│         FastAPI (app/main.py)    │
│  GET  /health                    │
│  POST /api/v1/query              │
│  POST /api/v1/train/upload       │
│  GET  /api/v1/train/{id}         │
└──────┬───────────────┬───────────┘
       │               │
       ▼               ▼
  Inference        Training
  Pipeline         Pipeline
  (sync)           (BackgroundTask)
```

---

## Inference Pipeline (`app/inference/pipeline.py`)

```
NLQ
 │
 ├─ kb.search_examples(nlq)        Index 1 → top-k {nlq, sql, score}
 │
 ├─ confidence = avg(scores)
 │
 ├─ generate_sql(client, nlq, examples)    attempt 1, temperature=0.0
 │
 ├─ validate_sql(sql)
 │
 ├─ if confidence < 0.75 OR invalid:
 │     kb.search_lessons(nlq)      Index 2 → top-k lessons
 │     generate_sql(..., lessons)  attempt 2, temperature=0.3
 │
 └─ return {sql, confidence, used_lesson}
```

**Key invariant:** inference never raises. LLMError on attempt 1 → `{sql: "", confidence: 0.0}`. LLMError on attempt 2 → keep attempt 1 result.

---

## Training Pipeline (`app/training/pipeline.py`)

```
CSV file
 │
 ├─ _load_csv()                    parse, require 'sql', drop empty rows
 ├─ _fill_missing_nlqs()           LLM generates NLQ from SQL if empty
 ├─ _split()                       80% train / 20% eval (reproducible seed)
 │
 ├─ insert_training_run()          SQLite: status='running'
 │
 ├─ for each train pair:
 │     insert_nlq_sql_pair()       SQLite
 │     kb.add_example()            Index 1
 │
 ├─ kb.save_indexes()              CHECKPOINT (protects Index 1 if eval crashes)
 │
 ├─ run_eval()                     eval loop on 20%
 │     └─ on pass@1 failure:
 │           critic → generate_lesson → kb.add_lesson    Index 2
 │
 ├─ for each eval pair:
 │     kb.add_example()            Index 1 (backfill)
 │
 ├─ kb.save_indexes()              FINAL STATE
 │
 ├─ compute_metrics()
 ├─ finish_training_run()          SQLite: status='done', metrics=JSON
 └─ return MetricsReport
```

---

## Two-Index Design

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  Index 1 — kb_nlq_sql_pairs │     │  Index 2 — kb_lessons       │
│                             │     │                             │
│  Embed: NLQ string          │     │  Embed: trigger field       │
│  Store: NLQ + gold SQL      │     │  Store: full Lesson JSON    │
│                             │     │                             │
│  Used for:                  │     │  Used for:                  │
│  - Few-shot examples        │     │  - Retry augmentation       │
│  - Confidence score         │     │  - Self-learning feedback   │
└─────────────────────────────┘     └─────────────────────────────┘
        populated by                         populated by
        training pipeline                    eval loop (failures only)
        + backfill
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `app/core/config.py` | All settings — paths, thresholds, model names |
| `app/db/sqlite_client.py` | Raw SQLite CRUD, no business logic |
| `app/embeddings/embedder.py` | sentence-transformers wrapper, L2-normalize |
| `app/vectorstore/faiss_index.py` | FAISS IndexFlatIP: add, search, save, load |
| `app/vectorstore/kb_manager.py` | Two-index coordinator |
| `app/llm/claude_client.py` | Claude API wrapper, fence stripping, LLMError |
| `app/llm/prompts.py` | All prompt builders, SCHEMA constant |
| `app/sql/generator.py` | generate_sql, generate_nlq — wires prompts + client |
| `app/sql/validator.py` | sqlglot AST safety check — SELECT/WITH only |
| `app/sql/comparator.py` | AST match + Jaccard token similarity |
| `app/lessons/critic.py` | Extract specific errors from a failure |
| `app/lessons/generator.py` | Abstract errors into a reusable Lesson |
| `app/evaluation/eval_loop.py` | pass@1 / pass@2 / hard_failure loop |
| `app/evaluation/metrics.py` | Compute + format + serialize MetricsReport |
| `app/training/pipeline.py` | Orchestrator: CSV → train → eval → metrics |
| `app/inference/pipeline.py` | Orchestrator: NLQ → retrieve → generate → retry |
| `app/api/routes_query.py` | POST /api/v1/query |
| `app/api/routes_train.py` | POST /api/v1/train/upload, GET /api/v1/train/{id} |
| `app/main.py` | FastAPI app, router mounting, startup |

---

## Data Flow: Self-Learning Loop

```
  Training run N:
    eval failure → critic → lesson → stored in Index 2

  Training run N+1 (or inference):
    similar NLQ arrives → lesson retrieved from Index 2
    → injected into retry prompt → higher chance of pass@2
```

Each training run makes the system smarter for future runs.

---

## SQLite Schema

```
nlq_sql_pairs        — training + eval pairs, links to FAISS index ID
lessons              — lessons learned from failures
training_runs        — one record per pipeline.run() call, status + metrics JSON
eval_results         — per-item outcome of each eval loop run
```

---

## Key Settings

| Setting | Value | Purpose |
|---|---|---|
| `embedding_model` | all-MiniLM-L6-v2 | 384-dim, fast, good quality |
| `confidence_threshold` | 0.75 | Below this → trigger retry |
| `eval_split` | 0.20 | 20% held out for eval |
| `top_k_examples` | 5 | Few-shot examples per query |
| `top_k_lessons` | 3 | Lessons per retry |
| `llm_temperature` | 0.0 | Deterministic first attempt |
| `llm_retry_temperature` | 0.3 | Slight variation on retry |
