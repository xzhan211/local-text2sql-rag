# Prompt Design

## Overview

There are four prompts in the system, all defined in `app/llm/prompts.py`. Each returns a `(system, human)` tuple passed to `LLMClient.complete()`.

| Prompt | Function | Called by |
|---|---|---|
| SQL generation | `build_sql_gen_prompt` | `sql/generator.py` |
| NLQ generation | `build_nlq_gen_prompt` | `sql/generator.py` |
| Critic | `build_critic_prompt` | `lessons/critic.py` |
| Lesson generation | `build_lesson_gen_prompt` | `lessons/generator.py` |

---

## 1. SQL Generation Prompt

**Purpose:** Generate a SQL query from a natural language question.

**Two modes:**
- **Attempt 1** (`lessons=None`): few-shot examples only, `temperature=0.0`
- **Attempt 2** (`lessons=[...]`): examples + lessons, `temperature=0.3`

**Key constraints in the system prompt:**
- Output raw SQL only — no markdown fences, no explanation
- Use only SELECT or WITH statements
- Use exact column names from the schema
- DuckDB dialect (DATE_TRUNC, not MONTH(); QUALIFY for window filtering)

**Why temperature=0.3 on retry?**
Attempt 1 was deterministic (0.0) and produced wrong SQL. Attempt 2 needs some variation to escape the failure mode — but not so much randomness that it ignores the lesson guidance. 0.3 is a small creative nudge.

**Why few-shot examples?**
Retrieved examples anchor the LLM to the specific schema and query patterns seen during training. Without them, the LLM generates generic SQL that may use column names or table structures that don't match our DuckDB schema.

---

## 2. NLQ Generation Prompt

**Purpose:** Generate a natural language question from a SQL query.

**Used by:** Training pipeline, when the input CSV has SQL but no NLQ.

**Key design:**
- Output a plain English question only — no explanation
- The question should be answerable by the SQL (not broader, not narrower)
- `temperature=0.0` — deterministic, the reverse mapping should be stable

**Why generate NLQ from SQL (not the reverse)?**
Training data often comes from SQL analysts who write queries but don't document the business question. Generating the NLQ fills this gap, enabling the pair to be stored in Index 1 and retrieved by future natural language queries.

---

## 3. Critic Prompt

**Purpose:** Extract a numbered list of specific SQL errors from a failure.

**Input:** `(nlq, pred_sql, gold_sql)`
**Output:** Plain text numbered list of errors, e.g.:
```
1. Used MONTH() which is not supported in DuckDB
2. Missing GROUP BY clause — aggregation over multiple rows
```

**Key design:**
- Output errors only — no preamble, no explanation of what the SQL does
- Be concrete: reference specific functions, clauses, column names
- `temperature=0.0` — error analysis should be deterministic

**Why a separate critic step?**
One-shot "generate a lesson from this failure" produces over-specific lessons that only match the exact failure. The critic forces concrete error extraction first, which the lesson generator then abstracts into a generalizable rule.

---

## 4. Lesson Generation Prompt

**Purpose:** Abstract critic errors into a reusable lesson JSON.

**Input:** `(nlq, pred_sql, gold_sql, errors)` — errors is the critic's numbered list
**Output:** A JSON object matching the Lesson schema

**Key constraints in the prompt:**
- `trigger` must be a general NLQ pattern, not the specific question
- `fix_rule` must be actionable — a rule the LLM can apply, not a description
- Output valid JSON only — the response is parsed with `json.loads`

**Parsing strategy:**
The LLM sometimes wraps JSON in prose ("Here is the lesson: {...}"). `_extract_json` uses `re.search(r"\{.*\}", text, re.DOTALL)` to find the JSON object regardless of surrounding text. `_parse_lesson` then validates it against the Pydantic `Lesson` model.

**Error handling:**
`LessonGenerationError` (not `LLMError`) signals parse/validation failure. It carries `raw_output` for debugging. The eval loop catches this silently — a failed lesson generation never aborts the eval loop.

---

## Schema Constant (`SCHEMA`)

All four prompts reference a `SCHEMA` constant defined in `prompts.py`. It contains:
- DuckDB DDL for all tables (`customers`, `orders`, `order_items`, `products`)
- Inline value hints for categorical columns (e.g., `status IN ('pending', 'shipped', 'delivered')`)

**Why inline value hints?**
Without them, the LLM may generate `WHERE status = 'active'` when the only valid value is `'pending'`. Inline hints are cheaper than a dynamic schema lookup and sufficient for a fixed-schema system.
