# Lesson Schema

## What a Lesson Is

A lesson is a structured, generalizable rule extracted from a SQL generation failure. It captures not just *what went wrong* but *why*, and *how to avoid it next time* — in a form general enough to help with future, semantically similar queries.

Lessons live in Index 2 (FAISS), keyed by their `trigger` embedding. When a retry is needed, the trigger is searched against the incoming NLQ to find relevant lessons.

---

## JSON Schema

```json
{
  "title":          "Use DATE_TRUNC for month filtering",
  "trigger":        "NLQ asks about filtering by month or time period",
  "diagnosis":      "Used MONTH() which is not supported in DuckDB",
  "fix_rule":       "Always use DATE_TRUNC('month', date_col) for month-level filtering",
  "error_category": "date_handling",
  "example": "{\"nlq\": \"orders last month\", \"pred_sql\": \"WHERE MONTH(created_at) = ...\", \"gold_sql\": \"WHERE DATE_TRUNC('month', created_at) = ...\"}"
}
```

---

## Field Semantics

| Field | Type | Purpose |
|---|---|---|
| `title` | str | Short human-readable name for the lesson |
| `trigger` | str | **The embedding key.** Must be a general NLQ pattern, not a specific question |
| `diagnosis` | str | What specifically went wrong in the failure that generated this lesson |
| `fix_rule` | str | The actionable rule — what the LLM should do differently |
| `error_category` | str | One of: `date_handling`, `aggregation`, `join`, `filtering`, `subquery`, `other` |
| `example` | str | JSON-encoded `{nlq, pred_sql, gold_sql}` from the original failure |

---

## Why `trigger` Must Be General

The trigger is embedded and searched against future NLQs at retrieval time. If it's too specific, it will only match the exact failure it came from — defeating the purpose of self-learning.

**Too specific (bad):**
```
"NLQ: how many orders were placed last month"
```

**General enough (good):**
```
"NLQ asks about filtering by month or time period"
```

The second form will match "revenue last month", "sales in March", "orders this quarter" — all queries that might hit the same DuckDB incompatibility.

---

## Two-Step Generation: Critic → Lesson

Lessons are generated in two steps to improve quality:

**Step 1 — Critic (`app/lessons/critic.py`):**
Extracts specific, concrete errors from the failure:
```
1. Used MONTH() function — not supported in DuckDB
2. Missing GROUP BY clause for aggregation
```

**Step 2 — Lesson Generator (`app/lessons/generator.py`):**
Abstracts the concrete errors into a generalizable lesson with a trigger general enough to match future queries.

**Why two steps?** A single prompt producing both diagnosis and generalization tends to over-fit to the specific failure. The two-step process forces separation between "what happened here" and "what should we do in general."

---

## Error Categories

| Category | Examples |
|---|---|
| `date_handling` | MONTH() vs DATE_TRUNC, date format issues |
| `aggregation` | Missing GROUP BY, wrong aggregate function |
| `join` | Wrong join type, missing join condition |
| `filtering` | Wrong WHERE clause, HAVING vs WHERE |
| `subquery` | Incorrect subquery structure, correlated subqueries |
| `other` | Anything that doesn't fit the above |

---

## Pydantic Model

```python
# app/db/models.py
class Lesson(BaseModel):
    title: str
    trigger: str
    diagnosis: str
    fix_rule: str
    error_category: str
    example: str        # JSON string of {nlq, pred_sql, gold_sql}
```

Validation is enforced at parse time — `LessonGenerationError` is raised if the LLM output cannot be parsed into this shape.
