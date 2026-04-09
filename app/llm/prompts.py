"""
All LLM prompt templates for the text-to-SQL RAG system.

Design decisions:
  - Each prompt is split into (system, human) — matching the Claude Messages API structure.
    The system prompt is static (role + hard rules); the human message carries dynamic content
    (schema, examples, lessons, the actual question).
  - Builder functions (not raw string constants) because prompts have conditional sections
    (e.g., lessons block only appears on retry). A function makes that branching explicit.
  - SCHEMA is a module-level constant. It's static for this project — DuckDB tables don't
    change at runtime. Inline comments on category/status values prevent hallucinated literals.
  - All four prompt types are defined here (sql_gen, nlq_gen, critic, lesson_gen) even though
    critic and lesson_gen aren't used until Phase 3. All prompt engineering lives in one place.
"""

# ── Database schema ────────────────────────────────────────────────────────────
# Injected into every SQL generation prompt.
# DDL format: gives the LLM precise column names, types, FK relationships.
# Inline value hints on category/status prevent hallucinated literals in WHERE clauses.

SCHEMA = """\
-- Database: DuckDB (PostgreSQL-compatible dialect)

customers (
    customer_id  INTEGER PRIMARY KEY,
    name         VARCHAR,
    email        VARCHAR,
    country      VARCHAR,        -- e.g. 'Germany', 'USA', 'UK', 'Spain', 'China', 'Italy', ...
    created_at   DATE
)

products (
    product_id  INTEGER PRIMARY KEY,
    name        VARCHAR,
    category    VARCHAR,         -- values: 'Electronics', 'Books', 'Furniture', 'Clothing'
    price       DECIMAL(10,2)
)

orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    order_date    DATE,
    total_amount  DECIMAL(10,2),
    status        VARCHAR         -- values: 'completed', 'pending', 'cancelled'
)

order_items (
    item_id     INTEGER PRIMARY KEY,
    order_id    INTEGER REFERENCES orders(order_id),
    product_id  INTEGER REFERENCES products(product_id),
    quantity    INTEGER,
    unit_price  DECIMAL(10,2)
)

-- Notes:
-- orders.total_amount  = pre-computed order-level total (use for order revenue)
-- unit_price * quantity = item-level revenue (use when aggregating by product)
-- For month-level date ops: DATE_TRUNC('month', date_col)  [not MONTH() or STRFTIME]
-- For date differences:    date_diff('day', start_date, end_date)
"""

# ── System prompts (static) ────────────────────────────────────────────────────

_SQL_GEN_SYSTEM = """\
You are an expert SQL analyst for a DuckDB database.

Rules — follow these exactly:
- Output ONLY the SQL query, nothing else.
- Use SELECT or WITH statements only. Never INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL.
- No markdown fences, no backticks, no explanation, no comments in output.
- Use exact table and column names from the schema — never invent columns.
- Do not add ORDER BY or LIMIT unless the question explicitly asks for ranked or limited results.
- Always alias aggregation columns (e.g., COUNT(*) AS total_count, SUM(price) AS total_revenue).
- Apply these mappings:
    "how many" / "count"      → COUNT(*)
    "total" / "sum of"        → SUM(col)
    "average" / "avg"         → AVG(col)
    "most" / "top" / "highest"→ ORDER BY col DESC LIMIT n
- DuckDB date functions: DATE_TRUNC('month', col), INTERVAL '1 month', date_diff('day', a, b).
- Follow the join and filter patterns from the examples as closely as possible.
- Keep SQL simple — do not over-engineer.\
"""

_NLQ_GEN_SYSTEM = """\
You are a data analyst who writes clear, natural English questions about database data.

Rules:
- Output ONLY the natural language question, nothing else.
- The question must be directly answerable by the provided SQL query.
- Write as a business user would ask — plain English, no SQL jargon.
- Do not mention table names, column names, or SQL syntax in the question.
- Keep it concise — one sentence.\
"""

_CRITIC_SYSTEM = """\
You are a SQL expert who diagnoses why a generated SQL query produced the wrong result.

Rules:
- Be specific and concise.
- List only actual logical errors — not style preferences.
- Focus on: wrong table, wrong column, wrong filter value, wrong aggregation, wrong join condition.
- Number each error on its own line.\
"""

_LESSON_GEN_SYSTEM = """\
You are a SQL teaching expert who creates reusable lessons from SQL generation failures.

A good lesson:
- Captures a GENERAL pattern, not a query-specific fact.
- Has a trigger that describes the NLQ pattern broadly (not the specific question text).
- Has a fix_rule that is a concrete, actionable instruction for future SQL generation.
- Assigns one error category: date_handling | join_logic | aggregation | filter | other

Rules:
- Output ONLY valid JSON — no explanation, no markdown fences, no trailing text.
- The trigger must be general enough to match future similar questions.
- The fix_rule must start with an action verb (e.g., "Always use ...", "Never use ...", "When ... use ...").\
"""


# ── Builder functions ──────────────────────────────────────────────────────────
# Each function returns (system_prompt: str, human_message: str).
# Callers pass this tuple directly to claude_client.complete().

def build_sql_gen_prompt(
    nlq: str,
    examples: list[dict],
    lessons: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Build the SQL generation prompt.

    First attempt:  lessons=None  → no lessons section, temperature=0.0 (in generator.py)
    Retry attempt:  lessons=[...] → lessons section injected, temperature=0.3

    Args:
        nlq:      The natural language question.
        examples: List of {nlq, sql, score} dicts from KBManager.search_examples().
        lessons:  Optional list of {title, trigger, fix_rule, error_category, ...} dicts.

    Returns:
        (system_prompt, human_message) tuple.
    """
    # ── Examples block ────────────────────────────────────────────────────────
    if examples:
        ex_lines = []
        for i, ex in enumerate(examples, 1):
            ex_lines.append(
                f"-- Example {i} (similarity: {ex['score']:.2f})\n"
                f"-- Question: {ex['nlq']}\n"
                f"{ex['sql']}"
            )
        examples_block = "\n\n".join(ex_lines)
    else:
        examples_block = "-- No examples available."

    # ── Lessons block (retry only) ─────────────────────────────────────────────
    if lessons:
        lesson_lines = []
        for i, lesson in enumerate(lessons, 1):
            lesson_lines.append(
                f"[{i}] {lesson['title']} (category: {lesson['error_category']})\n"
                f"    Trigger:  {lesson['trigger']}\n"
                f"    Fix rule: {lesson['fix_rule']}"
            )
        lessons_section = (
            "\n### Lessons Learned\n"
            "The following lessons were learned from past failures on similar questions.\n"
            "Apply the relevant fix rules when writing the SQL below.\n\n"
            + "\n\n".join(lesson_lines)
        )
    else:
        lessons_section = ""

    human = (
        f"### Schema\n{SCHEMA}\n"
        f"### Examples\n{examples_block}"
        f"{lessons_section}\n\n"
        f"### Question\n{nlq}"
    )

    return _SQL_GEN_SYSTEM, human


def build_nlq_gen_prompt(sql: str) -> tuple[str, str]:
    """
    Build the prompt that generates a natural language question from a SQL query.

    Used in the training pipeline when the input CSV has SQL but no NLQ.

    Args:
        sql: The SQL query to reverse into a natural language question.

    Returns:
        (system_prompt, human_message) tuple.
    """
    human = (
        f"Given this SQL query against our database, write the natural language question it answers.\n\n"
        f"### Schema\n{SCHEMA}\n"
        f"### SQL\n{sql}"
    )
    return _NLQ_GEN_SYSTEM, human


def build_critic_prompt(nlq: str, pred_sql: str, gold_sql: str) -> tuple[str, str]:
    """
    Build the critic prompt that identifies errors in a wrong SQL prediction.

    Used in Phase 3 (lessons/critic.py) after a hard failure.

    Args:
        nlq:      The natural language question.
        pred_sql: The SQL generated by the model (wrong).
        gold_sql: The correct gold SQL.

    Returns:
        (system_prompt, human_message) tuple.
    """
    human = (
        f"A SQL query was generated for the following question but produced the wrong result.\n"
        f"Identify the specific errors.\n\n"
        f"### Question\n{nlq}\n\n"
        f"### Predicted SQL (wrong)\n{pred_sql}\n\n"
        f"### Gold SQL (correct)\n{gold_sql}\n\n"
        f"List each error on a numbered line. Be specific."
    )
    return _CRITIC_SYSTEM, human


def build_lesson_gen_prompt(
    nlq: str,
    pred_sql: str,
    gold_sql: str,
    errors: str,
) -> tuple[str, str]:
    """
    Build the prompt that generates a structured lesson JSON from a failure.

    Used in Phase 3 (lessons/generator.py) after the critic has identified errors.

    Args:
        nlq:      The natural language question.
        pred_sql: The SQL generated by the model (wrong).
        gold_sql: The correct gold SQL.
        errors:   The critic's error list (plain text, numbered lines).

    Returns:
        (system_prompt, human_message) tuple.
    """
    human = (
        f"Generate a reusable lesson from this SQL generation failure.\n\n"
        f"### Question\n{nlq}\n\n"
        f"### Predicted SQL (wrong)\n{pred_sql}\n\n"
        f"### Gold SQL (correct)\n{gold_sql}\n\n"
        f"### Errors identified\n{errors}\n\n"
        f"Output a JSON object with exactly these fields:\n"
        f'{{\n'
        f'  "title": "short name for this lesson",\n'
        f'  "trigger": "what NLQ pattern triggers this lesson (general, not this specific question)",\n'
        f'  "diagnosis": "why the predicted SQL was wrong",\n'
        f'  "fix_rule": "actionable rule starting with a verb: Always/Never/When X use Y",\n'
        f'  "error_category": "date_handling | join_logic | aggregation | filter | other",\n'
        f'  "example": "{{\\"nlq\\": \\"..\\", \\"pred_sql\\": \\"..\\", \\"gold_sql\\": \\"..\\"}}" \n'
        f"}}"
    )
    return _LESSON_GEN_SYSTEM, human
