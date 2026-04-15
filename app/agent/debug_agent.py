"""
SQL Debug Agent.

Demonstrates the core Agent pattern: LLM + tools + loop.

Given a broken SQL query and the original NLQ, the agent drives a multi-turn
conversation with Claude using the Anthropic tool_use API. Claude decides
when to call each tool; the agent dispatches tool calls and feeds results back
until Claude either produces a valid SQL or gives up.

Agent loop:
  1. Claude calls validate_sql   → learns whether the SQL is parseable and safe
  2. Claude calls analyze_errors → learns the specific logical / syntax mistakes
  3. Claude proposes a fixed SQL in a plain text response (end_turn)
  4. The agent validates the candidate; if still broken, tells Claude and loops
  5. Terminates on: valid SQL, "GIVE_UP" text, or max_iterations exhausted

Design decisions:
  - Two tools only: validate_sql and analyze_errors. No fix_sql tool — the LLM
    is the one doing the fixing. Tools give observations; the LLM reasons and acts.
    This is the authentic Agent pattern.
  - _dispatch_tool is a closure inside run() that captures client and nlq. This
    avoids module-level state and makes the function pure and testable.
  - complete_with_tools() returns a raw Message object. The agent inspects
    stop_reason and iterates over content blocks directly — complete() is not
    suitable here because it returns a plain string with no stop_reason.
  - analyze_errors passes broken_sql as both pred_sql and gold_sql to critic.analyze().
    There is no gold SQL in the debug context. The critic identifies internal issues
    (syntax, wrong functions, bad structure) rather than semantic correctness.
  - DebugResult.history records one entry per iteration for observability — useful
    when learning how the agent reasons through a failure.
  - LLMError propagates out of run(). The caller (API route) decides how to handle it.
"""

import json
from dataclasses import dataclass, field

import anthropic

from app.lessons import critic
from app.llm.claude_client import LLMClient
from app.sql.validator import validate_sql

# ── Tool definitions (Anthropic JSON Schema format) ────────────────────────────

_TOOLS: list[anthropic.types.ToolParam] = [
    {
        "name": "validate_sql",
        "description": (
            "Parse and safety-check a SQL string using sqlglot. "
            "Returns {valid: bool, error: string | null}. "
            "Call this first to check whether the SQL is syntactically correct "
            "and safe (SELECT / WITH only)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to validate.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "analyze_errors",
        "description": (
            "Use the SQL critic to identify specific logical or syntactic errors "
            "in a broken query. Returns {errors: string} — a numbered list of "
            "concrete mistakes. Call this after validate_sql reports a failure "
            "and before attempting a fix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "broken_sql": {
                    "type": "string",
                    "description": "The SQL query that failed validation.",
                },
                "error_message": {
                    "type": "string",
                    "description": "The error string returned by validate_sql.",
                },
            },
            "required": ["broken_sql", "error_message"],
        },
    },
]

_SYSTEM_PROMPT = """\
You are a SQL debugger. Your job is to fix a broken SQL query.

Steps:
1. Call validate_sql on the original broken SQL to understand the parse error.
2. Call analyze_errors to get a detailed list of what is wrong.
3. Output the corrected SQL as plain text with no markdown fences.

Rules:
- Only output SELECT or WITH queries. Never INSERT, UPDATE, DELETE, DROP, etc.
- If you cannot fix the SQL after using the tools, output exactly: GIVE_UP
- Output only the SQL (or GIVE_UP). No explanation text.
"""


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class DebugResult:
    """
    Result of one agent debug run.

    Attributes:
        fixed_sql:  The repaired SQL string, or None if the agent gave up.
        success:    True if fixed_sql passed validate_sql().
        iterations: Number of repair attempts made.
        history:    One entry per iteration describing what happened.
    """

    fixed_sql: str | None
    success: bool
    iterations: int
    history: list[str] = field(default_factory=list)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(
    client: LLMClient,
    nlq: str,
    broken_sql: str,
    max_iterations: int = 3,
) -> DebugResult:
    """
    Run the SQL debug agent loop.

    Drives a multi-turn conversation with Claude using tool_use. Claude calls
    validate_sql and analyze_errors to observe the problem, then proposes a fix.
    The agent validates the candidate; if still broken, it feeds the error back
    and retries up to max_iterations times.

    Args:
        client:         Shared LLMClient instance.
        nlq:            The original natural language question the SQL was for.
        broken_sql:     The SQL query to debug and fix.
        max_iterations: Maximum number of fix attempts before giving up.

    Returns:
        DebugResult with the fixed SQL (or None), success flag, iteration count,
        and per-iteration history.

    Raises:
        LLMError: if the Claude API call fails.
    """
    history: list[str] = []

    # ── Tool dispatch (closure captures client and nlq) ────────────────────────
    def dispatch_tool(tool_name: str, tool_input: dict) -> str:
        if tool_name == "validate_sql":
            ok, err = validate_sql(tool_input["sql"])
            return json.dumps({"valid": ok, "error": err})

        if tool_name == "analyze_errors":
            # No gold SQL in the debug context — pass broken_sql as both pred
            # and gold so the critic identifies internal issues, not semantic diff.
            # Prepend the validation error (from the previous validate_sql call)
            # to the nlq so the critic has the concrete parse error as context.
            error_message = tool_input.get("error_message", "")
            nlq_with_error = f"{nlq}\n\nValidation error: {error_message}" if error_message else nlq
            errors = critic.analyze(
                client,
                nlq=nlq_with_error,
                pred_sql=tool_input["broken_sql"],
                gold_sql=tool_input["broken_sql"],
            )
            return json.dumps({"errors": errors})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Initial conversation ───────────────────────────────────────────────────
    messages: list[anthropic.types.MessageParam] = [
        {
            "role": "user",
            "content": (
                f"Fix this SQL for the question: {nlq}\n\n"
                f"Broken SQL:\n{broken_sql}"
            ),
        }
    ]

    candidate_sql = broken_sql
    iterations = 0

    while iterations < max_iterations:
        response = client.complete_with_tools(_SYSTEM_PROMPT, messages, _TOOLS)

        # ── Tool use turn: dispatch all tool calls, then continue ──────────────
        if response.stop_reason == "tool_use":
            # Append the assistant's tool_use message to the conversation.
            messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

            # Build a single user message with all tool results.
            tool_results: list[anthropic.types.ToolResultBlockParam] = []
            tool_names = []
            for block in response.content:
                if isinstance(block, anthropic.types.ToolUseBlock):
                    result_str = dispatch_tool(block.name, dict(block.input))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                    tool_names.append(block.name)

            messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]
            history.append(f"iteration {iterations + 1}: called tools {tool_names}")
            # Don't increment iterations — tool calls don't count as fix attempts.
            continue

        # ── End turn: extract the SQL candidate ───────────────────────────────
        iterations += 1
        text_block = next(
            (b for b in response.content if isinstance(b, anthropic.types.TextBlock)),
            None,
        )
        candidate_sql = _strip_fences(text_block.text if text_block else "")

        if candidate_sql == "GIVE_UP":
            history.append(f"iteration {iterations}: agent gave up")
            return DebugResult(
                fixed_sql=None,
                success=False,
                iterations=iterations,
                history=history,
            )

        ok, err = validate_sql(candidate_sql)
        if ok:
            history.append(f"iteration {iterations}: fixed SQL validated successfully")
            return DebugResult(
                fixed_sql=candidate_sql,
                success=True,
                iterations=iterations,
                history=history,
            )

        # SQL still broken — tell Claude and loop.
        history.append(f"iteration {iterations}: candidate SQL still invalid — {err}")
        messages.append({"role": "assistant", "content": candidate_sql})
        messages.append({
            "role": "user",
            "content": f"That SQL is still invalid: {err}. Please fix it.",
        })

    # Max iterations exhausted.
    history.append(f"max iterations ({max_iterations}) reached without a valid fix")
    return DebugResult(
        fixed_sql=candidate_sql,
        success=False,
        iterations=iterations,
        history=history,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from a SQL candidate string."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
