# Claude Code Instructions — local-text2sql-rag

## Agent Tool Implementation Rules

When writing or modifying agent tool dispatch code (`app/agent/`):

1. **Every `required` field in a tool's `input_schema` MUST be accessed via
   `tool_input["field_name"]` in the dispatch implementation.** After writing
   dispatch code, verify each required field is actually read and used —
   not just declared in the schema.

2. **Test assertions must verify exact arguments to downstream functions**, not
   just that they were called. Use `mock.assert_called_once_with(...)` rather
   than `mock.assert_called_once()`. This catches silent parameter drops.

3. **Schema and implementation must stay in sync.** If a required field is
   removed from `input_schema`, remove the corresponding access in dispatch
   too (and vice versa).

## Testing Rules

- Run `uv run pytest` before every commit. All 247+ tests must pass.
- New agent features require tests covering: tool dispatch arguments, loop
  convergence, retry behavior, give-up path, and LLMError propagation.
- Mock `critic.analyze` in all agent tests — never make real LLM calls.
