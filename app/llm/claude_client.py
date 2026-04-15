"""
Thin wrapper around the Anthropic Claude Messages API.

Responsibility: send a (system, human) prompt pair, return the response as a
plain string. Nothing else. No prompt building, no retry logic, no SQL parsing.

Design decisions:
  - LLMClient is a class, not a module-level function. The caller (generator.py,
    lessons/critic.py, etc.) instantiates one LLMClient per pipeline run and reuses
    it across calls. This avoids re-reading the API key on every call while keeping
    the interface explicit (no hidden global state).
  - complete() takes system + human separately — matching the Claude Messages API
    structure. This pairs naturally with the (system, human) tuples returned by
    prompts.py builder functions.
  - complete_with_tools() is the agentic variant. It accepts a multi-turn message
    list and a tools definition, and returns the raw Message object so the caller
    can inspect stop_reason and iterate over ToolUseBlock / TextBlock content.
    It does NOT strip fences — the agent handles that itself on the final SQL candidate.
  - Markdown fence stripping is done here (for complete()), not in generator.py.
    The client is the last place that touches raw API output; stripping here means
    every caller gets a clean string with no fences regardless of which prompt was used.
  - LLMError wraps all anthropic exceptions. This keeps pipeline code free of
    Anthropic-specific exception types — pipelines catch LLMError only.
  - No module-level Anthropic client singleton. Unlike sentence-transformers (300MB
    model load), the Anthropic client is a thin HTTP wrapper — cheap to instantiate.
"""

import re

import anthropic

from app.core.config import settings


class LLMError(Exception):
    """
    Raised when the Claude API call fails for any reason.

    Wraps anthropic exceptions so pipeline code never imports from anthropic directly.

    Attributes:
        cause: The original exception, if any.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class LLMClient:
    """
    Claude API client.

    Instantiate once per pipeline run and reuse across calls:

        client = LLMClient()
        sql = client.complete(system, human, temperature=0.0)

    Raises:
        LLMError: on any API failure (auth, rate limit, network, etc.)
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, system: str, human: str, temperature: float = 0.0) -> str:
        """
        Send a system + human prompt to Claude, return the response as a plain string.

        Markdown fences are stripped automatically. The returned string is ready to
        use as SQL, a question, or JSON — depending on which prompt was used.

        Args:
            system:      The system prompt (role definition + hard rules).
            human:       The human message (schema, examples, question, etc.).
            temperature: Sampling temperature. Use 0.0 for deterministic output
                         (first SQL attempt), 0.3 for slight variation (retry).

        Returns:
            The model's response as a stripped plain string.

        Raises:
            LLMError: wraps any anthropic.APIError subclass.
        """
        try:
            response = self._client.messages.create(
                model=settings.llm_model,
                max_tokens=1024,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": human}],
            )
            text_block = next(
                (b for b in response.content if isinstance(b, anthropic.types.TextBlock)),
                None,
            )
            if text_block is None:
                raise LLMError("Claude response contained no text block")
            return self._strip_fences(text_block.text)
        except anthropic.APIError as e:
            raise LLMError(f"Claude API call failed: {e}", cause=e) from e

    def complete_with_tools(
        self,
        system: str,
        messages: list[anthropic.types.MessageParam],
        tools: list[anthropic.types.ToolParam],
        temperature: float = 0.0,
    ) -> anthropic.types.Message:
        """
        Send a multi-turn conversation with tool definitions to Claude.

        Used by the debug agent loop. Unlike complete(), this method:
          - Accepts a full message list (multi-turn conversation history).
          - Accepts a tools definition list (JSON Schema format).
          - Returns the raw Message object, not a plain string, so the caller
            can inspect stop_reason ("tool_use" vs "end_turn") and iterate
            over ToolUseBlock / TextBlock content blocks.
          - Does NOT strip markdown fences — the agent handles that selectively
            on the final SQL candidate only.

        Args:
            system:      The system prompt.
            messages:    Conversation history as a list of MessageParam objects.
            tools:       Tool definitions as a list of ToolParam objects.
            temperature: Sampling temperature. Default 0.0 for deterministic output.

        Returns:
            The raw anthropic.types.Message object.

        Raises:
            LLMError: wraps any anthropic.APIError subclass.
        """
        try:
            return self._client.messages.create(
                model=settings.llm_model,
                max_tokens=1024,
                temperature=temperature,
                system=system,
                messages=messages,
                tools=tools,
            )
        except anthropic.APIError as e:
            raise LLMError(f"Claude API call failed: {e}", cause=e) from e

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        """
        Remove markdown code fences from model output.

        Claude occasionally wraps output in ```sql ... ``` or ``` ... ``` even
        when instructed not to. Strip defensively so callers always get clean text.

        Handles:
            ```sql\\nSELECT 1\\n```  →  SELECT 1
            ```\\nSELECT 1\\n```     →  SELECT 1
            SELECT 1                 →  SELECT 1  (no-op)
        """
        text = text.strip()
        text = re.sub(r"^```(?:sql)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
