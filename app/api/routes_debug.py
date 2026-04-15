"""
Debug route: POST /api/v1/debug

Accepts a broken SQL query and the original NLQ, runs the SQL debug agent,
and returns the fixed SQL (or a failure report if the agent could not fix it).
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent import debug_agent
from app.llm.claude_client import LLMClient

router = APIRouter(prefix="/api/v1")


class DebugRequest(BaseModel):
    nlq: str
    broken_sql: str
    max_iterations: int = 3


class DebugResponse(BaseModel):
    fixed_sql: str | None
    success: bool
    iterations: int
    history: list[str]


@router.post("/debug", response_model=DebugResponse)
def handle_debug(request: DebugRequest) -> DebugResponse:
    """
    Debug a broken SQL query using the SQL debug agent.

    The agent calls validate_sql and analyze_errors as tools, then proposes
    a fix. Retries up to max_iterations times if the fix is still invalid.
    Returns fixed_sql=null and success=false if the agent cannot fix the SQL.
    """
    client = LLMClient()
    result = debug_agent.run(
        client,
        nlq=request.nlq,
        broken_sql=request.broken_sql,
        max_iterations=request.max_iterations,
    )
    return DebugResponse(
        fixed_sql=result.fixed_sql,
        success=result.success,
        iterations=result.iterations,
        history=result.history,
    )
