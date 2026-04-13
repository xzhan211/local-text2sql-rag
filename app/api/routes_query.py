"""
Query route: POST /api/v1/query

Accepts a natural language question, runs the inference pipeline, and
returns the generated SQL with a confidence score.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.inference.pipeline import query

router = APIRouter(prefix="/api/v1")


class QueryRequest(BaseModel):
    nlq: str


class QueryResponse(BaseModel):
    sql: str
    confidence: float
    used_lesson: bool


@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Generate SQL from a natural language question.

    Uses the inference pipeline: retrieves examples, generates SQL,
    retries with lessons if confidence is low or SQL is invalid.
    """
    result = query(request.nlq)
    return QueryResponse(**result)
