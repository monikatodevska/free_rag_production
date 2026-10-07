from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    title: str | None = None


class AskRequest(SearchRequest):
    pass


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_chunks: list[SearchResult]
