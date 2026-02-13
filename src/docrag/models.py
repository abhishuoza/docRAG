"""Shared Pydantic data models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ScrapedDocument(BaseModel):
    """Represents a document scraped from a URL."""

    url: str
    title: str
    content: str
    scraped_at: datetime = Field(default_factory=datetime.now)


class TextChunk(BaseModel):
    """A chunk of text after splitting a document."""

    text: str
    source_url: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with relevance score."""

    text: str
    source_url: str
    score: float
    chunk_index: int


class GenerationRequest(BaseModel):
    """Request payload for code generation."""

    query: str
    doc_urls: list[str]
    top_k: int = 5
    search_query: str | None = None


class GenerationResponse(BaseModel):
    """Response payload with generated code and references."""

    query: str
    generated_code: str
    references: list[RetrievedChunk]
    model_used: str
    low_relevance: bool = False