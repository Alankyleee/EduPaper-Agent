from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    QA = "qa"
    REVIEW = "review"
    COMPARE = "compare"


class Citation(BaseModel):
    citation_id: str
    source: str
    page: int | None = None
    section: str | None = None
    snippet: str
    url: str | None = None
    document_id: str | None = None
    score: float | None = None


class TraceEvent(BaseModel):
    node: str
    detail: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    mode: QueryMode = QueryMode.QA
    top_k: int = Field(default=6, ge=1, le=15)
    document_ids: list[str] | None = None
    allow_arxiv_search: bool = True


class QueryResponse(BaseModel):
    run_id: str
    answer: str
    citations: list[Citation]
    used_arxiv: bool
    warnings: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)


class PaperResult(BaseModel):
    title: str
    authors: list[str]
    summary: str
    published: str | None = None
    pdf_url: str | None = None
    entry_url: str | None = None
    doi: str | None = None


class DocumentRecord(BaseModel):
    id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int
    parser: str = "pymupdf"
    file_path: str
    created_at: datetime


class UploadResponse(BaseModel):
    document: DocumentRecord
    duplicated: bool = False
    warnings: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    document_count: int
    chunk_count: int
    model_configured: bool
    default_pdf_parser: str
    parser_availability: dict[str, bool]
