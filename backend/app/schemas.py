from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    query: str
    k: int = 8  # Default to 8 chunks for better context coverage


class SourceResponse(BaseModel):
    chunk_id: int
    doc_id: str
    doc_title: str
    page_range: str
    snippet: str
    similarity_score: float


class RelatedPaperResponse(BaseModel):
    doc_id: str
    title: str
    authors: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    relevance_score: float
    matching_chunks: int


class GapResponse(BaseModel):
    description: str
    suggestions: List[str]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    related_papers: List[RelatedPaperResponse]
    gaps: List[GapResponse]


class UploadResponse(BaseModel):
    doc_id: str
    status: str
    num_chunks: int

