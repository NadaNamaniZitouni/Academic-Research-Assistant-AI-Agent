from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime


class QueryRequest(BaseModel):
    query: str
    k: int = 12  # Default to 12 chunks after hybrid retrieval (reranking + diversity)
    doc_id: Optional[str] = None  # Optional: filter to specific document (None = search all documents)


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    email: str
    username: str
    full_name: Optional[str]
    tier: str
    is_premium: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UsageStats(BaseModel):
    user_id: str
    tier: str
    documents_count: int
    queries_this_month: int
    queries_limit: int
    documents_limit: int
    can_export: bool
    can_use_api: bool


class ExportBibTeXRequest(BaseModel):
    doc_ids: List[str]


class ExportQueryRequest(BaseModel):
    query_result: Dict
    question: str


class DocumentComparisonRequest(BaseModel):
    doc_ids: List[str]


class CitationNetworkRequest(BaseModel):
    doc_ids: List[str] = []
    similarity_threshold: float = 0.7
