from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    tier = Column(String, default="free")  # free, starter, pro, team
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class UserUsage(Base):
    __tablename__ = "user_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    month = Column(String, nullable=False)  # Format: YYYY-MM
    queries_count = Column(Integer, default=0)
    documents_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    query_text = Column(Text, nullable=False)
    doc_id = Column(String, index=True)
    answer = Column(Text)
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)  # Link documents to users
    title = Column(String)
    authors = Column(Text)
    year = Column(Integer)
    doi = Column(String)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChunkMetadata(Base):
    __tablename__ = "chunk_metadata"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, index=True)
    doc_title = Column(String)
    source_path = Column(String)
    page_start = Column(Integer)
    page_end = Column(Integer)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

