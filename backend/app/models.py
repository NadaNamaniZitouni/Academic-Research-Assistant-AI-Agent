from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
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

