"""
Unit tests for RAG functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.rag import retrieve_chunks, format_context_for_llm, rerank_chunks, mmr_diversity_selection
from app.embedding_cache import EmbeddingCache


class TestRetrieveChunks:
    """Test chunk retrieval functionality"""
    
    @patch('app.rag.get_embedding_service')
    @patch('app.rag.get_faiss_index')
    def test_retrieve_chunks_empty_index(self, mock_faiss, mock_embedding):
        """Test retrieval with empty FAISS index"""
        mock_idx = Mock()
        mock_idx.get_total.return_value = 0
        mock_faiss.return_value = mock_idx
        
        db = Mock(spec=Session)
        
        with pytest.raises(ValueError, match="FAISS index is empty"):
            retrieve_chunks("test query", db, k=5)
    
    @patch('app.rag.get_embedding_service')
    @patch('app.rag.get_faiss_index')
    @patch('app.rag.get_embedding_cache')
    def test_retrieve_chunks_with_doc_id(self, mock_cache, mock_faiss, mock_embedding):
        """Test retrieval filtered by document ID"""
        # Mock embedding service
        mock_emb_service = Mock()
        mock_emb_service.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_embedding.return_value = mock_emb_service
        
        # Mock database
        db = Mock(spec=Session)
        chunk_meta = Mock()
        chunk_meta.chunk_id = 1
        chunk_meta.doc_id = "test-doc"
        chunk_meta.doc_title = "Test Document"
        chunk_meta.text = "Test text"
        chunk_meta.page_start = 1
        chunk_meta.page_end = 1
        chunk_meta.source_path = "/test/path"
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [chunk_meta]
        
        # Mock cache
        mock_cache_obj = Mock()
        mock_cache_obj.get_embeddings.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cache.return_value = mock_cache_obj
        
        chunks = retrieve_chunks("test query", db, k=5, doc_id="test-doc")
        
        assert len(chunks) > 0
        assert chunks[0]['doc_id'] == "test-doc"


class TestFormatContext:
    """Test context formatting for LLM"""
    
    def test_format_context_basic(self):
        """Test basic context formatting"""
        chunks = [
            {
                'doc_title': 'Test Doc',
                'page_start': 1,
                'page_end': 2,
                'text': 'This is test text'
            }
        ]
        
        context = format_context_for_llm(chunks)
        
        assert 'Test Doc' in context
        assert 'p1-2' in context
        assert 'This is test text' in context
    
    def test_format_context_multiple_chunks(self):
        """Test formatting with multiple chunks"""
        chunks = [
            {
                'doc_title': 'Doc 1',
                'page_start': 1,
                'page_end': 1,
                'text': 'Text 1'
            },
            {
                'doc_title': 'Doc 2',
                'page_start': 2,
                'page_end': 2,
                'text': 'Text 2'
            }
        ]
        
        context = format_context_for_llm(chunks)
        
        assert 'Doc 1' in context
        assert 'Doc 2' in context
        assert 'Text 1' in context
        assert 'Text 2' in context


class TestRerankChunks:
    """Test chunk reranking"""
    
    @patch('app.rag.get_embedding_cache')
    def test_rerank_chunks(self, mock_cache):
        """Test reranking functionality"""
        chunks = [
            {'chunk_id': 1, 'similarity_score': 0.5},
            {'chunk_id': 2, 'similarity_score': 0.8},
            {'chunk_id': 3, 'similarity_score': 0.3}
        ]
        
        mock_cache_obj = Mock()
        mock_cache_obj.get_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_cache.return_value = mock_cache_obj
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        
        reranked = rerank_chunks(query_embedding, chunks, mock_cache_obj, top_k=2)
        
        assert len(reranked) == 2
        # Should be sorted by relevance
        assert reranked[0]['relevance_score'] >= reranked[1]['relevance_score']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

