import { useState } from 'react';
import { query, getChunk } from '../services/api';

const QueryInterface = () => {
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [expandedChunks, setExpandedChunks] = useState(new Set());

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!queryText.trim()) return;

    setLoading(true);
    try {
      const response = await query(queryText);
      setResults(response);
    } catch (error) {
      // Extract error message from axios response
      console.error('Full error object:', error);
      console.error('Error response:', error.response);
      console.error('Error response data:', error.response?.data);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message || 
                          'Unknown error occurred';
      console.error('Extracted error message:', errorMessage);
      alert('Query failed: ' + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const toggleChunk = async (docId, chunkId) => {
    const key = `${docId}-${chunkId}`;
    if (expandedChunks.has(key)) {
      const newSet = new Set(expandedChunks);
      newSet.delete(key);
      setExpandedChunks(newSet);
    } else {
      try {
        const chunkData = await getChunk(docId, chunkId);
        // Store full chunk data (you might want to use state for this)
        const newSet = new Set(expandedChunks);
        newSet.add(key);
        setExpandedChunks(newSet);
      } catch (error) {
        alert('Failed to load chunk: ' + error.message);
      }
    }
  };

  return (
    <div className="query-container">
      <h2>Ask a Question</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="Enter your research question..."
          rows={4}
          style={{ width: '100%', padding: '10px' }}
        />
        <button type="submit" disabled={loading || !queryText.trim()}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {results && (
        <div className="results">
          <div className="answer-section">
            <h3>Answer</h3>
            <div className="answer-text">{results.answer}</div>
          </div>

          <div className="sources-section">
            <h3>Sources</h3>
            {results.sources.map((source, idx) => (
              <div key={idx} className="source-card">
                <h4>
                  {source.doc_title || source.doc_id || 'Untitled Document'} (p{source.page_range})
                  <span className="score">
                    Score: {source.similarity_score.toFixed(3)}
                  </span>
                </h4>
                <p className="snippet">{source.snippet}</p>
                <button
                  onClick={() => toggleChunk(source.doc_id, source.chunk_id)}
                >
                  {expandedChunks.has(`${source.doc_id}-${source.chunk_id}`)
                    ? 'Show Less'
                    : 'Show More'}
                </button>
              </div>
            ))}
          </div>

          {results.related_papers && results.related_papers.length > 0 && (
            <div className="related-papers-section">
              <h3>Related Papers</h3>
              {results.related_papers.map((paper, idx) => (
                <div key={idx} className="paper-card">
                  <h4>{paper.title || paper.doc_id || 'Untitled Document'}</h4>
                  <p>
                    {paper.authors || 'Unknown authors'} ({paper.year || 'N/A'}) - Relevance:{' '}
                    {paper.relevance_score.toFixed(3)}
                  </p>
                  {paper.doi && <p>DOI: {paper.doi}</p>}
                </div>
              ))}
            </div>
          )}

          {results.gaps && results.gaps.length > 0 && (
            <div className="gaps-section">
              <h3>Research Gaps</h3>
              {results.gaps.map((gap, idx) => (
                <div key={idx} className="gap-card">
                  <h4>Gap {idx + 1}</h4>
                  <p>{gap.description}</p>
                  {gap.suggestions && gap.suggestions.length > 0 && (
                    <ul>
                      {gap.suggestions.map((suggestion, sIdx) => (
                        <li key={sIdx}>{suggestion}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInterface;

