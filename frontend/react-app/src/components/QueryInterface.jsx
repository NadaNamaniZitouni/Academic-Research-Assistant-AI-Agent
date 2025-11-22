import { useState } from 'react';
import { query, getChunk, exportMarkdown, exportText } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const ExportButton = ({ format, results, question, onError }) => {
  const { user } = useAuth();
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    if (!user) {
      alert('Please log in to export');
      return;
    }

    // Check if user has export access (starter, pro, team tiers)
    const canExport = ['starter', 'pro', 'team'].includes(user.tier);
    if (!canExport) {
      alert('Export feature is only available in Starter, Pro, or Team plans. Please upgrade your account.');
      return;
    }

    setExporting(true);
    try {
      let blob;
      let filename;
      let mimeType;

      if (format === 'markdown') {
        blob = await exportMarkdown(results, question);
        filename = `query_result_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
        mimeType = 'text/markdown';
      } else if (format === 'text') {
        blob = await exportText(results, question);
        filename = `query_result_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
        mimeType = 'text/plain';
      }

      // Create download link
      const url = window.URL.createObjectURL(new Blob([blob], { type: mimeType }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Export failed';
      if (onError) {
        onError(errorMsg);
      } else {
        alert('Export failed: ' + errorMsg);
      }
    } finally {
      setExporting(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      style={{
        padding: '0.5rem 1rem',
        background: 'var(--accent-color, #6366f1)',
        border: 'none',
        borderRadius: '6px',
        color: 'white',
        cursor: 'pointer',
        fontSize: '0.85rem',
        fontWeight: 500
      }}
      title={`Export as ${format.toUpperCase()}`}
    >
      {exporting ? 'Exporting...' : `Export ${format.toUpperCase()}`}
    </button>
  );
};

const QueryInterface = ({ docId = null }) => {
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [expandedChunks, setExpandedChunks] = useState(new Set());
  const [searchAllDocs, setSearchAllDocs] = useState(false); // Toggle to search all docs

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!queryText.trim()) return;

    setLoading(true);
    try {
      // Use docId if available and not searching all documents
      const queryDocId = (searchAllDocs || !docId) ? null : docId;
      console.log('[QueryInterface] Querying with docId:', queryDocId, 'searchAllDocs:', searchAllDocs);
      const response = await query(queryText, undefined, queryDocId);
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3>Answer</h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <ExportButton
                  format="markdown"
                  results={results}
                  question={queryText}
                  onError={(error) => alert('Export failed: ' + error)}
                />
                <ExportButton
                  format="text"
                  results={results}
                  question={queryText}
                  onError={(error) => alert('Export failed: ' + error)}
                />
              </div>
            </div>
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

