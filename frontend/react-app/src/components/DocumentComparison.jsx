import { useState, useEffect } from 'react';
import { compareDocuments, getUserDocuments } from '../services/api';
import './DocumentComparison.css';

const DocumentComparison = () => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      // This would need to be implemented in the API
      // For now, we'll use a placeholder
      const docs = await getUserDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleCompare = async () => {
    if (selectedDocs.length < 2) {
      alert('Please select at least 2 documents to compare');
      return;
    }

    setLoading(true);
    try {
      const result = await compareDocuments(selectedDocs);
      setComparison(result);
    } catch (error) {
      alert('Error comparing documents: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const toggleDocument = (docId) => {
    if (selectedDocs.includes(docId)) {
      setSelectedDocs(selectedDocs.filter(id => id !== docId));
    } else {
      setSelectedDocs([...selectedDocs, docId]);
    }
  };

  return (
    <div className="comparison-container">
      <h2>📊 Document Comparison</h2>
      
      <div className="document-selector">
        <h3>Select Documents to Compare</h3>
        {documents.length === 0 ? (
          <p className="no-docs">No documents available. Upload documents first.</p>
        ) : (
          <div className="documents-list">
            {documents.map(doc => (
              <label key={doc.doc_id} className="document-checkbox">
                <input
                  type="checkbox"
                  checked={selectedDocs.includes(doc.doc_id)}
                  onChange={() => toggleDocument(doc.doc_id)}
                />
                <div className="document-info">
                  <div className="doc-title">{doc.title || doc.doc_id}</div>
                  <div className="doc-meta">
                    {doc.authors && <span>{doc.authors}</span>}
                    {doc.year && <span>{doc.year}</span>}
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}
        
        <button
          onClick={handleCompare}
          disabled={selectedDocs.length < 2 || loading}
          className="compare-button"
        >
          {loading ? 'Comparing...' : `Compare ${selectedDocs.length} Documents`}
        </button>
      </div>

      {comparison && (
        <div className="comparison-results">
          <h3>Comparison Results</h3>
          
          <div className="similarities-section">
            <h4>Document Similarities</h4>
            <div className="similarities-grid">
              {Object.entries(comparison.similarities).map(([key, value]) => {
                const [doc1, doc2] = key.split('_');
                return (
                  <div key={key} className="similarity-item">
                    <div className="similarity-pair">
                      {comparison.documents.find(d => d.doc_id === doc1)?.title || doc1.substring(0, 8)} ↔
                      {comparison.documents.find(d => d.doc_id === doc2)?.title || doc2.substring(0, 8)}
                    </div>
                    <div className="similarity-score">{(value * 100).toFixed(1)}%</div>
                  </div>
                );
              })}
            </div>
          </div>

          {comparison.common_themes && comparison.common_themes.length > 0 && (
            <div className="themes-section">
              <h4>Common Themes</h4>
              {comparison.common_themes.map((theme, idx) => (
                <div key={idx} className="theme-item">
                  <div className="theme-source">
                    {theme.doc_title} (p{theme.page})
                  </div>
                  <div className="theme-text">{theme.text}...</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentComparison;

