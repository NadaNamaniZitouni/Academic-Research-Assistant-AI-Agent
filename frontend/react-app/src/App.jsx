import { useState } from 'react';
import UploadPDF from './components/UploadPDF';
import QueryInterface from './components/QueryInterface';
import './App.css';

function App() {
  const [hasUploaded, setHasUploaded] = useState(false);

  const handleUploadComplete = () => {
    setHasUploaded(true);
  };

  return (
    <div className="App">
      <header>
        <h1>Academic Research Assistant</h1>
        <p className="subtitle">Upload PDFs and query your research documents</p>
      </header>

      <main>
        <div className="main-content">
          {/* Upload Section */}
          <section className="upload-section">
            <UploadPDF onUploadComplete={handleUploadComplete} />
            {!hasUploaded && (
              <div className="info-box">
                <h3>💡 Getting Started</h3>
                <p>Upload a PDF document to get started. Once uploaded, you'll be able to query it below.</p>
              </div>
            )}
          </section>

          {/* Query Section - Show after upload or always but with message */}
          <section className="query-section">
            <QueryInterface />
            {hasUploaded && (
              <div className="info-box">
                <h3>📚 How it works</h3>
                <p>The system will find relevant sections, provide citations, suggest related papers, and identify research gaps based on your uploaded documents.</p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;

