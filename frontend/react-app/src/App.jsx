import { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import UploadPDF from './components/UploadPDF';
import QueryInterface from './components/QueryInterface';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import DocumentComparison from './components/DocumentComparison';
import Login from './components/Login';
import Register from './components/Register';
import './App.css';

const AppContent = () => {
  const { user, loading, logout } = useAuth();
  // ALL hooks must be called at the top, before any conditional returns
  const [hasUploaded, setHasUploaded] = useState(false);
  const [currentDocId, setCurrentDocId] = useState(null);
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
  const [activeSection, setActiveSection] = useState('main'); // 'main', 'analytics', 'compare', 'network'

  const handleUploadComplete = (docId) => {
    setHasUploaded(true);
    setCurrentDocId(docId);
  };

  if (loading) {
    return (
      <div className="App">
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-secondary)' }}>
          Loading...
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="App">
        <header>
          <h1>Academic Research Assistant</h1>
          <p className="subtitle">Upload PDFs and query your research documents</p>
        </header>
        <main>
          {authMode === 'login' ? (
            <Login onSwitchToRegister={() => setAuthMode('register')} />
          ) : (
            <Register onSwitchToLogin={() => setAuthMode('login')} />
          )}
        </main>
      </div>
    );
  }

  return (
    <div className="App">
      <header>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            <h1>Academic Research Assistant</h1>
            <p className="subtitle">Upload PDFs and query your research documents</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <nav style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                onClick={() => setActiveSection('main')}
                style={{
                  padding: '0.5rem 1rem',
                  background: activeSection === 'main' ? '#15a6b1' : 'transparent',
                  border: '2px solid #15a6b1',
                  borderRadius: '6px',
                  color: activeSection === 'main' ? '#ffffff' : '#15a6b1',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  transition: 'all 0.3s ease'
                }}
              >
                Main
              </button>
              <button
                onClick={() => setActiveSection('analytics')}
                style={{
                  padding: '0.5rem 1rem',
                  background: activeSection === 'analytics' ? '#15a6b1' : 'transparent',
                  border: '2px solid #15a6b1',
                  borderRadius: '6px',
                  color: activeSection === 'analytics' ? '#ffffff' : '#15a6b1',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  transition: 'all 0.3s ease'
                }}
              >
                Analytics
              </button>
              <button
                onClick={() => setActiveSection('compare')}
                style={{
                  padding: '0.5rem 1rem',
                  background: activeSection === 'compare' ? '#15a6b1' : 'transparent',
                  border: '2px solid #15a6b1',
                  borderRadius: '6px',
                  color: activeSection === 'compare' ? '#ffffff' : '#15a6b1',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  transition: 'all 0.3s ease'
                }}
              >
                Compare
              </button>
            </nav>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{user.username}</div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                {user.tier.charAt(0).toUpperCase() + user.tier.slice(1)} Plan
              </div>
            </div>
            <button
              onClick={logout}
              style={{
                padding: '0.5rem 1rem',
                background: 'var(--card-bg)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                cursor: 'pointer',
                fontSize: '0.9rem'
              }}
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <main>
        {activeSection === 'main' ? (
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
              <QueryInterface docId={currentDocId} />
              {hasUploaded && (
                <div className="info-box">
                  <h3>📚 How it works</h3>
                  <p>The system will find relevant sections, provide citations, suggest related papers, and identify research gaps based on your uploaded documents.</p>
                </div>
              )}
            </section>
          </div>
        ) : activeSection === 'analytics' ? (
          <AnalyticsDashboard />
        ) : activeSection === 'compare' ? (
          <DocumentComparison />
        ) : null}
      </main>
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;

