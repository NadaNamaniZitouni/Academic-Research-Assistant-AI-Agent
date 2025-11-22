import { useState, useEffect } from 'react';
import { getQueryAnalytics, getUserStats } from '../services/api';
import './AnalyticsDashboard.css';

const AnalyticsDashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('overview');
  const [lastRefresh, setLastRefresh] = useState(new Date());

  useEffect(() => {
    fetchData();
  }, [days]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchData();
      setLastRefresh(new Date());
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [days]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [analyticsData, statsData] = await Promise.all([
        getQueryAnalytics(days),
        getUserStats()
      ]);
      setAnalytics(analyticsData);
      setStats(statsData);
    } catch (error) {
      console.error('Error fetching analytics:', error);
      // Set empty data structure on error so UI can still render
      setAnalytics({
        total_queries: 0,
        average_response_time_ms: 0,
        most_asked_questions: [],
        queries_per_day: [],
        response_time_stats: { min: 0, max: 0, median: 0 },
        documents_queried: [],
        period_days: days
      });
      setStats({
        documents_count: 0,
        total_chunks: 0,
        total_queries: 0,
        queries_this_month: 0,
        tier: 'free',
        account_created: null
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="analytics-container">
        <div className="loading">Loading analytics...</div>
      </div>
    );
  }

  if (!analytics || !stats) {
    return (
      <div className="analytics-container">
        <div className="error">
          <h3>No analytics data available</h3>
          <p>Start uploading documents and making queries to see your analytics here!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="analytics-container">
      <div className="analytics-header">
        <div>
          <h2>📊 Analytics Dashboard</h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Available to all users (Free plan included) • Track your research activity and query patterns
            {lastRefresh && (
              <span style={{ marginLeft: '1rem', fontSize: '0.85rem' }}>
                • Last updated: {lastRefresh.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button
            onClick={() => {
              fetchData();
              setLastRefresh(new Date());
            }}
            style={{
              padding: '0.5rem 1rem',
              background: 'var(--card-bg)',
              border: '2px solid #15a6b1',
              borderRadius: '6px',
              color: '#15a6b1',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#15a6b1';
              e.target.style.color = '#ffffff';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'var(--card-bg)';
              e.target.style.color = '#15a6b1';
            }}
          >
            🔄 Refresh
          </button>
          <div className="period-selector">
            <label>Period: </label>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </div>
        </div>
      </div>

      <div className="tabs">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={activeTab === 'queries' ? 'active' : ''}
          onClick={() => setActiveTab('queries')}
        >
          Query Analytics
        </button>
        <button
          className={activeTab === 'performance' ? 'active' : ''}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </button>
      </div>

      {activeTab === 'overview' && (
        <div className="overview-tab">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Total Documents</div>
              <div className="stat-value">{stats.documents_count}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Total Chunks</div>
              <div className="stat-value">{stats.total_chunks}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Total Queries (All Time)</div>
              <div className="stat-value">{stats.total_queries}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Queries This Month</div>
              <div className="stat-value">{stats.queries_this_month}</div>
              <div className="stat-sublabel">Tier: {stats.tier.charAt(0).toUpperCase() + stats.tier.slice(1)}</div>
            </div>
          </div>

          <div className="section">
            <h3>Recent Activity</h3>
            <div className="activity-summary">
              {analytics.total_queries > 0 ? (
                <>
                  <p>Queries in last {days} days: <strong>{analytics.total_queries}</strong></p>
                  <p>Average response time: <strong>{analytics.average_response_time_ms || 0}ms</strong></p>
                </>
              ) : (
                <p className="no-data">No queries in the last {days} days. Start querying your documents to see activity here!</p>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'queries' && (
        <div className="queries-tab">
          <div className="section">
            <h3>Most Asked Questions</h3>
            {analytics.most_asked_questions.length > 0 ? (
              <div className="questions-list">
                {analytics.most_asked_questions.map((item, idx) => (
                  <div key={idx} className="question-item">
                    <div className="question-count">{item.count}x</div>
                    <div className="question-text">{item.query}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-data">No queries yet</p>
            )}
          </div>

          <div className="section">
            <h3>Queries Per Day</h3>
            {analytics.queries_per_day.length > 0 ? (
              <div className="queries-chart">
                {analytics.queries_per_day.map((item, idx) => (
                  <div key={idx} className="chart-bar-container">
                    <div className="chart-bar" style={{ height: `${(item.count / Math.max(...analytics.queries_per_day.map(q => q.count))) * 100}%` }}>
                      <span className="chart-value">{item.count}</span>
                    </div>
                    <div className="chart-label">{new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-data">No query data available</p>
            )}
          </div>

          <div className="section">
            <h3>Most Queried Documents</h3>
            {analytics.documents_queried.length > 0 ? (
              <div className="documents-list">
                {analytics.documents_queried.map((item, idx) => (
                  <div key={idx} className="document-item">
                    <div className="document-id">{item.doc_id.substring(0, 8)}...</div>
                    <div className="document-count">{item.count} queries</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-data">No document queries yet</p>
            )}
          </div>
        </div>
      )}

      {activeTab === 'performance' && (
        <div className="performance-tab">
          <div className="section">
            <h3>Response Time Statistics</h3>
            {analytics.total_queries > 0 && analytics.response_time_stats ? (
              <div className="performance-stats">
                <div className="perf-stat">
                  <div className="perf-label">Average</div>
                  <div className="perf-value">{analytics.average_response_time_ms || 0}ms</div>
                </div>
                <div className="perf-stat">
                  <div className="perf-label">Minimum</div>
                  <div className="perf-value">{analytics.response_time_stats.min || 0}ms</div>
                </div>
                <div className="perf-stat">
                  <div className="perf-label">Maximum</div>
                  <div className="perf-value">{analytics.response_time_stats.max || 0}ms</div>
                </div>
                <div className="perf-stat">
                  <div className="perf-label">Median</div>
                  <div className="perf-value">{analytics.response_time_stats.median || 0}ms</div>
                </div>
              </div>
            ) : (
              <p className="no-data">No performance data available yet. Make some queries to see response time statistics!</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyticsDashboard;

