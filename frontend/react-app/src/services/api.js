import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8010';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15 * 60 * 1000, // 15 minutes timeout for large file uploads and processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests if available (FIRST interceptor - runs first)
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add request interceptor for debugging and FormData handling (runs after auth interceptor)
api.interceptors.request.use(
  (config) => {
    console.log('[API Interceptor] Request intercepted:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      fullURL: `${config.baseURL}${config.url}`,
      isFormData: config.data instanceof FormData,
      contentType: config.headers?.['Content-Type'] || config.headers?.common?.['Content-Type'] || 'not set',
      hasAuth: !!config.headers?.Authorization
    });
    
    // CRITICAL: If FormData, remove Content-Type so browser sets it with boundary
    if (config.data instanceof FormData) {
      console.log('[API Interceptor] FormData detected, removing Content-Type header');
      // Delete from headers object
      if (config.headers) {
        delete config.headers['Content-Type'];
        // Also remove from common headers if it exists
        if (config.headers.common && config.headers.common['Content-Type']) {
          delete config.headers.common['Content-Type'];
        }
      }
      console.log('[API Interceptor] Content-Type removed. Browser will set it with boundary.');
    }
    return config;
  },
  (error) => {
    console.error('[API Interceptor] Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log('[API] Response:', {
      status: response.status,
      statusText: response.statusText,
      url: response.config.url,
      data: response.data,
    });
    return response;
  },
  (error) => {
    console.error('[API] Response error:', {
      message: error.message,
      code: error.code,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      url: error.config?.url,
    });
    return Promise.reject(error);
  }
);

export const uploadPDF = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // CRITICAL FIX: Remove global axios headers that interfere with FormData
  delete axios.defaults.headers.post['Content-Type'];
  delete axios.defaults.headers.common['Content-Type'];
  
  // Create a separate axios instance with Content-Type explicitly set to undefined
  // This forces the browser to set multipart/form-data with boundary
  const token = localStorage.getItem('token');
  const uploadAxios = axios.create({
    baseURL: API_BASE_URL,
    timeout: 15 * 60 * 1000, // 15 minutes for upload + processing
    headers: {
      'Content-Type': undefined, // Forces browser to generate boundary
      ...(token && { Authorization: `Bearer ${token}` }), // Include auth token
    },
  });
  
  try {
    const response = await uploadAxios.post('/upload', formData, {
      headers: {
        'Content-Type': undefined, // Explicitly undefined in request too
        ...(token && { Authorization: `Bearer ${token}` }), // Include auth token
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('[uploadPDF] Upload error:', {
      message: error.message,
      code: error.code,
      status: error.response?.status,
      data: error.response?.data,
    });
    throw error;
  }
};

export const checkIngestStatus = async (docId) => {
  const response = await api.get(`/ingest/status/${docId}`);
  return response.data;
};

export const query = async (queryText, k = 12, docId = null) => {
  const requestBody = {
    query: queryText,
    k: k,
  };
  
  // Only include doc_id if provided (null = search all documents)
  if (docId) {
    requestBody.doc_id = docId;
  }
  
  const response = await api.post('/query', requestBody);
  return response.data;
};

export const getChunk = async (docId, chunkId) => {
  const response = await api.get(`/doc/${docId}/chunks/${chunkId}`);
  return response.data;
};

export const searchMetadata = async (query) => {
  const response = await api.get('/search-metadata', {
    params: { q: query },
  });
  return response.data;
};

// Authentication functions
export const login = async (username, password) => {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  
  const response = await api.post('/auth/login-json', {
    username,
    password
  });
  return response.data;
};

export const register = async (email, username, password, fullName = null) => {
  const response = await api.post('/auth/register', {
    email,
    username,
    password,
    full_name: fullName
  });
  return response.data;
};

export const getCurrentUser = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

export const getUsageStats = async () => {
  const response = await api.get('/auth/usage');
  return response.data;
};

// Analytics functions
export const getQueryAnalytics = async (days = 30) => {
  const response = await api.get('/analytics/queries', {
    params: { days }
  });
  return response.data;
};

export const getUserStats = async () => {
  const response = await api.get('/analytics/stats');
  return response.data;
};

// Export functions
export const exportBibTeX = async (docIds) => {
  const response = await api.post('/export/bibtex', { doc_ids: docIds }, {
    responseType: 'blob'
  });
  return response.data;
};

export const exportMarkdown = async (queryResult, question) => {
  const response = await api.post('/export/markdown', {
    query_result: queryResult,
    question: question
  }, {
    responseType: 'blob'
  });
  return response.data;
};

export const exportText = async (queryResult, question) => {
  const response = await api.post('/export/text', {
    query_result: queryResult,
    question: question
  }, {
    responseType: 'blob'
  });
  return response.data;
};

// Document comparison functions
export const compareDocuments = async (docIds) => {
  const response = await api.post('/compare/documents', {
    doc_ids: docIds
  });
  return response.data;
};

export const getCitationNetwork = async (docIds = [], similarityThreshold = 0.7) => {
  const response = await api.post('/citation-network', {
    doc_ids: docIds,
    similarity_threshold: similarityThreshold
  });
  return response.data;
};

export const getUserDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

export default api;

