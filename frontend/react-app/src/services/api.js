import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8010';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15 * 60 * 1000, // 15 minutes timeout for large file uploads and processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for debugging and FormData handling
api.interceptors.request.use(
  (config) => {
    console.log('[API Interceptor] Request intercepted:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      fullURL: `${config.baseURL}${config.url}`,
      isFormData: config.data instanceof FormData,
      contentType: config.headers?.['Content-Type'] || config.headers?.common?.['Content-Type'] || 'not set'
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
  const uploadAxios = axios.create({
    baseURL: API_BASE_URL,
    timeout: 15 * 60 * 1000, // 15 minutes for upload + processing
    headers: {
      'Content-Type': undefined, // Forces browser to generate boundary
    },
  });
  
  try {
    const response = await uploadAxios.post('/upload', formData, {
      headers: {
        'Content-Type': undefined, // Explicitly undefined in request too
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

export const query = async (queryText, k = 5) => {
  const response = await api.post('/query', {
    query: queryText,
    k: k,
  });
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

export default api;

