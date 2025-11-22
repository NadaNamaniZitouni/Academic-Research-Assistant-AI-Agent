import { useState } from 'react';
import { uploadPDF, checkIngestStatus } from '../services/api';

const UploadPDF = ({ onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [docId, setDocId] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file');
      return;
    }

    // Validate file size (100MB max)
    const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
    if (file.size > MAX_FILE_SIZE) {
      alert(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      return;
    }

    if (file.size === 0) {
      alert('File is empty');
      return;
    }

    setUploading(true);
    setStatus('Uploading file...');

    try {
      console.log(`[UploadPDF] Starting upload for file: ${file.name}, size: ${(file.size / 1024).toFixed(2)} KB`);
      console.log(`[UploadPDF] File type: ${file.type}, lastModified: ${new Date(file.lastModified).toISOString()}`);
      
      // Upload with progress tracking and timeout (10 minutes for large files)
      const uploadPromise = uploadPDF(file, (progress) => {
        console.log(`[UploadPDF] Progress callback: ${progress}%`);
        setUploadProgress(progress);
        if (progress < 100) {
          setStatus(`Uploading file... ${progress}%`);
        } else {
          setStatus('Processing PDF...');
        }
      });
      
      console.log(`[UploadPDF] Upload promise created, waiting for response...`);
      
      // Axios already has a timeout, so we don't need Promise.race
      // This was causing issues where the timeout would trigger even if the request was working
      const result = await uploadPromise;
      console.log(`[UploadPDF] Upload completed, result:`, result);
      
      console.log('Upload successful:', result);
      setDocId(result.doc_id);
      
      // Since we process synchronously now, check if it's already completed
      setUploadProgress(100);
      if (result.status === 'completed') {
        setUploading(false);
        setStatus('Completed');
        setUploadProgress(0);
        alert(`PDF processed successfully! Created ${result.num_chunks} chunks.`);
        // Notify parent component that upload is complete with doc_id
        if (onUploadComplete) {
          onUploadComplete(result.doc_id);
        }
      } else {
        // Fallback: poll for status (for backwards compatibility)
        setStatus('Processing...');
        const pollStatus = async () => {
          try {
            const statusResult = await checkIngestStatus(result.doc_id);
            setStatus(statusResult.status);

            if (statusResult.status === 'completed') {
              setUploading(false);
              alert('PDF processed successfully!');
              // Notify parent component that upload is complete with doc_id
              if (onUploadComplete) {
                onUploadComplete(result.doc_id);
              }
            } else if (statusResult.status === 'failed') {
              setUploading(false);
              alert('Processing failed: ' + (statusResult.error || 'Unknown error'));
            } else {
              setTimeout(pollStatus, 2000);
            }
          } catch (pollError) {
            console.error('Error polling status:', pollError);
            setUploading(false);
            setStatus('Error checking status');
          }
        };
        setTimeout(pollStatus, 1000);
      }
    } catch (error) {
      setUploading(false);
      setUploadProgress(0);
      console.error('Upload error:', error);
      
      // Extract detailed error message
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message || 
                          'Unknown error occurred';
      
      setStatus(`Error: ${errorMessage}`);
      
      // Show user-friendly error message with more details
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        alert(`Upload timed out after 15 minutes. This might mean:\n- The file is very large and processing is slow\n- The backend server is not responding\n- There's a network connectivity issue\n\nCheck the browser console and backend logs for more details.`);
      } else if (error.message.includes('Network Error') || error.message.includes('ERR_CONNECTION') || error.code === 'ERR_NETWORK') {
        alert(`Network error: Cannot connect to backend server at ${error.config?.baseURL || 'http://localhost:8010'}\n\nPlease check:\n- Is the backend server running?\n- Is the URL correct?\n- Are there any firewall issues?`);
      } else if (error.response) {
        // Server responded with an error
        alert(`Server error (${error.response.status}): ${errorMessage}\n\nCheck backend logs for details.`);
      } else {
        alert(`Upload failed: ${errorMessage}\n\nError code: ${error.code || 'Unknown'}\n\nCheck the browser console for more details.`);
      }
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload PDF</h2>
      <div className="upload-form">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={uploading}
        />
        <button onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? 'Uploading...' : 'Upload PDF'}
        </button>
      </div>
      {status && (
        <div className="status">
          <p>Status: {status}</p>
          {uploading && uploadProgress > 0 && uploadProgress < 100 && (
            <div style={{ marginTop: '10px' }}>
              <div style={{ 
                width: '100%', 
                backgroundColor: '#e0e0e0', 
                borderRadius: '4px',
                overflow: 'hidden',
                height: '20px'
              }}>
                <div style={{
                  width: `${uploadProgress}%`,
                  height: '100%',
                  backgroundColor: '#007bff',
                  transition: 'width 0.3s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {uploadProgress}%
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UploadPDF;

