import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0', // Listen on all addresses (needed for Docker)
    strictPort: true, // Exit if port is already in use
    hmr: {
      clientPort: 5173, // Explicitly set HMR port
      protocol: 'ws',
      host: 'localhost', // HMR host
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8010',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})

