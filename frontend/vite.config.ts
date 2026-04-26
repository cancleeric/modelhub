import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 3950,
    proxy: {
      '/api': {
        target: process.env.MODELHUB_API_URL || 'http://modelhub-api-dev:8000',
        changeOrigin: true,
      },
      '/lids': {
        target: process.env.LIDS_URL || 'http://squid-lids-dev:8073',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/lids/, ''),
      },
    },
  },
})
