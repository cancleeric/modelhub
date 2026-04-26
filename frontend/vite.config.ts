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
        target: 'http://localhost:8950',
        changeOrigin: true,
      },
      '/lids': {
        target: 'http://localhost:8073',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/lids/, ''),
      },
    },
  },
})
