import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'react-hot-toast'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    <Toaster
      position="top-right"
      toastOptions={{
        duration: 3000,
        style: {
          fontSize: '0.875rem',
          borderRadius: '0.5rem',
          boxShadow: '0 1px 3px rgba(0,0,0,.12)',
        },
      }}
    />
  </StrictMode>,
)
