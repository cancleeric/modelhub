import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import LoginPage from './pages/LoginPage'
import CallbackPage from './pages/CallbackPage'
import SubmissionListPage from './pages/SubmissionListPage'
import SubmissionFormPage from './pages/SubmissionFormPage'
import SubmissionDetailPage from './pages/SubmissionDetailPage'
import ReviewPage from './pages/ReviewPage'
import RegistryPage from './pages/RegistryPage'
import AcceptancePage from './pages/AcceptancePage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* 公開路由 */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/callback" element={<CallbackPage />} />

          {/* 需登入的路由 */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<SubmissionListPage />} />
                    <Route path="/submit" element={<SubmissionFormPage />} />
                    <Route path="/submissions/:req_no" element={<SubmissionDetailPage />} />
                    <Route path="/review" element={<ReviewPage />} />
                    <Route path="/registry" element={<RegistryPage />} />
                    <Route path="/registry/:id/accept" element={<AcceptancePage />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
