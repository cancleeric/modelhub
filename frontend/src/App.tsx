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
import AcceptanceQueuePage from './pages/AcceptanceQueuePage'
import StatsPage from './pages/StatsPage'
import ApiKeyPage from './pages/ApiKeyPage'
import PredictPage from './pages/PredictPage'

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
                    <Route path="/submit/:req_no" element={<SubmissionFormPage />} />
                    <Route path="/submissions/:req_no" element={<SubmissionDetailPage />} />
                    <Route path="/review" element={<ReviewPage />} />
                    <Route path="/registry" element={<RegistryPage />} />
                    <Route path="/registry/:id/accept" element={<AcceptancePage />} />
                    <Route path="/acceptance" element={<AcceptanceQueuePage />} />
                    <Route path="/stats" element={<StatsPage />} />
                    <Route path="/admin/api-keys" element={<ApiKeyPage />} />
                    <Route path="/predict" element={<PredictPage />} />
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
