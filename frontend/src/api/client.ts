import axios from 'axios'
import { getToken } from '../auth'

// vite proxy 把 /api 轉 http://localhost:8950，容器模式下改為直連
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

// 自動帶 Bearer token
api.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 401 時導向登入頁
api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    const status = (error as { response?: { status?: number } })?.response?.status
    if (status === 401) {
      localStorage.removeItem('modelhub_access_token')
      localStorage.removeItem('modelhub_id_token')
      localStorage.removeItem('modelhub_userinfo')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  },
)

// --- Types ---

export type SubmissionStatus =
  | 'draft'
  | 'submitted'
  | 'approved'
  | 'rejected'
  | 'training'
  | 'trained'
  | 'accepted'
  | 'failed'

export interface Submission {
  id: number
  req_no: string
  req_name: string | null
  product: string
  company: string
  submitter: string | null
  purpose: string | null
  priority: string
  model_type: string | null
  class_list: string | null
  map50_threshold: number | null
  map50_target: number | null
  map50_95_target: number | null
  inference_latency_ms: number | null
  model_size_limit_mb: number | null
  arch: string | null
  input_spec: string | null
  deploy_env: string | null
  dataset_source: string | null
  dataset_count: string | null
  dataset_val_count: number | null
  dataset_test_count: number | null
  class_count: number | null
  label_format: string | null
  kaggle_dataset_url: string | null
  expected_delivery: string | null
  status: SubmissionStatus
  reviewer_note: string | null
  reviewed_by: string | null
  reviewed_at: string | null
  // Sprint 2
  rejection_reasons: string | null
  rejection_note: string | null
  resubmit_count: number | null
  resubmitted_at: string | null
  // Sprint 3
  kaggle_kernel_slug: string | null
  kaggle_kernel_version: number | null
  kaggle_status: string | null
  kaggle_status_updated_at: string | null
  kaggle_log_url: string | null
  training_started_at: string | null
  training_completed_at: string | null
  // Sprint 4
  gpu_seconds: number | null
  estimated_cost_usd: number | null
  total_attempts: number | null
  // Sprint 6
  max_retries: number | null
  retry_count: number | null
  max_budget_usd: number | null
  budget_exceeded_notified: boolean | null
  // Dataset unblock
  dataset_status: string | null
  blocked_reason: string | null
  created_at: string
}

export interface SubmissionHistoryItem {
  id: number
  req_no: string
  action: string
  actor: string | null
  reasons: string[] | null
  note: string | null
  meta: Record<string, unknown> | null
  created_at: string
}

export interface KaggleStatus {
  req_no: string
  kaggle_kernel_slug: string | null
  kaggle_kernel_version: number | null
  kaggle_status: string | null
  kaggle_status_updated_at: string | null
  kaggle_log_url: string | null
  training_started_at: string | null
  training_completed_at: string | null
  gpu_seconds: number | null
  estimated_cost_usd: number | null
}

export interface ModelVersion {
  id: number
  req_no: string
  product: string
  model_name: string
  version: string
  train_date: string | null
  map50: number | null
  map50_95: number | null
  file_path: string | null
  status: string
  notes: string | null
  kaggle_kernel_url: string | null
  epochs: number | null
  batch_size: number | null
  arch: string | null
  map50_actual: number | null
  map50_95_actual: number | null
  pass_fail: string | null
  accepted_by: string | null
  accepted_at: string | null
  acceptance_note: string | null
  is_current: boolean | null
  created_at: string
}

export interface StatsSummary {
  total: number
  by_status: Record<string, number>
}

// --- Submission API ---

export const submissionsApi = {
  list: (params?: { status?: string; product?: string; dataset_status?: string }) =>
    api.get<Submission[]>('/api/submissions/', { params }).then((r) => r.data),

  get: (req_no: string) =>
    api.get<Submission>(`/api/submissions/${req_no}`).then((r) => r.data),

  create: (data: Partial<Submission>) =>
    api
      .post<{ submission: Submission; warnings: string[]; suggestions: string[] }>(
        '/api/submissions/',
        data,
      )
      .then((r) => r.data),

  update: (req_no: string, data: Partial<Submission>) =>
    api.patch<Submission>(`/api/submissions/${req_no}`, data).then((r) => r.data),

  stats: () =>
    api.get<StatsSummary>('/api/submissions/stats/summary').then((r) => r.data),

  action: (
    req_no: string,
    action: string,
    payload?: { note?: string; actor?: string },
  ) =>
    api
      .post(`/api/submissions/${req_no}/actions/${action}`, payload ?? {})
      .then((r) => r.data),

  reject: (
    req_no: string,
    payload: { reasons: string[]; note?: string; actor?: string },
  ) =>
    api.post(`/api/submissions/${req_no}/reject`, payload).then((r) => r.data),

  resubmit: (req_no: string, payload: { note?: string; actor?: string }) =>
    api.post(`/api/submissions/${req_no}/resubmit`, payload).then((r) => r.data),

  history: (req_no: string) =>
    api
      .get<SubmissionHistoryItem[]>(`/api/submissions/${req_no}/history`)
      .then((r) => r.data),

  attachKernel: (
    req_no: string,
    payload: { slug: string; version?: number; actor?: string },
  ) =>
    api
      .post(`/api/submissions/${req_no}/attach-kernel`, payload)
      .then((r) => r.data),

  kaggleStatus: (req_no: string) =>
    api
      .get<KaggleStatus>(`/api/submissions/${req_no}/kaggle-status`)
      .then((r) => r.data),

  refreshKaggle: (req_no: string) =>
    api.post(`/api/submissions/${req_no}/refresh-kaggle`).then((r) => r.data),
}

// --- Admin API (Sprint 7.1) ---

export interface ApiKeyListItem {
  id: number
  key_preview: string
  name: string
  created_by: string | null
  created_at: string
  last_used_at: string | null
  disabled: boolean
}

export interface ApiKeyFull extends ApiKeyListItem {
  key: string
}

export const adminApi = {
  listApiKeys: () =>
    api.get<ApiKeyListItem[]>('/api/admin/api-keys/').then((r) => r.data),

  createApiKey: (name: string) =>
    api.post<ApiKeyFull>('/api/admin/api-keys/', { name }).then((r) => r.data),

  disableApiKey: (id: number) =>
    api.post(`/api/admin/api-keys/${id}/disable`).then((r) => r.data),

  enableApiKey: (id: number) =>
    api.post(`/api/admin/api-keys/${id}/enable`).then((r) => r.data),
}

// --- Version ---

export interface VersionInfo {
  version: string
  build: string
  commit: string
}

export const versionApi = {
  get: () => api.get<VersionInfo>('/version').then((r) => r.data),
}

// --- Registry API ---

export const registryApi = {
  list: (params?: { req_no?: string; product?: string; status?: string }) =>
    api.get<ModelVersion[]>('/api/registry/', { params }).then((r) => r.data),

  get: (id: number) =>
    api.get<ModelVersion>(`/api/registry/${id}`).then((r) => r.data),

  byReq: (req_no: string) =>
    api.get<ModelVersion[]>(`/api/registry/by-req/${req_no}`).then((r) => r.data),

  create: (data: Partial<ModelVersion>) =>
    api.post<ModelVersion>('/api/registry/', data).then((r) => r.data),

  accept: (
    id: number,
    data: {
      map50_actual: number
      map50_95_actual?: number
      acceptance_note?: string
      accepted_by?: string
    },
  ) => api.post<ModelVersion>(`/api/registry/${id}/accept`, data).then((r) => r.data),
}
