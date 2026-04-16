import axios from 'axios'

// vite proxy 把 /api 轉 http://localhost:8950，容器模式下改為直連
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

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
  created_at: string
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
  created_at: string
}

export interface StatsSummary {
  total: number
  by_status: Record<string, number>
}

// --- Submission API ---

export const submissionsApi = {
  list: (params?: { status?: string; product?: string }) =>
    api.get<Submission[]>('/api/submissions/', { params }).then((r) => r.data),

  get: (req_no: string) =>
    api.get<Submission>(`/api/submissions/${req_no}`).then((r) => r.data),

  create: (data: Partial<Submission>) =>
    api.post<Submission>('/api/submissions/', data).then((r) => r.data),

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
