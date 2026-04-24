import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { submissionsApi, type Submission } from '../api/client'
import StatusBadge from '../components/StatusBadge'
import RejectModal from '../components/RejectModal'

export default function ReviewPage() {
  const qc = useQueryClient()
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [rejectTarget, setRejectTarget] = useState<Submission | null>(null)

  const { data: rawData, isLoading } = useQuery({
    queryKey: ['submissions', 'submitted'],
    queryFn: () => submissionsApi.list({ status: 'submitted', limit: 1000, offset: 0 }),
  })
  const data = rawData?.items

  const actionMut = useMutation({
    mutationFn: ({
      req_no,
      action,
      note,
    }: {
      req_no: string
      action: string
      note?: string
    }) => submissionsApi.action(req_no, action, { note }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['submissions'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setErrors((prev) => ({ ...prev, [vars.req_no]: '' }))
      setRejectTarget(null)
    },
    onError: (err: unknown, vars) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '操作失敗'
      setErrors((prev) => ({ ...prev, [vars.req_no]: String(msg) }))
    },
  })

  const rejectMut = useMutation({
    mutationFn: ({
      req_no,
      reasons,
      note,
    }: {
      req_no: string
      reasons: string[]
      note: string
    }) => submissionsApi.reject(req_no, { reasons, note }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['submissions'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setErrors((prev) => ({ ...prev, [vars.req_no]: '' }))
      setRejectTarget(null)
    },
    onError: (err: unknown, vars) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '退件失敗'
      setErrors((prev) => ({ ...prev, [vars.req_no]: String(msg) }))
    },
  })

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">CTO 審核佇列</h1>

      {isLoading && <p className="text-gray-500">載入中...</p>}

      {data && data.length === 0 && (
        <div className="bg-white rounded shadow p-8 text-center text-gray-400">
          目前沒有待審核的需求單
        </div>
      )}

      {data && data.length > 0 && (
        <div className="space-y-4">
          {data.map((s) => (
            <div key={s.req_no} className="bg-white rounded shadow p-5">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Link
                      to={`/submissions/${s.req_no}`}
                      className="font-mono text-indigo-600 hover:underline"
                    >
                      {s.req_no}
                    </Link>
                    <StatusBadge status={s.status} />
                    <span className="text-xs font-mono text-gray-500">{s.priority}</span>
                    {!!s.resubmit_count && s.resubmit_count > 0 && (
                      <span className="text-xs font-semibold text-orange-700 bg-orange-100 px-2 py-0.5 rounded">
                        第 {s.resubmit_count + 1} 次送審
                      </span>
                    )}
                  </div>
                  <p className="text-sm font-medium text-gray-800">
                    {s.req_name || s.product}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {s.company} / {s.submitter || '未知提交人'}
                  </p>
                  {s.purpose && (
                    <p className="text-sm text-gray-600 mt-2 line-clamp-2">{s.purpose}</p>
                  )}

                  {/* Technical Specs */}
                  <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-gray-500 bg-gray-50 rounded p-3">
                    {s.model_type && <span><span className="font-medium text-gray-600">類型：</span>{s.model_type}</span>}
                    {s.arch && <span><span className="font-medium text-gray-600">架構：</span>{s.arch}</span>}
                    {s.map50_target != null && <span><span className="font-medium text-gray-600">mAP50 目標：</span>{s.map50_target}</span>}
                    {s.map50_95_target != null && <span><span className="font-medium text-gray-600">mAP50-95 目標：</span>{s.map50_95_target}</span>}
                    {s.inference_latency_ms != null && <span><span className="font-medium text-gray-600">推論延遲上限：</span>{s.inference_latency_ms} ms</span>}
                    {s.model_size_limit_mb != null && <span><span className="font-medium text-gray-600">模型大小上限：</span>{s.model_size_limit_mb} MB</span>}
                    {s.input_spec && <span><span className="font-medium text-gray-600">輸入規格：</span>{s.input_spec}</span>}
                    {s.deploy_env && <span><span className="font-medium text-gray-600">部署環境：</span>{s.deploy_env}</span>}
                    {s.dataset_count && <span><span className="font-medium text-gray-600">資料集：</span>{s.dataset_count}</span>}
                    {s.dataset_val_count != null && <span><span className="font-medium text-gray-600">Validation：</span>{s.dataset_val_count} 張</span>}
                    {s.dataset_test_count != null && <span><span className="font-medium text-gray-600">Test：</span>{s.dataset_test_count} 張</span>}
                    {s.label_format && <span><span className="font-medium text-gray-600">標注格式：</span>{s.label_format}</span>}
                    {s.kaggle_dataset_url && (
                      <span className="col-span-2">
                        <span className="font-medium text-gray-600">Kaggle Dataset：</span>
                        <a
                          href={s.kaggle_dataset_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-indigo-500 hover:underline ml-1"
                        >
                          {s.kaggle_dataset_url}
                        </a>
                      </span>
                    )}
                  </div>
                </div>
                <span className="text-xs text-gray-400 ml-4 shrink-0">
                  {new Date(s.created_at).toLocaleDateString('zh-TW')}
                </span>
              </div>

              {errors[s.req_no] && (
                <div className="mt-2 text-sm text-red-600">{errors[s.req_no]}</div>
              )}

              <div className="mt-3 flex gap-2 items-end">
                <textarea
                  rows={2}
                  placeholder="核准意見（選填）"
                  className="flex-1 border rounded p-2 text-sm"
                  value={notes[s.req_no] ?? ''}
                  onChange={(e) =>
                    setNotes((prev) => ({ ...prev, [s.req_no]: e.target.value }))
                  }
                />
                <div className="flex flex-col gap-2">
                  <button
                    disabled={actionMut.isPending}
                    onClick={() =>
                      actionMut.mutate({
                        req_no: s.req_no,
                        action: 'approve',
                        note: notes[s.req_no],
                      })
                    }
                    className="px-4 py-1.5 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                  >
                    核准
                  </button>
                  <button
                    disabled={actionMut.isPending}
                    onClick={() => setRejectTarget(s)}
                    className="px-4 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
                  >
                    拒絕
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {rejectTarget && (
        <RejectModal
          submission={rejectTarget}
          onConfirm={(reasons, note) =>
            rejectMut.mutate({ req_no: rejectTarget.req_no, reasons, note })
          }
          onCancel={() => setRejectTarget(null)}
          isPending={rejectMut.isPending}
        />
      )}
    </div>
  )
}
