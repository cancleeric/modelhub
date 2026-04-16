import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { submissionsApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

export default function ReviewPage() {
  const qc = useQueryClient()
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})

  const { data, isLoading } = useQuery({
    queryKey: ['submissions', 'submitted'],
    queryFn: () => submissionsApi.list({ status: 'submitted' }),
  })

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
    },
    onError: (err: unknown, vars) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '操作失敗'
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
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Link
                      to={`/submissions/${s.req_no}`}
                      className="font-mono text-indigo-600 hover:underline"
                    >
                      {s.req_no}
                    </Link>
                    <StatusBadge status={s.status} />
                    <span className="text-xs font-mono text-gray-500">{s.priority}</span>
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
                  <div className="flex gap-4 mt-1 text-xs text-gray-400">
                    {s.map50_target != null && (
                      <span>mAP50 目標：{s.map50_target}</span>
                    )}
                    {s.model_type && <span>類型：{s.model_type}</span>}
                    {s.dataset_count && <span>資料集：{s.dataset_count}</span>}
                  </div>
                </div>
                <span className="text-xs text-gray-400">
                  {new Date(s.created_at).toLocaleDateString('zh-TW')}
                </span>
              </div>

              {errors[s.req_no] && (
                <div className="mt-2 text-sm text-red-600">{errors[s.req_no]}</div>
              )}

              <div className="mt-3 flex gap-2 items-end">
                <textarea
                  rows={2}
                  placeholder="審核意見（拒絕時必填）"
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
                    onClick={() => {
                      if (!notes[s.req_no]) {
                        setErrors((prev) => ({
                          ...prev,
                          [s.req_no]: '拒絕必須填寫審核意見',
                        }))
                        return
                      }
                      actionMut.mutate({
                        req_no: s.req_no,
                        action: 'reject',
                        note: notes[s.req_no],
                      })
                    }}
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
    </div>
  )
}
