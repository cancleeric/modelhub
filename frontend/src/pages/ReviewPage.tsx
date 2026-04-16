import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { submissionsApi, type Submission } from '../api/client'
import StatusBadge from '../components/StatusBadge'

const REJECT_REASONS = [
  'expected_delivery 未填',
  '資料集數量不足',
  '資料集 test set 未規劃',
  'submitter 非真實聯絡人',
  'map50_95_target 未填',
  'model_type 格式不符',
  '架構選型需重新評估',
]

interface RejectModalProps {
  submission: Submission
  onConfirm: (note: string) => void
  onCancel: () => void
  isPending: boolean
}

function RejectModal({ submission, onConfirm, onCancel, isPending }: RejectModalProps) {
  const [checked, setChecked] = useState<string[]>([])
  const [otherChecked, setOtherChecked] = useState(false)
  const [otherText, setOtherText] = useState('')

  const toggleReason = (reason: string) => {
    setChecked((prev) =>
      prev.includes(reason) ? prev.filter((r) => r !== reason) : [...prev, reason],
    )
  }

  const buildNote = () => {
    const parts: string[] = []
    if (checked.length > 0) {
      parts.push('退件原因：')
      checked.forEach((r) => parts.push(`- ${r}`))
      if (otherChecked) parts.push('- 其他')
    }
    if (otherChecked && otherText.trim()) {
      parts.push(`補充說明：${otherText.trim()}`)
    }
    return parts.join('\n')
  }

  const canConfirm = checked.length > 0 || (otherChecked && otherText.trim())

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">退件確認</h3>
        <p className="text-sm text-gray-500 mb-4">
          {submission.req_no} — {submission.req_name || submission.product}
        </p>

        <p className="text-sm font-medium text-gray-700 mb-2">退件原因（可複選）</p>
        <div className="space-y-2 mb-4">
          {REJECT_REASONS.map((reason) => (
            <label key={reason} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={checked.includes(reason)}
                onChange={() => toggleReason(reason)}
                className="rounded"
              />
              {reason}
            </label>
          ))}
          <label className="flex items-start gap-2 text-sm text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={otherChecked}
              onChange={(e) => setOtherChecked(e.target.checked)}
              className="rounded mt-0.5"
            />
            其他（自由文字）
          </label>
        </div>

        {otherChecked && (
          <textarea
            rows={3}
            placeholder="補充說明..."
            className="w-full border rounded p-2 text-sm mb-4"
            value={otherText}
            onChange={(e) => setOtherText(e.target.value)}
          />
        )}

        {!canConfirm && (
          <p className="text-xs text-red-500 mb-3">請至少選擇一個退件原因</p>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={isPending}
            className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-50 disabled:opacity-50"
          >
            取消
          </button>
          <button
            type="button"
            disabled={!canConfirm || isPending}
            onClick={() => onConfirm(buildNote())}
            className="px-4 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
          >
            {isPending ? '處理中...' : '確認退件'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function ReviewPage() {
  const qc = useQueryClient()
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [rejectTarget, setRejectTarget] = useState<Submission | null>(null)

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
      setRejectTarget(null)
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
          onConfirm={(note) =>
            actionMut.mutate({ req_no: rejectTarget.req_no, action: 'reject', note })
          }
          onCancel={() => setRejectTarget(null)}
          isPending={actionMut.isPending}
        />
      )}
    </div>
  )
}
