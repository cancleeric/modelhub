import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { submissionsApi, registryApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

// 依目前狀態決定可用的 action button
const AVAILABLE_ACTIONS: Record<string, { action: string; label: string; color: string }[]> = {
  draft:    [{ action: 'submit', label: '提交審核', color: 'blue' }],
  submitted:[
    { action: 'approve', label: '核准', color: 'green' },
    { action: 'reject',  label: '拒絕', color: 'red' },
  ],
  approved: [{ action: 'start_training', label: '開始訓練', color: 'yellow' }],
  training: [{ action: 'complete_training', label: '訓練完成', color: 'purple' }],
  trained:  [
    { action: 'accept', label: '驗收通過', color: 'emerald' },
    { action: 'fail',   label: '驗收失敗', color: 'red' },
  ],
}

const COLOR_MAP: Record<string, string> = {
  blue:    'bg-blue-600 hover:bg-blue-700 text-white',
  green:   'bg-green-600 hover:bg-green-700 text-white',
  red:     'bg-red-600 hover:bg-red-700 text-white',
  yellow:  'bg-yellow-500 hover:bg-yellow-600 text-white',
  purple:  'bg-purple-600 hover:bg-purple-700 text-white',
  emerald: 'bg-emerald-600 hover:bg-emerald-700 text-white',
}

export default function SubmissionDetailPage() {
  const { req_no } = useParams<{ req_no: string }>()
  const qc = useQueryClient()
  const [noteInput, setNoteInput] = useState('')
  const [actionError, setActionError] = useState('')

  const { data: sub, isLoading } = useQuery({
    queryKey: ['submission', req_no],
    queryFn: () => submissionsApi.get(req_no!),
    enabled: !!req_no,
  })

  const { data: versions } = useQuery({
    queryKey: ['registry', 'by-req', req_no],
    queryFn: () => registryApi.byReq(req_no!),
    enabled: !!req_no,
  })

  const actionMut = useMutation({
    mutationFn: ({ action, note }: { action: string; note?: string }) =>
      submissionsApi.action(req_no!, action, { note }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
      qc.invalidateQueries({ queryKey: ['submissions'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setNoteInput('')
      setActionError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '操作失敗'
      setActionError(String(msg))
    },
  })

  if (isLoading) return <p className="text-gray-500">載入中...</p>
  if (!sub) return <p className="text-red-500">找不到需求單 {req_no}</p>

  const actions = AVAILABLE_ACTIONS[sub.status] ?? []

  return (
    <div className="max-w-3xl">
      <div className="flex items-center gap-3 mb-6">
        <Link to="/" className="text-gray-400 hover:text-gray-600 text-sm">← 返回列表</Link>
        <span className="text-gray-300">/</span>
        <span className="font-mono text-gray-700">{sub.req_no}</span>
        <StatusBadge status={sub.status} />
      </div>

      {/* 基本資訊 */}
      <div className="bg-white rounded shadow p-5 mb-4">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">
          {sub.req_name || sub.product}
        </h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
          <Item label="需求單號" value={sub.req_no} mono />
          <Item label="產品" value={sub.product} />
          <Item label="業務公司" value={sub.company} />
          <Item label="提交人" value={sub.submitter} />
          <Item label="優先度" value={sub.priority} mono />
          <Item label="模型類型" value={sub.model_type} />
          <Item label="mAP50 目標" value={sub.map50_target?.toString()} />
          <Item label="架構" value={sub.arch} />
          <Item label="資料來源" value={sub.dataset_source} />
          <Item label="標注格式" value={sub.label_format} />
          <Item label="預計交付" value={sub.expected_delivery} />
          <Item label="建立時間" value={new Date(sub.created_at).toLocaleString('zh-TW')} />
        </dl>
        {sub.purpose && (
          <div className="mt-4">
            <span className="text-xs font-medium text-gray-500 uppercase">業務描述</span>
            <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{sub.purpose}</p>
          </div>
        )}
        {sub.class_list && (
          <div className="mt-3">
            <span className="text-xs font-medium text-gray-500 uppercase">類別清單</span>
            <p className="text-sm text-gray-700 mt-1 font-mono">{sub.class_list}</p>
          </div>
        )}
        {sub.reviewer_note && (
          <div className="mt-3 bg-yellow-50 border border-yellow-200 rounded p-3">
            <span className="text-xs font-medium text-yellow-700">審核意見</span>
            <p className="text-sm text-gray-700 mt-1">{sub.reviewer_note}</p>
          </div>
        )}
      </div>

      {/* 狀態機操作 */}
      {actions.length > 0 && (
        <div className="bg-white rounded shadow p-5 mb-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">狀態操作</h3>
          {actionError && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-3 py-2 mb-3">
              {actionError}
            </div>
          )}
          <textarea
            className="w-full border rounded p-2 text-sm mb-3"
            rows={2}
            placeholder="備注（審核意見、拒絕原因等）"
            value={noteInput}
            onChange={(e) => setNoteInput(e.target.value)}
          />
          <div className="flex gap-2">
            {actions.map(({ action, label, color }) => (
              <button
                key={action}
                disabled={actionMut.isPending}
                onClick={() => actionMut.mutate({ action, note: noteInput || undefined })}
                className={`px-4 py-2 text-sm rounded font-medium ${COLOR_MAP[color]} disabled:opacity-50`}
              >
                {actionMut.isPending ? '處理中...' : label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 相關模型版本 */}
      <div className="bg-white rounded shadow p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">相關模型版本</h3>
        {!versions || versions.length === 0 ? (
          <p className="text-sm text-gray-400">尚無模型版本記錄</p>
        ) : (
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">版本</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">訓練日期</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">mAP50</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">Pass/Fail</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">狀態</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">驗收</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {versions.map((v) => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 font-mono">{v.version}</td>
                  <td className="px-3 py-2 text-gray-500">{v.train_date || '-'}</td>
                  <td className="px-3 py-2">{v.map50_actual ?? v.map50 ?? '-'}</td>
                  <td className="px-3 py-2">
                    {v.pass_fail ? (
                      <span
                        className={`text-xs font-bold ${
                          v.pass_fail === 'pass' ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {v.pass_fail.toUpperCase()}
                      </span>
                    ) : '-'}
                  </td>
                  <td className="px-3 py-2"><StatusBadge status={v.status} /></td>
                  <td className="px-3 py-2">
                    <Link
                      to={`/registry/${v.id}/accept`}
                      className="text-indigo-600 text-xs hover:underline"
                    >
                      驗收
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function Item({
  label,
  value,
  mono,
}: {
  label: string
  value?: string | null
  mono?: boolean
}) {
  return (
    <div>
      <dt className="text-xs text-gray-400 uppercase font-medium">{label}</dt>
      <dd className={`text-gray-800 mt-0.5 ${mono ? 'font-mono text-xs' : ''}`}>
        {value || <span className="text-gray-300">-</span>}
      </dd>
    </div>
  )
}
