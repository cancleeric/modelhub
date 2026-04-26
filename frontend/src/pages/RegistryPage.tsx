import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { registryApi } from '../api/client'
import { getToken } from '../auth'
import StatusBadge from '../components/StatusBadge'
import { EmptyState } from '../design-system/EmptyState'
import { SkeletonRect } from '../design-system/Skeleton'
import { Database, X, Download, CheckCircle } from 'lucide-react'

function downloadModel(id: number, filename: string) {
  const url = `/api/registry/${id}/download`
  const token = getToken()
  if (!token) {
    alert('尚未登入，請先登入再下載')
    return
  }
  fetch(url, { headers: { 'Authorization': `Bearer ${token}` } })
    .then((r) => {
      if (!r.ok) throw new Error(`下載失敗 (${r.status})`)
      return r.blob()
    })
    .then((blob) => {
      const objectUrl = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = objectUrl
      a.download = filename || `model-${id}.pt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(objectUrl)
    })
    .catch((err: Error) => alert(err.message))
}

const STATUS_OPTIONS = ['', 'active', 'pending_acceptance', 'retired', 'testing']

// 版本資料型別（只列需要的欄位）
interface ModelVersionRow {
  id: number
  req_no: string
  product: string
  model_name: string
  version: string
  arch?: string
  map50?: number | null
  map50_actual?: number | null
  map50_95?: number | null
  map50_95_actual?: number | null
  file_path?: string | null
  status: string
  pass_fail?: string | null
  accepted_by?: string | null
  accepted_at?: string | null
  is_current: boolean
  created_at: string
  train_date?: string | null
}

function DrawerDetail({
  v,
  onClose,
}: {
  v: ModelVersionRow
  onClose: () => void
}) {
  const navigate = useNavigate()
  const map50Display = v.map50_actual ?? v.map50 ?? '-'
  const map9595Display = v.map50_95_actual ?? v.map50_95 ?? '-'

  return (
    <>
      {/* 遮罩 */}
      <div
        className="fixed inset-0 z-40 bg-black/30"
        onClick={onClose}
      />
      {/* Drawer */}
      <div className="fixed top-0 right-0 z-50 h-full w-full max-w-md bg-white shadow-2xl flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <div>
            <h2 className="text-base font-semibold text-gray-800">{v.model_name}</h2>
            <div className="text-xs text-gray-400 font-mono mt-0.5">{v.version}</div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            aria-label="關閉"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <Row label="需求單號">
            <Link to={`/submissions/${v.req_no}`} className="text-indigo-600 hover:underline font-mono text-sm">
              {v.req_no}
            </Link>
          </Row>
          <Row label="產品">{v.product}</Row>
          <Row label="架構">{v.arch || '-'}</Row>
          <Row label="mAP50">{map50Display}</Row>
          <Row label="mAP50-95">{map9595Display}</Row>
          <Row label="Pass/Fail">
            {v.pass_fail ? (
              <span className={`text-sm font-bold ${v.pass_fail === 'pass' ? 'text-emerald-600' : 'text-red-600'}`}>
                {v.pass_fail.toUpperCase()}
              </span>
            ) : '-'}
          </Row>
          <Row label="狀態"><StatusBadge status={v.status} /></Row>
          <Row label="當前版本">
            {v.is_current ? (
              <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-700">
                <CheckCircle size={14} /> 是
              </span>
            ) : '否'}
          </Row>
          <Row label="驗收人">{v.accepted_by || '-'}</Row>
          <Row label="驗收時間">
            {v.accepted_at ? new Date(v.accepted_at).toLocaleDateString('zh-TW') : '-'}
          </Row>
          <Row label="訓練日期">
            {v.train_date ? new Date(v.train_date).toLocaleDateString('zh-TW') : '-'}
          </Row>
          <Row label="檔案路徑">
            <span className="font-mono text-xs text-gray-500 break-all">{v.file_path || '-'}</span>
          </Row>
        </div>

        <div className="px-5 py-4 border-t border-gray-100 flex gap-3">
          {v.status === 'pending_acceptance' && (
            <button
              onClick={() => { navigate(`/registry/${v.id}/accept`); onClose() }}
              className="flex-1 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
            >
              前往驗收
            </button>
          )}
          {v.file_path && (
            <button
              onClick={() => downloadModel(v.id, v.file_path!.split('/').pop() ?? `model-${v.id}.pt`)}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 text-sm rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Download size={14} />
              下載
            </button>
          )}
        </div>
      </div>
    </>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-4">
      <div className="w-24 text-xs text-gray-400 flex-shrink-0 pt-0.5">{label}</div>
      <div className="text-sm text-gray-800 flex-1">{children}</div>
    </div>
  )
}

export default function RegistryPage() {
  const [productFilter, setProductFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [reqNoFilter, setReqNoFilter] = useState('')
  const [onlyCurrent, setOnlyCurrent] = useState(false)
  const [selectedRow, setSelectedRow] = useState<ModelVersionRow | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['registry', productFilter, statusFilter, reqNoFilter],
    queryFn: () =>
      registryApi.list({
        product: productFilter || undefined,
        status: statusFilter || undefined,
        req_no: reqNoFilter || undefined,
      }),
  })

  const filtered = data?.filter((v) => (onlyCurrent ? v.is_current : true)) as ModelVersionRow[] | undefined

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">模型版本清冊</h1>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap items-center">
        <select
          className="border rounded-md px-3 py-1.5 text-sm bg-white"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">全部狀態</option>
          {STATUS_OPTIONS.filter(Boolean).map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Product 篩選..."
          className="border rounded-md px-3 py-1.5 text-sm"
          value={productFilter}
          onChange={(e) => setProductFilter(e.target.value)}
        />
        <input
          type="text"
          placeholder="需求單號 (例：MH-2026-001)"
          className="border rounded-md px-3 py-1.5 text-sm font-mono"
          value={reqNoFilter}
          onChange={(e) => setReqNoFilter(e.target.value)}
        />
        <label className="inline-flex items-center gap-1.5 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={onlyCurrent}
            onChange={(e) => setOnlyCurrent(e.target.checked)}
          />
          僅顯示 current
        </label>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-x-auto">
        {isLoading ? (
          <div className="p-6 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <SkeletonRect key={i} height={36} />
            ))}
          </div>
        ) : filtered && filtered.length > 0 ? (
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-100">
              <tr>
                <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs uppercase tracking-wide whitespace-nowrap">模型</th>
                <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs uppercase tracking-wide whitespace-nowrap">版本</th>
                <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs uppercase tracking-wide whitespace-nowrap">狀態</th>
                <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs uppercase tracking-wide whitespace-nowrap">日期</th>
                <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs uppercase tracking-wide whitespace-nowrap">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filtered.map((v) => (
                <tr
                  key={v.id}
                  className={`hover:bg-indigo-50/40 cursor-pointer transition-colors ${v.is_current ? 'bg-emerald-50/40' : ''}`}
                  onClick={() => setSelectedRow(v)}
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-800">{v.model_name}</div>
                    <div className="text-xs text-gray-400 font-mono">{v.req_no}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-gray-600">{v.version}</span>
                    {v.is_current && (
                      <span className="ml-2 inline-block px-1.5 py-0.5 text-[10px] font-bold bg-emerald-600 text-white rounded">
                        CURRENT
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={v.status} />
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs whitespace-nowrap">
                    {new Date(v.created_at).toLocaleDateString('zh-TW')}
                  </td>
                  <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                    <div className="flex gap-2 items-center">
                      {v.status === 'pending_acceptance' ? (
                        <Link
                          to={`/registry/${v.id}/accept`}
                          className="text-indigo-600 text-xs hover:underline"
                          data-testid={`accept-link-${v.id}`}
                        >
                          驗收
                        </Link>
                      ) : (
                        <span className="text-xs text-gray-400">
                          {v.pass_fail ? (v.pass_fail === 'pass' ? '已通過' : '已失敗') : '已完成'}
                        </span>
                      )}
                      {v.file_path && (
                        <button
                          onClick={() =>
                            downloadModel(
                              v.id,
                              v.file_path!.split('/').pop() ?? `model-${v.id}.pt`,
                            )
                          }
                          className="text-xs px-2 py-0.5 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors"
                        >
                          下載
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <EmptyState
            title="尚無模型版本"
            description="審核通過並完成訓練的需求單，模型版本將顯示於此"
            Icon={Database}
          />
        )}
      </div>

      {/* Row 詳情抽屜 */}
      {selectedRow && (
        <DrawerDetail
          v={selectedRow}
          onClose={() => setSelectedRow(null)}
        />
      )}
    </div>
  )
}
