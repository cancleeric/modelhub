import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { registryApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

const API_KEY = import.meta.env.VITE_MODELHUB_API_KEY ?? 'modelhub-dev-key-2026'

function downloadModel(id: number, filename: string) {
  const url = `/api/registry/${id}/download`
  const a = document.createElement('a')
  a.href = url
  a.download = filename || `model-${id}.pt`
  // API Key 下載：透過 fetch + blob，因為需要自訂 header
  fetch(url, { headers: { 'X-Api-Key': API_KEY } })
    .then((r) => {
      if (!r.ok) throw new Error(`下載失敗 (${r.status})`)
      return r.blob()
    })
    .then((blob) => {
      const objectUrl = URL.createObjectURL(blob)
      a.href = objectUrl
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(objectUrl)
    })
    .catch((err: Error) => alert(err.message))
}

const STATUS_OPTIONS = ['', 'active', 'pending_acceptance', 'retired', 'testing']

export default function RegistryPage() {
  const [productFilter, setProductFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [reqNoFilter, setReqNoFilter] = useState('')
  const [onlyCurrent, setOnlyCurrent] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['registry', productFilter, statusFilter, reqNoFilter],
    queryFn: () =>
      registryApi.list({
        product: productFilter || undefined,
        status: statusFilter || undefined,
        req_no: reqNoFilter || undefined,
      }),
  })

  const filtered = data?.filter((v) => (onlyCurrent ? v.is_current : true))

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">模型版本清冊</h1>
      </div>

      <div className="flex gap-3 mb-4 flex-wrap items-center">
        <select
          className="border rounded px-3 py-1.5 text-sm bg-white"
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
          className="border rounded px-3 py-1.5 text-sm"
          value={productFilter}
          onChange={(e) => setProductFilter(e.target.value)}
        />
        <input
          type="text"
          placeholder="需求單號 (例：MH-2026-001)"
          className="border rounded px-3 py-1.5 text-sm font-mono"
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

      {isLoading && <p className="text-gray-500">載入中...</p>}

      {filtered && (
        <div className="bg-white rounded shadow overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">需求單號</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">產品</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">模型名稱</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">版本</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">架構</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">mAP50</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">Pass/Fail</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">狀態</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">檔案</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filtered.map((v) => (
                <tr key={v.id} className={`hover:bg-gray-50 ${v.is_current ? 'bg-green-50' : ''}`}>
                  <td className="px-4 py-3">
                    <Link
                      to={`/submissions/${v.req_no}`}
                      className="text-indigo-600 hover:underline font-mono text-xs"
                    >
                      {v.req_no}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-700">{v.product}</td>
                  <td className="px-4 py-3 text-gray-800 font-medium">
                    <span>{v.model_name}</span>
                    {v.is_current && (
                      <span className="ml-2 inline-block px-1.5 py-0.5 text-xs font-bold bg-green-600 text-white rounded">
                        CURRENT
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-gray-600">{v.version}</td>
                  <td className="px-4 py-3 text-gray-500 text-xs">{v.arch || '-'}</td>
                  <td className="px-4 py-3">
                    {v.map50_actual ?? v.map50 ?? '-'}
                  </td>
                  <td className="px-4 py-3">
                    {v.pass_fail ? (
                      <span
                        className={`text-xs font-bold ${
                          v.pass_fail === 'pass' ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {v.pass_fail.toUpperCase()}
                      </span>
                    ) : (
                      <span className="text-gray-300">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={v.status} />
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-gray-500">
                    {v.file_path || '-'}
                  </td>
                  <td className="px-4 py-3 flex gap-2 items-center">
                    <Link
                      to={`/registry/${v.id}/accept`}
                      className="text-indigo-600 text-xs hover:underline"
                    >
                      驗收
                    </Link>
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
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={10} className="text-center py-8 text-gray-400">
                    無資料
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
