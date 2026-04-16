import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { submissionsApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

const STATUS_OPTIONS = [
  '', 'draft', 'submitted', 'approved', 'rejected',
  'training', 'trained', 'accepted', 'failed',
]

export default function SubmissionListPage() {
  const [statusFilter, setStatusFilter] = useState('')
  const [productFilter, setProductFilter] = useState('')

  const { data, isLoading, error } = useQuery({
    queryKey: ['submissions', statusFilter, productFilter],
    queryFn: () =>
      submissionsApi.list({
        status: statusFilter || undefined,
        product: productFilter || undefined,
      }),
  })

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: submissionsApi.stats,
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">訓練需求單列表</h1>
        <Link
          to="/submit"
          className="bg-indigo-600 text-white text-sm px-4 py-2 rounded hover:bg-indigo-700"
        >
          + 新增需求單
        </Link>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-3 mb-6">
          {Object.entries(stats.by_status).map(([s, c]) => (
            <div key={s} className="bg-white rounded shadow-sm p-3 text-center">
              <div className="text-2xl font-bold text-gray-800">{c}</div>
              <StatusBadge status={s} />
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3 mb-4">
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
      </div>

      {/* Table */}
      {isLoading && <p className="text-gray-500">載入中...</p>}
      {error && <p className="text-red-500">載入失敗</p>}
      {data && (
        <div className="bg-white rounded shadow overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">需求單號</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">名稱</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">產品</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">提交人</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">狀態</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">優先</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">建立時間</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {data.map((s) => (
                <tr key={s.req_no} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link
                      to={`/submissions/${s.req_no}`}
                      className="text-indigo-600 hover:underline font-mono"
                    >
                      {s.req_no}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-700">{s.req_name || '-'}</td>
                  <td className="px-4 py-3 text-gray-700">{s.product}</td>
                  <td className="px-4 py-3 text-gray-500">{s.submitter || '-'}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={s.status} />
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs font-mono text-gray-500">{s.priority}</span>
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs">
                    {new Date(s.created_at).toLocaleDateString('zh-TW')}
                  </td>
                </tr>
              ))}
              {data.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-gray-400">
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
