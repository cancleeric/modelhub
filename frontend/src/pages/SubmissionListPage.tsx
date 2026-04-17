import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { submissionsApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

const STATUS_OPTIONS = [
  '', 'draft', 'submitted', 'approved', 'rejected',
  'training', 'trained', 'accepted', 'failed',
]

const PRIORITY_OPTIONS = ['', 'P0', 'P1', 'P2', 'P3']

function getStartOfWeek(): Date {
  const now = new Date()
  const day = now.getDay() // 0=Sun
  const diff = now.getDate() - day + (day === 0 ? -6 : 1) // Mon
  const mon = new Date(now)
  mon.setDate(diff)
  mon.setHours(0, 0, 0, 0)
  return mon
}

const DATASET_STATUS_LABEL: Record<string, string> = {
  ready: '資料集就緒',
  missing_labels: '缺標籤',
  missing_data: '資料不足',
  partial: '部分完成',
}

const DATASET_STATUS_COLOR: Record<string, string> = {
  ready: 'bg-green-100 text-green-700',
  missing_labels: 'bg-orange-100 text-orange-700',
  missing_data: 'bg-red-100 text-red-700',
  partial: 'bg-yellow-100 text-yellow-700',
}

export default function SubmissionListPage() {
  const [statusFilter, setStatusFilter] = useState('')
  const [priorityFilter, setPriorityFilter] = useState('')
  const [productFilter, setProductFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')

  // Load all submissions (no status filter at API level so we can count stats)
  const { data: allData } = useQuery({
    queryKey: ['submissions-all'],
    queryFn: () => submissionsApi.list(),
  })

  // Load filtered submissions for table
  const { data, isLoading, error } = useQuery({
    queryKey: ['submissions', statusFilter, productFilter, datasetFilter],
    queryFn: () =>
      submissionsApi.list({
        status: statusFilter || undefined,
        product: productFilter || undefined,
        dataset_status: datasetFilter || undefined,
      }),
  })

  // Compute 4 stats from allData
  const statsCards = useMemo(() => {
    if (!allData) return null
    const weekStart = getStartOfWeek()
    const pending = allData.filter((s) => s.status === 'submitted').length
    const training = allData.filter((s) => s.status === 'training').length
    const approvedThisWeek = allData.filter(
      (s) => s.status === 'approved' && new Date(s.created_at) >= weekStart,
    ).length
    const totalGpuSec = allData.reduce((acc, s) => acc + (s.gpu_seconds || 0), 0)
    const totalGpuHours = Math.floor(totalGpuSec / 3600)
    const totalCost = allData.reduce((acc, s) => acc + (s.estimated_cost_usd || 0), 0)
    return { pending, training, approvedThisWeek, totalGpuHours, totalCost }
  }, [allData])

  // Priority filter applied on frontend
  const filteredData = useMemo(() => {
    if (!data) return data
    if (!priorityFilter) return data
    return data.filter((s) => s.priority === priorityFilter)
  }, [data, priorityFilter])

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

      {/* Summary Cards */}
      {statsCards && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded shadow-sm p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-yellow-100 flex items-center justify-center text-yellow-600 font-bold text-lg">
              {statsCards.pending}
            </div>
            <div>
              <div className="text-sm text-gray-500">待審</div>
              <div className="text-xs text-gray-400">submitted 狀態</div>
            </div>
          </div>
          <div className="bg-white rounded shadow-sm p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-lg">
              {statsCards.training}
            </div>
            <div>
              <div className="text-sm text-gray-500">訓練中</div>
              <div className="text-xs text-gray-400">training 狀態</div>
            </div>
          </div>
          <div className="bg-white rounded shadow-sm p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center text-green-600 font-bold text-lg">
              {statsCards.approvedThisWeek}
            </div>
            <div>
              <div className="text-sm text-gray-500">本週核准</div>
              <div className="text-xs text-gray-400">approved，本週建立</div>
            </div>
          </div>
          <div className="bg-white rounded shadow-sm p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center text-purple-600 font-bold text-base">
              {statsCards.totalGpuHours}h
            </div>
            <div>
              <div className="text-sm text-gray-500">累計 GPU</div>
              <div className="text-xs text-gray-400">估算 ${statsCards.totalCost.toFixed(2)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
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
        <select
          className="border rounded px-3 py-1.5 text-sm bg-white"
          value={priorityFilter}
          onChange={(e) => setPriorityFilter(e.target.value)}
        >
          <option value="">全部優先度</option>
          {PRIORITY_OPTIONS.filter(Boolean).map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Product 篩選..."
          className="border rounded px-3 py-1.5 text-sm"
          value={productFilter}
          onChange={(e) => setProductFilter(e.target.value)}
        />
        <select
          className="border rounded px-3 py-1.5 text-sm bg-white"
          value={datasetFilter}
          onChange={(e) => setDatasetFilter(e.target.value)}
        >
          <option value="">全部資料集狀態</option>
          <option value="ready">可訓練（ready）</option>
          <option value="missing_labels">缺標籤</option>
          <option value="missing_data">資料不足</option>
          <option value="partial">部分完成</option>
        </select>
      </div>

      {/* Table */}
      {isLoading && <p className="text-gray-500">載入中...</p>}
      {error && <p className="text-red-500">載入失敗</p>}
      {filteredData && (
        <div className="bg-white rounded shadow overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">需求單號</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">名稱</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">產品</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">提交人</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">狀態</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">資料集</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">優先</th>
                <th className="text-left px-4 py-3 text-gray-600 font-medium">建立時間</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filteredData.map((s) => (
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
                    {s.dataset_status ? (
                      <span
                        className={`text-xs font-medium px-2 py-0.5 rounded ${
                          DATASET_STATUS_COLOR[s.dataset_status] ?? 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {DATASET_STATUS_LABEL[s.dataset_status] ?? s.dataset_status}
                      </span>
                    ) : (
                      <span className="text-gray-300 text-xs">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs font-mono text-gray-500">{s.priority}</span>
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs">
                    {new Date(s.created_at).toLocaleDateString('zh-TW')}
                  </td>
                </tr>
              ))}
              {filteredData.length === 0 && (
                <tr>
                  <td colSpan={8} className="text-center py-8 text-gray-400">
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
