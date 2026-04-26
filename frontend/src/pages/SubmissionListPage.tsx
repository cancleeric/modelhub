import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { submissionsApi, api } from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { KpiCard } from '../design-system/KpiCard'
import { SkeletonKpiCard } from '../design-system/Skeleton'

interface QueueWaitingItem {
  req_no: string
  priority: string
  position: number
  enqueued_at: string
}

interface QueueRunningItem {
  req_no: string
  target_resource: string | null
  dispatched_at: string | null
  status: string
}

interface QueueStatus {
  waiting: QueueWaitingItem[]
  running: QueueRunningItem[]
  max_concurrent: number
  count_waiting: number
  count_running: number
}

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

  // F-01/B-01: 後端分頁，limit=1000 確保統計完整
  const { data, isLoading, error } = useQuery({
    queryKey: ['submissions', statusFilter, productFilter, datasetFilter],
    queryFn: () =>
      submissionsApi.list({
        status: statusFilter || undefined,
        product: productFilter || undefined,
        dataset_status: datasetFilter || undefined,
        limit: 1000,
        offset: 0,
      }),
    refetchInterval: (query) => {
      const items = query.state.data?.items
      if (items && items.some((s) => s.status === 'training')) return 30000
      return false
    },
  })

  // F-04: summary card 永遠從篩選後的 items 計算，確保一致
  const statsCards = useMemo(() => {
    const items = data?.items
    if (!items) return null
    const weekStart = getStartOfWeek()
    const pending = items.filter((s) => s.status === 'submitted').length
    const training = items.filter((s) => s.status === 'training').length
    const approvedThisWeek = items.filter(
      (s) => s.status === 'approved' && s.reviewed_at && new Date(s.reviewed_at) >= weekStart,
    ).length
    const totalGpuSec = items.reduce((acc, s) => acc + (s.gpu_seconds || 0), 0)
    const totalGpuHours = Math.floor(totalGpuSec / 3600)
    const totalCost = items.reduce((acc, s) => acc + (s.estimated_cost_usd || 0), 0)
    return { pending, training, approvedThisWeek, totalGpuHours, totalCost }
  }, [data])

  // 訓練隊列狀態（每 30 秒 refresh）
  const { data: queueStatus } = useQuery<QueueStatus>({
    queryKey: ['queue-status'],
    queryFn: () => api.get<QueueStatus>('/api/queue/status').then((r) => r.data),
    refetchInterval: 30000,
  })

  // F-04: Priority filter 在前端做，從 items 篩選
  const filteredData = useMemo(() => {
    const items = data?.items
    if (!items) return items
    if (!priorityFilter) return items
    return items.filter((s) => s.priority === priorityFilter)
  }, [data, priorityFilter])

  const total = data?.total ?? 0

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

      {/* Summary Cards — F-04: 矩形 KpiCard（M4），isLoading 顯示 Skeleton */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-2">
        {isLoading ? (
          <>
            <SkeletonKpiCard />
            <SkeletonKpiCard />
            <SkeletonKpiCard />
            <SkeletonKpiCard />
          </>
        ) : statsCards ? (
          <>
            <KpiCard label="待審" value={statsCards.pending} sub="submitted 狀態" tone="amber" />
            <KpiCard label="訓練中" value={statsCards.training} sub="training 狀態" tone="blue" />
            <KpiCard label="本週核准" value={statsCards.approvedThisWeek} sub="approved，本週建立" tone="green" />
            <KpiCard
              label="累計 GPU"
              value={`${statsCards.totalGpuHours}h`}
              sub={`估算 $${statsCards.totalCost.toFixed(2)}`}
              tone="purple"
            />
          </>
        ) : null}
      </div>
      {/* F-04: 篩選結果提示 */}
      {data && (
        <div className="text-xs text-gray-400 mb-4 pl-1">
          目前篩選結果：共 {total} 筆
          {priorityFilter && filteredData && filteredData.length !== total
            ? `，顯示 ${filteredData.length} 筆（${priorityFilter}）`
            : ''}
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
                    <span className="inline-flex items-center gap-1.5">
                      <StatusBadge status={s.status} />
                      {s.status === 'training' && (
                        <span
                          className="inline-block w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"
                          title="訓練中"
                        />
                      )}
                      {s.status === 'approved' && queueStatus && (() => {
                        const waitingEntry = queueStatus.waiting.find((w) => w.req_no === s.req_no)
                        const runningEntry = queueStatus.running.find((r) => r.req_no === s.req_no)
                        if (waitingEntry) {
                          return (
                            <span
                              className="text-xs font-medium px-1.5 py-0.5 rounded bg-orange-100 text-orange-700"
                              title={`排隊中，優先度 ${waitingEntry.priority}`}
                            >
                              排隊中 #{waitingEntry.position}
                            </span>
                          )
                        }
                        if (runningEntry) {
                          return (
                            <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-blue-100 text-blue-700">
                              派發中
                            </span>
                          )
                        }
                        return null
                      })()}
                    </span>
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
