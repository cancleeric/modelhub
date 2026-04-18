/**
 * QueuePage.tsx — 訓練隊列管理頁（Sprint 24 Task 24-3）
 *
 * 列出 waiting/running 條目，reviewer 可調整優先序或取消排隊。
 * 路由：/queue
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { Link } from 'react-router-dom'

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

const PRIORITY_OPTIONS = ['P0', 'P1', 'P2', 'P3']

const PRIORITY_COLORS: Record<string, string> = {
  P0: 'bg-red-100 text-red-700 border-red-200',
  P1: 'bg-orange-100 text-orange-700 border-orange-200',
  P2: 'bg-blue-100 text-blue-700 border-blue-200',
  P3: 'bg-gray-100 text-gray-600 border-gray-200',
}

function formatTime(isoStr: string | null): string {
  if (!isoStr) return '-'
  return new Date(isoStr).toLocaleString('zh-TW', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function QueuePage() {
  const queryClient = useQueryClient()

  const { data: queueStatus, isLoading, error } = useQuery<QueueStatus>({
    queryKey: ['queue-status'],
    queryFn: () => api.get<QueueStatus>('/api/queue/status').then((r) => r.data),
    refetchInterval: 15000,
  })

  // 調整優先序 mutation（呼叫 approve action 並傳入新優先序，或直接 PATCH submission priority）
  const updatePriorityMutation = useMutation({
    mutationFn: async ({ req_no, priority }: { req_no: string; priority: string }) => {
      // PATCH submission priority 欄位
      await api.patch(`/api/submissions/${req_no}`, { priority })
      // 重新 enqueue（透過 re-approve 觸發，或由 dispatcher 自動讀取新優先序）
      // 此處先更新 submission priority，queue_dispatcher 下次 peek_next 會重新排序
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['queue-status'] })
    },
  })

  if (isLoading) {
    return <p className="text-gray-500">載入中...</p>
  }
  if (error) {
    return <p className="text-red-500">載入失敗</p>
  }

  const { waiting = [], running = [], max_concurrent = 2 } = queueStatus ?? {}

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">訓練隊列</h1>
        <div className="flex gap-3 text-sm text-gray-500">
          <span>上限：{max_concurrent} 個並行</span>
          <span>等待中：{waiting.length}</span>
          <span>執行中：{running.length}</span>
        </div>
      </div>

      {/* Running */}
      <div className="mb-8">
        <h2 className="text-base font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse" />
          執行中 ({running.length})
        </h2>
        {running.length === 0 ? (
          <p className="text-sm text-gray-400 bg-white rounded shadow p-4">目前無任務執行中</p>
        ) : (
          <div className="bg-white rounded shadow overflow-hidden">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">需求單號</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">狀態</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">資源</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">派發時間</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {running.map((r) => (
                  <tr key={r.req_no} className="bg-blue-50/30">
                    <td className="px-4 py-3">
                      <Link
                        to={`/submissions/${r.req_no}`}
                        className="text-indigo-600 hover:underline font-mono"
                      >
                        {r.req_no}
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded bg-blue-100 text-blue-700">
                        <span className="w-2 h-2 rounded-full border border-blue-500 border-t-transparent animate-spin" />
                        {r.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600 text-xs font-mono">
                      {r.target_resource || '-'}
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">
                      {formatTime(r.dispatched_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Waiting */}
      <div>
        <h2 className="text-base font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-orange-400" />
          等待中 ({waiting.length})
        </h2>
        {waiting.length === 0 ? (
          <p className="text-sm text-gray-400 bg-white rounded shadow p-4">隊列空閒</p>
        ) : (
          <div className="bg-white rounded shadow overflow-hidden">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">排名</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">需求單號</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">優先度</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">入隊時間</th>
                  <th className="text-left px-4 py-3 text-gray-600 font-medium">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {waiting.map((w) => (
                  <tr key={w.req_no} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-gray-500 font-mono text-base font-bold">
                      #{w.position}
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        to={`/submissions/${w.req_no}`}
                        className="text-indigo-600 hover:underline font-mono"
                      >
                        {w.req_no}
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`text-xs font-medium px-2 py-0.5 rounded border ${
                          PRIORITY_COLORS[w.priority] ?? 'bg-gray-100 text-gray-600 border-gray-200'
                        }`}
                      >
                        {w.priority}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">
                      {formatTime(w.enqueued_at)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <select
                          className="border rounded px-2 py-0.5 text-xs bg-white"
                          defaultValue={w.priority}
                          onChange={(e) => {
                            const newPriority = e.target.value
                            if (newPriority !== w.priority) {
                              updatePriorityMutation.mutate({ req_no: w.req_no, priority: newPriority })
                            }
                          }}
                        >
                          {PRIORITY_OPTIONS.map((p) => (
                            <option key={p} value={p}>{p}</option>
                          ))}
                        </select>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
