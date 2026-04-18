import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { registryApi, submissionsApi, type ModelVersion } from '../api/client'

export default function AcceptanceQueuePage() {
  const qc = useQueryClient()
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [error, setError] = useState('')
  const [batchNote, setBatchNote] = useState('')

  const { data: pending, isLoading } = useQuery({
    queryKey: ['registry', 'pending_acceptance'],
    queryFn: () => registryApi.list({ status: 'pending_acceptance' }),
  })

  // 抓對應 submission（target 值）
  const { data: allSubs } = useQuery({
    queryKey: ['submissions-all'],
    queryFn: () => submissionsApi.list(),
  })

  const targetByReq = useMemo(() => {
    const map: Record<string, { map50_target: number | null; req_name: string | null }> = {}
    if (allSubs) {
      allSubs.forEach((s) => {
        map[s.req_no] = { map50_target: s.map50_target, req_name: s.req_name }
      })
    }
    return map
  }, [allSubs])

  const acceptMut = useMutation({
    mutationFn: ({ v, note }: { v: ModelVersion; note: string }) =>
      registryApi.accept(v.id, {
        map50_actual: v.map50_actual ?? v.map50 ?? 0,
        map50_95_actual: v.map50_95_actual ?? v.map50_95 ?? undefined,
        acceptance_note: note || undefined,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['registry', 'pending_acceptance'] })
      setError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '驗收失敗'
      setError(String(msg))
    },
  })

  const toggle = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const canAccept = (v: ModelVersion) => {
    return (v.map50_actual ?? v.map50) != null
  }

  const batchAccept = async () => {
    if (!pending) return
    const target = pending.filter((v) => selected.has(v.id) && canAccept(v))
    // 平行執行
    await Promise.all(target.map((v) => acceptMut.mutateAsync({ v, note: batchNote })))
    setSelected(new Set())
  }

  const toggleAll = () => {
    if (!pending) return
    const eligible = pending.filter(canAccept)
    if (selected.size === eligible.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(eligible.map((v) => v.id)))
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">驗收佇列</h1>
        <Link to="/" className="text-sm text-gray-500 hover:text-gray-700">
          ← 返回列表
        </Link>
      </div>

      {/* 批次備注輸入 */}
      <div className="mb-4">
        <textarea
          rows={2}
          className="w-full border rounded p-2 text-sm"
          placeholder="批次驗收備注（可選）"
          value={batchNote}
          onChange={(e) => setBatchNote(e.target.value)}
        />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-3 py-2 mb-3">
          {error}
        </div>
      )}

      {isLoading && <p className="text-gray-500">載入中...</p>}

      {pending && pending.length === 0 && (
        <div className="bg-white rounded shadow p-8 text-center text-gray-400">
          目前沒有待驗收的模型版本
        </div>
      )}

      {pending && pending.length > 0 && (
        <div className="bg-white rounded shadow">
          <div className="flex items-center justify-between p-4 border-b">
            <span className="text-sm text-gray-600">
              共 {pending.length} 個待驗收（選中 {selected.size}）
            </span>
            <div className="flex gap-2">
              <button
                onClick={toggleAll}
                className="px-3 py-1.5 text-xs border rounded hover:bg-gray-50"
              >
                全選 / 取消
              </button>
              <button
                disabled={selected.size === 0 || acceptMut.isPending}
                onClick={batchAccept}
                className="px-4 py-1.5 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
              >
                {acceptMut.isPending ? '處理中...' : `批次通過 ${selected.size}`}
              </button>
            </div>
          </div>

          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b text-xs">
              <tr>
                <th className="px-3 py-2"></th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">需求單</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">版本</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">mAP50 實 / 目標</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">訓練日</th>
                <th className="text-left px-3 py-2 text-gray-500 font-medium">動作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {pending.map((v) => {
                const actual = v.map50_actual ?? v.map50
                const target = targetByReq[v.req_no]?.map50_target
                const reqName = targetByReq[v.req_no]?.req_name
                const pass = actual != null && target != null && actual >= target
                const eligible = canAccept(v)
                return (
                  <tr key={v.id} className={!eligible ? 'bg-yellow-50' : 'hover:bg-gray-50'}>
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        disabled={!eligible}
                        checked={selected.has(v.id)}
                        onChange={() => toggle(v.id)}
                      />
                    </td>
                    <td className="px-3 py-2">
                      <Link
                        to={`/submissions/${v.req_no}`}
                        className="text-indigo-600 font-mono hover:underline"
                      >
                        {v.req_no}
                      </Link>
                      {reqName && (
                        <div className="text-xs text-gray-500">{reqName}</div>
                      )}
                    </td>
                    <td className="px-3 py-2 font-mono">{v.version}</td>
                    <td className="px-3 py-2">
                      {actual != null ? (
                        <span className={pass ? 'text-green-600 font-semibold' : 'text-orange-600'}>
                          {actual.toFixed(3)}
                          {target != null && (
                            <span className="text-gray-400 text-xs ml-1">
                              / {target.toFixed(3)}
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="text-xs text-yellow-700">
                          (未抓到 — 需手動驗收)
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-gray-500">{v.train_date || '-'}</td>
                    <td className="px-3 py-2">
                      <Link
                        to={`/registry/${v.id}/accept`}
                        className="text-indigo-600 text-xs hover:underline"
                      >
                        手動驗收 →
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
