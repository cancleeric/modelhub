import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { adminApi, type ApiKeyFull } from '../api/client'

export default function ApiKeyPage() {
  const qc = useQueryClient()
  const [name, setName] = useState('')
  const [newKey, setNewKey] = useState<ApiKeyFull | null>(null)
  const [error, setError] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['admin', 'api-keys'],
    queryFn: () => adminApi.listApiKeys(),
  })

  const createMut = useMutation({
    mutationFn: (n: string) => adminApi.createApiKey(n),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['admin', 'api-keys'] })
      setNewKey(res)
      setName('')
      setError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '建立失敗'
      setError(String(msg))
    },
  })

  const toggleMut = useMutation({
    mutationFn: ({ id, disabled }: { id: number; disabled: boolean }) =>
      disabled ? adminApi.enableApiKey(id) : adminApi.disableApiKey(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin', 'api-keys'] }),
  })

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">API Key 管理</h1>

      <div className="bg-white rounded shadow p-5 mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">建立新 Key</h3>
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-3 py-2 mb-2">
            {error}
          </div>
        )}
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="用途標籤（例：AICAD pipeline）"
            className="flex-1 border rounded px-3 py-2 text-sm"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button
            disabled={createMut.isPending || !name.trim()}
            onClick={() => createMut.mutate(name.trim())}
            className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {createMut.isPending ? '建立中...' : '建立 Key'}
          </button>
        </div>
        {newKey && (
          <div className="mt-3 bg-yellow-50 border border-yellow-300 rounded p-3 text-sm">
            <div className="font-semibold text-yellow-800 mb-1">
              ⚠️ 請立即複製保存，關閉頁面後無法再取得完整 key：
            </div>
            <code className="block bg-white border rounded p-2 font-mono text-xs break-all">
              {newKey.key}
            </code>
            <button
              onClick={() => setNewKey(null)}
              className="mt-2 text-xs text-yellow-700 underline"
            >
              我已複製，隱藏
            </button>
          </div>
        )}
      </div>

      <div className="bg-white rounded shadow overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">ID</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">名稱</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">Key</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">建立者</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">最後使用</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">狀態</th>
              <th className="text-left px-4 py-3 text-gray-600 font-medium">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {isLoading && (
              <tr>
                <td colSpan={7} className="text-center py-8 text-gray-400">載入中...</td>
              </tr>
            )}
            {data && data.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-8 text-gray-400">
                  尚未建立任何 API Key，使用 bootstrap key 中。
                </td>
              </tr>
            )}
            {data?.map((k) => (
              <tr key={k.id} className={k.disabled ? 'bg-gray-50 text-gray-400' : 'hover:bg-gray-50'}>
                <td className="px-4 py-3 font-mono text-xs">{k.id}</td>
                <td className="px-4 py-3">{k.name}</td>
                <td className="px-4 py-3 font-mono text-xs">{k.key_preview}</td>
                <td className="px-4 py-3 text-xs text-gray-500">{k.created_by || '-'}</td>
                <td className="px-4 py-3 text-xs text-gray-500">
                  {k.last_used_at ? new Date(k.last_used_at).toLocaleString('zh-TW') : '從未使用'}
                </td>
                <td className="px-4 py-3">
                  {k.disabled ? (
                    <span className="text-xs bg-gray-200 px-2 py-0.5 rounded">已停用</span>
                  ) : (
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">啟用中</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <button
                    disabled={toggleMut.isPending}
                    onClick={() => toggleMut.mutate({ id: k.id, disabled: k.disabled })}
                    className="text-xs text-indigo-600 hover:underline disabled:opacity-50"
                  >
                    {k.disabled ? '啟用' : '停用'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
