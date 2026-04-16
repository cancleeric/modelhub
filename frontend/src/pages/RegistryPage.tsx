import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { registryApi } from '../api/client'
import StatusBadge from '../components/StatusBadge'

export default function RegistryPage() {
  const [productFilter, setProductFilter] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['registry', productFilter],
    queryFn: () => registryApi.list({ product: productFilter || undefined }),
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">模型版本清冊</h1>
      </div>

      <div className="flex gap-3 mb-4">
        <input
          type="text"
          placeholder="Product 篩選..."
          className="border rounded px-3 py-1.5 text-sm"
          value={productFilter}
          onChange={(e) => setProductFilter(e.target.value)}
        />
      </div>

      {isLoading && <p className="text-gray-500">載入中...</p>}

      {data && (
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
              {data.map((v) => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link
                      to={`/submissions/${v.req_no}`}
                      className="text-indigo-600 hover:underline font-mono text-xs"
                    >
                      {v.req_no}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-700">{v.product}</td>
                  <td className="px-4 py-3 text-gray-800 font-medium">{v.model_name}</td>
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
                  <td className="px-4 py-3">
                    <Link
                      to={`/registry/${v.id}/accept`}
                      className="text-indigo-600 text-xs hover:underline"
                    >
                      驗收
                    </Link>
                  </td>
                </tr>
              ))}
              {data.length === 0 && (
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
