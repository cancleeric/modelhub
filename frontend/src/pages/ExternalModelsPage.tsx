/**
 * ExternalModelsPage.tsx — M22 外部模型登記清單頁
 *
 * 顯示所有已登記的外部 pretrained model（status=registered）。
 * 供內部人員查閱各模型的路徑、版本、最後使用時間。
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { externalModelsApi, type ExternalModelOut } from '../api/client'

function formatBytes(bytes: number | null): string {
  if (bytes == null) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function formatDate(iso: string | null): string {
  if (!iso) return '-'
  return new Date(iso).toLocaleString('zh-TW', { timeZone: 'Asia/Taipei' })
}

function SourceBadge({ source }: { source: string | null }) {
  if (!source) return <span className="text-gray-300">-</span>
  const label = source.startsWith('huggingface://') ? 'HuggingFace' :
    source.startsWith('s3://') ? 'S3' :
    source.startsWith('local://') ? 'Local' : 'External'
  const colors: Record<string, string> = {
    HuggingFace: 'bg-orange-100 text-orange-700',
    S3: 'bg-green-100 text-green-700',
    Local: 'bg-gray-100 text-gray-600',
    External: 'bg-blue-100 text-blue-700',
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${colors[label] ?? colors.External}`}>
      {label}
    </span>
  )
}

function ModelRow({ model }: { model: ExternalModelOut }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <>
      <tr
        className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-4 py-3 text-sm font-mono text-gray-600">{model.req_no}</td>
        <td className="px-4 py-3">
          <div className="font-medium text-sm text-gray-800">{model.model_name}</div>
          <div className="text-xs text-gray-400">{model.product} · {model.version}</div>
        </td>
        <td className="px-4 py-3">
          <SourceBadge source={model.external_source} />
        </td>
        <td className="px-4 py-3 text-sm text-gray-500">{formatBytes(model.size_bytes)}</td>
        <td className="px-4 py-3 text-xs text-gray-400">{formatDate(model.last_used_at)}</td>
        <td className="px-4 py-3">
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            model.status === 'registered' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
          }`}>
            {model.status}
          </span>
        </td>
        <td className="px-4 py-3 text-gray-400 text-xs">{expanded ? '▲' : '▼'}</td>
      </tr>
      {expanded && (
        <tr className="bg-gray-50">
          <td colSpan={7} className="px-6 py-3">
            <div className="space-y-2 text-xs text-gray-600">
              {model.file_path && (
                <div>
                  <span className="font-semibold text-gray-500">本機路徑：</span>
                  <code className="ml-2 bg-gray-100 px-2 py-0.5 rounded font-mono break-all">
                    {model.file_path}
                  </code>
                </div>
              )}
              {model.external_source && (
                <div>
                  <span className="font-semibold text-gray-500">來源：</span>
                  <code className="ml-2 bg-gray-100 px-2 py-0.5 rounded font-mono break-all">
                    {model.external_source}
                  </code>
                </div>
              )}
              {model.external_sha256 && (
                <div>
                  <span className="font-semibold text-gray-500">SHA256：</span>
                  <code className="ml-2 bg-gray-100 px-2 py-0.5 rounded font-mono">
                    {model.external_sha256}
                  </code>
                </div>
              )}
              {model.arch && (
                <div>
                  <span className="font-semibold text-gray-500">架構：</span>
                  <span className="ml-2">{model.arch}</span>
                </div>
              )}
              {model.notes && (
                <div>
                  <span className="font-semibold text-gray-500">備注：</span>
                  <span className="ml-2">{model.notes}</span>
                </div>
              )}
              <div>
                <span className="font-semibold text-gray-500">登記時間：</span>
                <span className="ml-2">{formatDate(model.created_at)}</span>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function ExternalModelsPage() {
  const [productFilter, setProductFilter] = useState('')

  const { data: models, isLoading, isError, refetch } = useQuery({
    queryKey: ['external-models', productFilter],
    queryFn: () => externalModelsApi.list(productFilter || undefined),
    staleTime: 60_000,
  })

  const products = [...new Set((models ?? []).map((m) => m.product))].sort()

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-gray-800">外部模型登記</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            外部 pretrained model（非訓練產出）的版本登記清單
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 text-gray-600"
        >
          重新整理
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4">
        <select
          value={productFilter}
          onChange={(e) => setProductFilter(e.target.value)}
          className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          <option value="">全部產品</option>
          {products.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        {productFilter && (
          <button
            onClick={() => setProductFilter('')}
            className="text-sm text-gray-400 hover:text-gray-600"
          >
            清除篩選
          </button>
        )}
      </div>

      {isLoading && (
        <div className="text-center py-12 text-gray-400">載入中...</div>
      )}

      {isError && (
        <div className="text-center py-12 text-red-500">無法載入外部模型清單</div>
      )}

      {!isLoading && !isError && (
        <>
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">工單</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">模型</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">來源</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">大小</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">最後使用</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">狀態</th>
                  <th className="px-4 py-3 w-8"></th>
                </tr>
              </thead>
              <tbody>
                {(models ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-12 text-center text-gray-400 text-sm">
                      沒有外部模型登記記錄
                    </td>
                  </tr>
                ) : (
                  (models ?? []).map((model) => (
                    <ModelRow key={model.id} model={model} />
                  ))
                )}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            共 {(models ?? []).length} 筆記錄 · 點擊行展開詳細資訊
          </p>
        </>
      )}
    </div>
  )
}
