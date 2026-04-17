import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { predictApi } from '../api/client'
import type { PredictResult } from '../api/client'

export default function PredictPage() {
  const [selectedModel, setSelectedModel] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: models = [], isLoading: modelsLoading, isError: modelsError } = useQuery({
    queryKey: ['predict-available'],
    queryFn: () => predictApi.available(),
    retry: 1,
  })

  function handleFile(file: File) {
    if (!file.type.startsWith('image/')) {
      setError('請上傳圖片檔案（JPG、PNG、BMP 等）')
      return
    }
    setImageFile(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  async function handlePredict() {
    if (!selectedModel) { setError('請先選擇模型'); return }
    if (!imageFile) { setError('請先上傳圖片'); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await predictApi.predict(selectedModel, imageFile)
      setResult(res)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })
        ?.response?.data?.detail ?? (e as { message?: string })?.message ?? '推論失敗'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  // 計算 bar chart 最大值
  const maxScore = result
    ? Math.max(...Object.values(result.all_scores))
    : 1

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">推論測試</h1>

      {/* 模型選擇 */}
      <div className="bg-white rounded-lg shadow p-5 mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">選擇模型</label>
        {modelsLoading && <p className="text-sm text-gray-500">載入可用模型中...</p>}
        {modelsError && (
          <p className="text-sm text-red-500">
            無法連線至推論 server（:8951）。請確認 inference server 正在運行。
          </p>
        )}
        {!modelsLoading && !modelsError && (
          <select
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            value={selectedModel}
            onChange={(e) => { setSelectedModel(e.target.value); setResult(null) }}
          >
            <option value="">-- 選擇模型 --</option>
            {models.map((m) => (
              <option key={m.req_no} value={m.req_no}>
                {m.req_no} — {m.name}
                {m.accuracy != null ? ` (acc ${(m.accuracy * 100).toFixed(1)}%)` : ''}
              </option>
            ))}
          </select>
        )}
        {selectedModel && models.length > 0 && (() => {
          const m = models.find((x) => x.req_no === selectedModel)
          return m ? (
            <div className="mt-2 text-xs text-gray-500">
              類別（{m.classes.length}）：{m.classes.join('、')}
            </div>
          ) : null
        })()}
      </div>

      {/* 上傳圖片 */}
      <div className="bg-white rounded-lg shadow p-5 mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">上傳圖片</label>
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isDragging
              ? 'border-indigo-500 bg-indigo-50'
              : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
          }`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {imagePreview ? (
            <div className="space-y-2">
              <img
                src={imagePreview}
                alt="預覽"
                className="max-h-48 mx-auto rounded shadow"
              />
              <p className="text-xs text-gray-500">{imageFile?.name}</p>
              <p className="text-xs text-indigo-500">點擊或拖放以更換圖片</p>
            </div>
          ) : (
            <div className="text-gray-400 space-y-1">
              <div className="text-4xl">+</div>
              <p className="text-sm">拖放圖片至此，或點擊選擇檔案</p>
              <p className="text-xs">支援 JPG、PNG、BMP、TIFF</p>
            </div>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
        />
      </div>

      {/* 送出按鈕 */}
      <button
        className={`w-full py-3 rounded-lg font-semibold text-white transition-colors mb-4 ${
          loading || !selectedModel || !imageFile
            ? 'bg-gray-300 cursor-not-allowed'
            : 'bg-indigo-600 hover:bg-indigo-700'
        }`}
        onClick={handlePredict}
        disabled={loading || !selectedModel || !imageFile}
      >
        {loading ? '推論中...' : '送出推論'}
      </button>

      {/* 錯誤訊息 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 mb-4 text-sm">
          {error}
        </div>
      )}

      {/* 結果 */}
      {result && (
        <div className="bg-white rounded-lg shadow p-5">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">推論結果</h2>
          <div className="grid grid-cols-2 gap-4 mb-5">
            <div className="bg-indigo-50 rounded-lg p-4 text-center">
              <div className="text-xs text-indigo-500 font-medium mb-1">預測類別</div>
              <div className="text-xl font-bold text-indigo-800 break-all">{result.prediction}</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 text-center">
              <div className="text-xs text-green-600 font-medium mb-1">信心度</div>
              <div className="text-xl font-bold text-green-700">
                {(result.confidence * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* 分數 Bar Chart */}
          <div>
            <h3 className="text-sm font-medium text-gray-600 mb-2">各類別分數</h3>
            <div className="space-y-1">
              {Object.entries(result.all_scores)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, score]) => (
                  <div key={cls} className="flex items-center gap-2">
                    <div className="w-28 text-xs text-right text-gray-600 truncate flex-shrink-0" title={cls}>
                      {cls}
                    </div>
                    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                      <div
                        className={`h-4 rounded-full transition-all ${
                          cls === result.prediction
                            ? 'bg-indigo-500'
                            : 'bg-gray-300'
                        }`}
                        style={{ width: `${(score / maxScore) * 100}%` }}
                      />
                    </div>
                    <div className="w-14 text-xs text-gray-500 text-right flex-shrink-0">
                      {(score * 100).toFixed(2)}%
                    </div>
                  </div>
                ))}
            </div>
          </div>

          <div className="mt-4 text-xs text-gray-400">
            模型：{result.req_no} ({result.model})
            {result.accuracy != null && (
              <span> · 訓練準確率 {(result.accuracy * 100).toFixed(1)}%</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
