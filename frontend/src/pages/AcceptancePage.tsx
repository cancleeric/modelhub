import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { registryApi, submissionsApi } from '../api/client'

export default function AcceptancePage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [form, setForm] = useState({
    map50_actual: '',
    map50_95_actual: '',
    acceptance_note: '',
    accepted_by: '',
  })
  const [error, setError] = useState('')

  const { data: version } = useQuery({
    queryKey: ['registry', 'item', id],
    queryFn: () => registryApi.get(Number(id)),
    enabled: !!id,
  })

  const { data: submission } = useQuery({
    queryKey: ['submission', version?.req_no],
    queryFn: () => submissionsApi.get(version!.req_no),
    enabled: !!version?.req_no,
  })

  const acceptMut = useMutation({
    mutationFn: () =>
      registryApi.accept(Number(id), {
        map50_actual: parseFloat(form.map50_actual),
        map50_95_actual: form.map50_95_actual
          ? parseFloat(form.map50_95_actual)
          : undefined,
        acceptance_note: form.acceptance_note || undefined,
        accepted_by: form.accepted_by || undefined,
      }),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['registry'] })
      qc.invalidateQueries({ queryKey: ['submission', result.req_no] })
      navigate(`/submissions/${result.req_no}`)
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '驗收失敗'
      setError(String(msg))
    },
  })

  const map50Target = submission?.map50_target
  const map50Actual = form.map50_actual ? parseFloat(form.map50_actual) : null
  const predictedResult =
    map50Target != null && map50Actual != null
      ? map50Actual >= map50Target
        ? 'PASS'
        : 'FAIL'
      : null

  if (!version) return <p className="text-gray-500">載入中...</p>

  return (
    <div className="max-w-lg">
      <div className="flex items-center gap-2 mb-6">
        <Link
          to={`/submissions/${version.req_no}`}
          className="text-gray-400 hover:text-gray-600 text-sm"
        >
          ← 返回需求單
        </Link>
      </div>

      <h1 className="text-2xl font-bold text-gray-900 mb-1">模型驗收</h1>
      <p className="text-sm text-gray-500 mb-6">
        {version.req_no} / {version.model_name} {version.version}
      </p>

      {map50Target != null && (
        <div className="bg-blue-50 border border-blue-200 rounded px-4 py-2 text-sm text-blue-700 mb-4">
          mAP50 目標值：<strong>{map50Target}</strong>
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault()
          acceptMut.mutate()
        }}
        className="bg-white rounded shadow p-6 space-y-4"
      >
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded px-4 py-2 text-sm">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            mAP50 實測值 *
          </label>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            required
            className="w-full border rounded px-3 py-2 text-sm"
            value={form.map50_actual}
            onChange={(e) => setForm((p) => ({ ...p, map50_actual: e.target.value }))}
            placeholder="0.00"
          />
          {predictedResult && (
            <p
              className={`mt-1 text-sm font-bold ${
                predictedResult === 'PASS' ? 'text-green-600' : 'text-red-600'
              }`}
            >
              預測結果：{predictedResult}
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            mAP50-95 實測值
          </label>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            className="w-full border rounded px-3 py-2 text-sm"
            value={form.map50_95_actual}
            onChange={(e) => setForm((p) => ({ ...p, map50_95_actual: e.target.value }))}
            placeholder="可選"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">驗收人</label>
          <input
            className="w-full border rounded px-3 py-2 text-sm"
            value={form.accepted_by}
            onChange={(e) => setForm((p) => ({ ...p, accepted_by: e.target.value }))}
            placeholder="驗收人名稱"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">驗收說明</label>
          <textarea
            rows={3}
            className="w-full border rounded px-3 py-2 text-sm"
            value={form.acceptance_note}
            onChange={(e) => setForm((p) => ({ ...p, acceptance_note: e.target.value }))}
            placeholder="可選備注"
          />
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-50"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={acceptMut.isPending}
            className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {acceptMut.isPending ? '提交中...' : '確認驗收'}
          </button>
        </div>
      </form>
    </div>
  )
}
