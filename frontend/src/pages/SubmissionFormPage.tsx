import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { submissionsApi } from '../api/client'

const PRODUCTS = ['AICAD', '打工仔', '天機', 'RS Conch', '其他']

export default function SubmissionFormPage() {
  const navigate = useNavigate()
  const [form, setForm] = useState({
    req_name: '',
    product: 'AICAD',
    company: '',
    submitter: '',
    purpose: '',
    class_list: '',
    map50_target: '',
    dataset_source: '',
    dataset_count: '',
    label_format: '',
    expected_delivery: '',
    priority: 'P2',
    model_type: 'detection',
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const payload = {
        ...form,
        map50_target: form.map50_target ? parseFloat(form.map50_target) : undefined,
      }
      const created = await submissionsApi.create(payload)
      navigate(`/submissions/${created.req_no}`)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '提交失敗，請稍後再試'
      setError(String(msg))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">新增訓練需求單</h1>

      <form onSubmit={handleSubmit} className="bg-white rounded shadow p-6 space-y-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded px-4 py-2 text-sm">
            {error}
          </div>
        )}

        <Field label="需求名稱 *" required>
          <input
            name="req_name"
            value={form.req_name}
            onChange={handleChange}
            required
            className="input"
            placeholder="例：P&ID 管線辨識 v3"
          />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="產品 *" required>
            <select name="product" value={form.product} onChange={handleChange} className="input">
              {PRODUCTS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </Field>
          <Field label="優先度">
            <select name="priority" value={form.priority} onChange={handleChange} className="input">
              {['P0', 'P1', 'P2', 'P3'].map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="業務公司 *" required>
            <input
              name="company"
              value={form.company}
              onChange={handleChange}
              required
              className="input"
              placeholder="HurricaneEdge"
            />
          </Field>
          <Field label="提交人">
            <input
              name="submitter"
              value={form.submitter}
              onChange={handleChange}
              className="input"
              placeholder="聯絡人名稱或 email"
            />
          </Field>
        </div>

        <Field label="業務問題描述 *" required>
          <textarea
            name="purpose"
            value={form.purpose}
            onChange={handleChange}
            required
            rows={3}
            className="input"
            placeholder="描述此模型要解決的業務問題"
          />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="模型類型">
            <select name="model_type" value={form.model_type} onChange={handleChange} className="input">
              <option value="detection">detection</option>
              <option value="classification">classification</option>
              <option value="segmentation">segmentation</option>
            </select>
          </Field>
          <Field label="mAP50 目標 *" required>
            <input
              name="map50_target"
              value={form.map50_target}
              onChange={handleChange}
              required
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="input"
              placeholder="0.90"
            />
          </Field>
        </div>

        <Field label="類別清單 *" required>
          <textarea
            name="class_list"
            value={form.class_list}
            onChange={handleChange}
            required
            rows={3}
            className="input"
            placeholder="每行一個類別，或逗號分隔"
          />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="資料來源 *" required>
            <input
              name="dataset_source"
              value={form.dataset_source}
              onChange={handleChange}
              required
              className="input"
              placeholder="Kaggle / 自建 / HurricaneEdge 提供"
            />
          </Field>
          <Field label="資料集大小描述 *" required>
            <input
              name="dataset_count"
              value={form.dataset_count}
              onChange={handleChange}
              required
              className="input"
              placeholder="例：約 5000 張"
            />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="標注格式 *" required>
            <input
              name="label_format"
              value={form.label_format}
              onChange={handleChange}
              required
              className="input"
              placeholder="YOLO / COCO / VOC"
            />
          </Field>
          <Field label="預計交付日期 *" required>
            <input
              name="expected_delivery"
              value={form.expected_delivery}
              onChange={handleChange}
              required
              className="input"
              placeholder="2026-05-01"
            />
          </Field>
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-50"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {submitting ? '提交中...' : '儲存草稿'}
          </button>
        </div>
      </form>

      <style>{`
        .input {
          width: 100%;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 14px;
          outline: none;
          box-sizing: border-box;
        }
        .input:focus { border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,0.15); }
      `}</style>
    </div>
  )
}

function Field({
  label,
  required,
  children,
}: {
  label: string
  required?: boolean
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-400 ml-0.5">*</span>}
      </label>
      {children}
    </div>
  )
}
