import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { submissionsApi } from '../api/client'

const PRODUCTS = ['AICAD', '打工仔', '天機', 'RS Conch', '其他']

type ModelTemplateKey = 'detection' | 'classification' | 'segmentation' | 'ocr'

interface ModelTemplate {
  arch: string
  map50_threshold: number
  map50_95_target: number | null
  inference_latency_ms: number
  model_size_limit_mb: number
  label_format: string
  input_spec: string
}

const MODEL_TEMPLATES: Record<ModelTemplateKey, ModelTemplate> = {
  detection: {
    arch: 'YOLOv8m',
    map50_threshold: 0.85,
    map50_95_target: 0.65,
    inference_latency_ms: 200,
    model_size_limit_mb: 100,
    label_format: 'YOLO',
    input_spec: '640x640',
  },
  classification: {
    arch: 'MobileNetV2',
    map50_threshold: 0.90,
    map50_95_target: null,
    inference_latency_ms: 50,
    model_size_limit_mb: 50,
    label_format: 'ImageFolder',
    input_spec: '224x224',
  },
  segmentation: {
    arch: 'UNet-ResNet18',
    map50_threshold: 0.80,
    map50_95_target: 0.60,
    inference_latency_ms: 150,
    model_size_limit_mb: 100,
    label_format: 'COCO',
    input_spec: '512x512',
  },
  ocr: {
    arch: 'TrOCR-small',
    map50_threshold: 0.95,
    map50_95_target: null,
    inference_latency_ms: 50,
    model_size_limit_mb: 250,
    label_format: 'text_label',
    input_spec: '32x128',
  },
}

const MODEL_TYPE_OPTIONS: ModelTemplateKey[] = ['detection', 'classification', 'segmentation', 'ocr']

const EMPTY_FORM = {
  req_name: '',
  product: 'AICAD',
  company: '',
  submitter: '',
  purpose: '',
  class_list: '',
  map50_target: '',
  map50_threshold: '',
  map50_95_target: '',
  inference_latency_ms: '',
  model_size_limit_mb: '',
  arch: '',
  input_spec: '',
  dataset_source: '',
  dataset_count: '',
  dataset_train_count: '',
  dataset_val_count: '',
  dataset_test_count: '',
  label_format: '',
  expected_delivery: '',
  priority: 'P2',
  model_type: 'detection',
  kaggle_dataset_url: '',
  max_budget_usd: '5.0',
  max_retries: '2',
}

export default function SubmissionFormPage() {
  const navigate = useNavigate()
  // P1-3: 支援 /submit/:req_no 路由，載入時 fetch 並 pre-fill
  const { req_no } = useParams<{ req_no?: string }>()
  const [appliedTemplate, setAppliedTemplate] = useState<ModelTemplateKey | null>(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [draftReqNo, setDraftReqNo] = useState<string | null>(req_no ?? null)
  const [submitting, setSubmitting] = useState(false)
  const [savingDraft, setSavingDraft] = useState(false)
  const [error, setError] = useState('')
  const [warnings, setWarnings] = useState<string[]>([])
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [draftSaved, setDraftSaved] = useState(false)

  // P1-3: 若路由帶 req_no，載入草稿並 pre-fill
  useEffect(() => {
    if (!req_no) return
    submissionsApi.get(req_no).then((sub) => {
      if (sub.status !== 'draft') return
      setDraftReqNo(sub.req_no)
      setForm({
        req_name: sub.req_name ?? '',
        product: sub.product ?? 'AICAD',
        company: sub.company ?? '',
        submitter: sub.submitter ?? '',
        purpose: sub.purpose ?? '',
        class_list: sub.class_list ?? '',
        map50_target: sub.map50_target?.toString() ?? '',
        map50_threshold: sub.map50_threshold?.toString() ?? '',
        map50_95_target: sub.map50_95_target?.toString() ?? '',
        inference_latency_ms: sub.inference_latency_ms?.toString() ?? '',
        model_size_limit_mb: sub.model_size_limit_mb?.toString() ?? '',
        arch: sub.arch ?? '',
        input_spec: sub.input_spec ?? '',
        dataset_source: sub.dataset_source ?? '',
        dataset_count: sub.dataset_count ?? '',
        dataset_train_count: '',
        dataset_val_count: sub.dataset_val_count?.toString() ?? '',
        dataset_test_count: sub.dataset_test_count?.toString() ?? '',
        label_format: sub.label_format ?? '',
        expected_delivery: sub.expected_delivery ?? '',
        priority: sub.priority ?? 'P2',
        model_type: sub.model_type ?? 'detection',
        kaggle_dataset_url: sub.kaggle_dataset_url ?? '',
        max_budget_usd: sub.max_budget_usd?.toString() ?? '5.0',
        max_retries: sub.max_retries?.toString() ?? '2',
      })
    }).catch(() => {})
  }, [req_no])

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const handleTemplateApply = (type: ModelTemplateKey) => {
    const tpl = MODEL_TEMPLATES[type]
    setForm((prev) => ({
      ...prev,
      model_type: type,
      arch: tpl.arch,
      map50_threshold: String(tpl.map50_threshold),
      map50_95_target: tpl.map50_95_target != null ? String(tpl.map50_95_target) : '',
      inference_latency_ms: String(tpl.inference_latency_ms),
      model_size_limit_mb: String(tpl.model_size_limit_mb),
      label_format: tpl.label_format,
      input_spec: tpl.input_spec,
    }))
    setAppliedTemplate(type)
  }

  const buildPayload = () => ({
    ...form,
    map50_target: form.map50_target ? parseFloat(form.map50_target) : undefined,
    map50_threshold: form.map50_threshold ? parseFloat(form.map50_threshold) : undefined,
    map50_95_target: form.map50_95_target ? parseFloat(form.map50_95_target) : undefined,
    inference_latency_ms: form.inference_latency_ms ? parseInt(form.inference_latency_ms) : undefined,
    model_size_limit_mb: form.model_size_limit_mb ? parseInt(form.model_size_limit_mb) : undefined,
    dataset_train_count: form.dataset_train_count ? parseInt(form.dataset_train_count) : undefined,
    dataset_val_count: form.dataset_val_count ? parseInt(form.dataset_val_count) : undefined,
    dataset_test_count: form.dataset_test_count ? parseInt(form.dataset_test_count) : undefined,
    max_budget_usd: form.max_budget_usd ? parseFloat(form.max_budget_usd) : undefined,
    max_retries: form.max_retries ? parseInt(form.max_retries) : undefined,
  })

  // P1-3: 儲存草稿 — POST 建立 draft，不跳轉，更新 URL
  const handleSaveDraft = async () => {
    setSavingDraft(true)
    setError('')
    try {
      const payload = buildPayload()
      if (draftReqNo) {
        // 已有草稿：PATCH 更新
        await submissionsApi.update(draftReqNo, payload)
      } else {
        // 新草稿：POST 建立 status=draft
        const result = await submissionsApi.create(payload)
        const newReqNo = result.submission.req_no
        setDraftReqNo(newReqNo)
        // 不 navigate，只更新 URL
        window.history.replaceState(null, '', `/submit/${newReqNo}`)
      }
      setDraftSaved(true)
      setTimeout(() => setDraftSaved(false), 3000)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '儲存草稿失敗'
      setError(String(msg))
    } finally {
      setSavingDraft(false)
    }
  }

  // P1-3: 繼續編輯後送審（呼叫 submit action）
  const handleSubmitDraft = async () => {
    if (!draftReqNo) {
      setError('請先儲存草稿')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      // 先 PATCH 更新欄位
      await submissionsApi.update(draftReqNo, buildPayload())
      // 再呼叫 submit action
      await submissionsApi.action(draftReqNo, 'submit')
      navigate(`/submissions/${draftReqNo}`)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '提交失敗，請稍後再試'
      setError(String(msg))
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // 若已有草稿 req_no，走 handleSubmitDraft
    if (draftReqNo) {
      await handleSubmitDraft()
      return
    }
    setSubmitting(true)
    setError('')
    setWarnings([])
    try {
      const payload = buildPayload()
      const result = await submissionsApi.create(payload)
      const hasWarnings = result.warnings && result.warnings.length > 0
      const hasSuggestions = result.suggestions && result.suggestions.length > 0
      if (hasWarnings || hasSuggestions) {
        setWarnings(result.warnings ?? [])
        setSuggestions(result.suggestions ?? [])
        setTimeout(() => navigate(`/submissions/${result.submission.req_no}`), 4500)
      } else {
        navigate(`/submissions/${result.submission.req_no}`)
      }
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
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">
          {draftReqNo ? `草稿 ${draftReqNo}` : '新增訓練需求單'}
        </h1>
        {draftSaved && (
          <span className="text-sm text-green-600 font-medium">草稿已儲存</span>
        )}
      </div>

      <form onSubmit={handleSubmit} className="bg-white rounded shadow p-6 space-y-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded px-4 py-2 text-sm">
            {error}
          </div>
        )}
        {warnings.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-300 text-yellow-800 rounded px-4 py-2 text-sm">
            <div className="font-medium mb-1">送出成功，但有以下提醒：</div>
            <ul className="list-disc list-inside space-y-0.5">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        {suggestions.length > 0 && (
          <div className="bg-blue-50 border border-blue-300 text-blue-800 rounded px-4 py-2 text-sm">
            <div className="font-medium mb-1">智能建議（Anemone LLM）：</div>
            <ul className="list-disc list-inside space-y-0.5">
              {suggestions.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        )}
        {(warnings.length > 0 || suggestions.length > 0) && (
          <p className="text-xs text-gray-500">4 秒後自動跳轉詳情頁...</p>
        )}

        {/* Template Selector */}
        <div className="bg-indigo-50 border border-indigo-100 rounded p-4">
          <p className="text-sm font-medium text-indigo-800 mb-2">快速套用需求單範本</p>
          <div className="flex gap-2 flex-wrap">
            {MODEL_TYPE_OPTIONS.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => handleTemplateApply(type)}
                className={`px-3 py-1.5 text-sm rounded border transition-colors ${
                  appliedTemplate === type
                    ? 'bg-indigo-600 text-white border-indigo-600'
                    : 'bg-white text-indigo-700 border-indigo-300 hover:bg-indigo-100'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
          {appliedTemplate && (
            <p className="text-xs text-indigo-600 mt-2">
              已套用 {appliedTemplate} 範本，可依需求調整
            </p>
          )}
        </div>

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
            <select
              name="model_type"
              value={form.model_type}
              onChange={(e) => {
                handleChange(e)
                setAppliedTemplate(null)
              }}
              className="input"
            >
              {MODEL_TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </Field>
          <Field label="架構">
            <input
              name="arch"
              value={form.arch}
              onChange={handleChange}
              className="input"
              placeholder="YOLOv8m"
            />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
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
          <Field label="mAP50 門檻">
            <input
              name="map50_threshold"
              value={form.map50_threshold}
              onChange={handleChange}
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="input"
              placeholder="0.85"
            />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="mAP50-95 目標">
            <input
              name="map50_95_target"
              value={form.map50_95_target}
              onChange={handleChange}
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="input"
              placeholder="0.65"
            />
          </Field>
          <Field label="推論延遲上限 (ms)">
            <input
              name="inference_latency_ms"
              value={form.inference_latency_ms}
              onChange={handleChange}
              type="number"
              min="0"
              className="input"
              placeholder="200"
            />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="模型大小上限 (MB)">
            <input
              name="model_size_limit_mb"
              value={form.model_size_limit_mb}
              onChange={handleChange}
              type="number"
              min="0"
              className="input"
              placeholder="100"
            />
          </Field>
          <Field label="輸入規格">
            <input
              name="input_spec"
              value={form.input_spec}
              onChange={handleChange}
              className="input"
              placeholder="640x640"
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

        <div className="grid grid-cols-3 gap-4">
          <Field label="Train 數量">
            <input
              name="dataset_train_count"
              value={form.dataset_train_count}
              onChange={handleChange}
              type="number"
              min="0"
              className="input"
              placeholder="3500"
            />
          </Field>
          <Field label="Validation 數量">
            <input
              name="dataset_val_count"
              value={form.dataset_val_count}
              onChange={handleChange}
              type="number"
              min="0"
              className="input"
              placeholder="1000"
            />
          </Field>
          <Field label="Test 數量">
            <input
              name="dataset_test_count"
              value={form.dataset_test_count}
              onChange={handleChange}
              type="number"
              min="0"
              className="input"
              placeholder="500"
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

        <Field label="Kaggle dataset URL（選填）">
          <input
            name="kaggle_dataset_url"
            value={form.kaggle_dataset_url}
            onChange={handleChange}
            className="input"
            placeholder="https://www.kaggle.com/datasets/..."
          />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="預算上限 (USD)">
            <input
              name="max_budget_usd"
              value={form.max_budget_usd}
              onChange={handleChange}
              type="number"
              step="0.5"
              min="0"
              className="input"
              placeholder="5.0"
            />
          </Field>
          <Field label="自動重試次數上限">
            <input
              name="max_retries"
              value={form.max_retries}
              onChange={handleChange}
              type="number"
              min="0"
              max="5"
              className="input"
              placeholder="2"
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
            type="button"
            disabled={savingDraft}
            onClick={handleSaveDraft}
            className="px-4 py-2 text-sm border border-indigo-300 text-indigo-700 rounded hover:bg-indigo-50 disabled:opacity-50"
          >
            {savingDraft ? '儲存中...' : '儲存草稿'}
          </button>
          {draftReqNo && (
            <button
              type="button"
              disabled={submitting}
              onClick={handleSubmitDraft}
              className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
            >
              {submitting ? '送審中...' : '送出審核'}
            </button>
          )}
          {!draftReqNo && (
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
            >
              {submitting ? '提交中...' : '直接送審'}
            </button>
          )}
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
