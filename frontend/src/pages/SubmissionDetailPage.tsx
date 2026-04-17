import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { submissionsApi, registryApi, type Submission } from '../api/client'
import StatusBadge from '../components/StatusBadge'
import RejectModal from '../components/RejectModal'

const AVAILABLE_ACTIONS: Record<string, { action: string; label: string; color: string }[]> = {
  draft:    [{ action: 'submit', label: '提交審核', color: 'blue' }],
  submitted:[
    { action: 'approve', label: '核准', color: 'green' },
    // reject 走獨立 modal
  ],
  approved: [{ action: 'start_training', label: '開始訓練', color: 'yellow' }],
  training: [{ action: 'complete_training', label: '訓練完成', color: 'purple' }],
  trained:  [
    { action: 'accept', label: '驗收通過', color: 'emerald' },
    { action: 'fail',   label: '驗收失敗', color: 'red' },
  ],
  failed:   [{ action: 'retrain', label: '重新訓練', color: 'yellow' }],
}

const COLOR_MAP: Record<string, string> = {
  blue:    'bg-blue-600 hover:bg-blue-700 text-white',
  green:   'bg-green-600 hover:bg-green-700 text-white',
  red:     'bg-red-600 hover:bg-red-700 text-white',
  yellow:  'bg-yellow-500 hover:bg-yellow-600 text-white',
  purple:  'bg-purple-600 hover:bg-purple-700 text-white',
  emerald: 'bg-emerald-600 hover:bg-emerald-700 text-white',
}

const KAGGLE_STATUS_COLOR: Record<string, string> = {
  queued:   'bg-gray-100 text-gray-700',
  running:  'bg-blue-100 text-blue-700',
  complete: 'bg-green-100 text-green-700',
  error:    'bg-red-100 text-red-700',
}

type TabKey = 'info' | 'history' | 'kaggle'

export default function SubmissionDetailPage() {
  const { req_no } = useParams<{ req_no: string }>()
  const qc = useQueryClient()
  const [noteInput, setNoteInput] = useState('')
  const [actionError, setActionError] = useState('')
  const [tab, setTab] = useState<TabKey>('info')
  const [resubmitNote, setResubmitNote] = useState('')
  const [kernelSlug, setKernelSlug] = useState('')
  const [kernelVersion, setKernelVersion] = useState('')
  const [showRejectModal, setShowRejectModal] = useState(false)
  const [showEditFields, setShowEditFields] = useState(false)
  const [editFields, setEditFields] = useState({
    class_list: '',
    dataset_count: '',
    dataset_val_count: '',
    dataset_test_count: '',
    dataset_source: '',
    kaggle_dataset_url: '',
  })

  const { data: sub, isLoading } = useQuery({
    queryKey: ['submission', req_no],
    queryFn: () => submissionsApi.get(req_no!),
    enabled: !!req_no,
  })

  const { data: versions } = useQuery({
    queryKey: ['registry', 'by-req', req_no],
    queryFn: () => registryApi.byReq(req_no!),
    enabled: !!req_no,
  })

  const { data: history } = useQuery({
    queryKey: ['submission', 'history', req_no],
    queryFn: () => submissionsApi.history(req_no!),
    enabled: !!req_no && tab === 'history',
  })

  const actionMut = useMutation({
    mutationFn: ({ action, note }: { action: string; note?: string }) =>
      submissionsApi.action(req_no!, action, { note }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
      qc.invalidateQueries({ queryKey: ['submissions'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setNoteInput('')
      setActionError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '操作失敗'
      setActionError(String(msg))
    },
  })

  const rejectMut = useMutation({
    mutationFn: ({ reasons, note }: { reasons: string[]; note: string }) =>
      submissionsApi.reject(req_no!, { reasons, note }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
      qc.invalidateQueries({ queryKey: ['submissions'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setShowRejectModal(false)
      setActionError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '退件失敗'
      setActionError(String(msg))
    },
  })

  const updateFieldsMut = useMutation({
    mutationFn: (fields: Record<string, string | number | undefined>) =>
      submissionsApi.update(req_no!, fields),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '更新欄位失敗'
      setActionError(String(msg))
    },
  })

  const resubmitMut = useMutation({
    mutationFn: (note: string) => submissionsApi.resubmit(req_no!, { note }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
      qc.invalidateQueries({ queryKey: ['submission', 'history', req_no] })
      qc.invalidateQueries({ queryKey: ['submissions'] })
      setResubmitNote('')
      setActionError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '補件失敗'
      setActionError(String(msg))
    },
  })

  const attachMut = useMutation({
    mutationFn: ({ slug, version }: { slug: string; version?: number }) =>
      submissionsApi.attachKernel(req_no!, { slug, version }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
      setKernelSlug('')
      setKernelVersion('')
      setActionError('')
    },
    onError: (err: unknown) => {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        '綁定 kernel 失敗'
      setActionError(String(msg))
    },
  })

  const refreshKaggleMut = useMutation({
    mutationFn: () => submissionsApi.refreshKaggle(req_no!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['submission', req_no] })
    },
  })

  if (isLoading) return <p className="text-gray-500">載入中...</p>
  if (!sub) return <p className="text-red-500">找不到需求單 {req_no}</p>

  const actions = AVAILABLE_ACTIONS[sub.status] ?? []
  const rejectionReasons = parseReasons(sub.rejection_reasons)

  return (
    <div className="max-w-3xl">
      <div className="flex items-center gap-3 mb-6 flex-wrap">
        <Link to="/" className="text-gray-400 hover:text-gray-600 text-sm">← 返回列表</Link>
        <span className="text-gray-300">/</span>
        <span className="font-mono text-gray-700">{sub.req_no}</span>
        <StatusBadge status={sub.status} />
        {!!sub.resubmit_count && sub.resubmit_count > 0 && (
          <span className="text-xs font-semibold text-orange-700 bg-orange-100 px-2 py-0.5 rounded">
            第 {sub.resubmit_count + 1} 次送審
          </span>
        )}
        {sub.kaggle_status && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded ${
              KAGGLE_STATUS_COLOR[sub.kaggle_status] ?? 'bg-gray-100 text-gray-700'
            }`}
          >
            Kaggle: {sub.kaggle_status}
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 border-b">
        {(['info', 'kaggle', 'history'] as TabKey[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm -mb-px border-b-2 ${
              tab === t
                ? 'border-indigo-600 text-indigo-700 font-medium'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {t === 'info' ? '需求資訊' : t === 'kaggle' ? 'Kaggle 訓練' : '審核軌跡'}
          </button>
        ))}
      </div>

      {tab === 'info' && (
        <>
          {/* 退件 banner */}
          {sub.status === 'rejected' && rejectionReasons.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-300 rounded p-4 mb-4">
              <h3 className="text-sm font-semibold text-yellow-800 mb-2">退件缺失項</h3>
              <ul className="text-sm text-gray-700 list-disc list-inside space-y-1">
                {rejectionReasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
              {sub.rejection_note && (
                <p className="text-sm text-gray-600 mt-2">審核員補充：{sub.rejection_note}</p>
              )}

              <div className="mt-4">
                {/* P3-5: 可展開欄位編輯區 */}
                <button
                  type="button"
                  onClick={() => {
                    if (!showEditFields && sub) {
                      setEditFields({
                        class_list: sub.class_list || '',
                        dataset_count: sub.dataset_count || '',
                        dataset_val_count: sub.dataset_val_count?.toString() || '',
                        dataset_test_count: sub.dataset_test_count?.toString() || '',
                        dataset_source: sub.dataset_source || '',
                        kaggle_dataset_url: sub.kaggle_dataset_url || '',
                      })
                    }
                    setShowEditFields((v) => !v)
                  }}
                  className="text-sm text-indigo-600 hover:underline mb-2 flex items-center gap-1"
                >
                  {showEditFields ? '▲ 收起欄位編輯' : '▼ 編輯需求單欄位'}
                </button>
                {showEditFields && (
                  <div className="border border-indigo-200 rounded p-3 mb-3 bg-indigo-50 space-y-2">
                    <p className="text-xs text-indigo-700 font-medium">修改欄位後點「儲存變更」，再填補件說明送審</p>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">類別清單</label>
                      <textarea
                        rows={2}
                        className="w-full border rounded p-2 text-sm"
                        value={editFields.class_list}
                        onChange={(e) => setEditFields((v) => ({ ...v, class_list: e.target.value }))}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-gray-500 mb-0.5">資料集大小描述</label>
                        <input
                          className="w-full border rounded p-2 text-sm"
                          value={editFields.dataset_count}
                          onChange={(e) => setEditFields((v) => ({ ...v, dataset_count: e.target.value }))}
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-0.5">資料來源</label>
                        <input
                          className="w-full border rounded p-2 text-sm"
                          value={editFields.dataset_source}
                          onChange={(e) => setEditFields((v) => ({ ...v, dataset_source: e.target.value }))}
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-0.5">Validation 數量</label>
                        <input
                          type="number"
                          className="w-full border rounded p-2 text-sm"
                          value={editFields.dataset_val_count}
                          onChange={(e) => setEditFields((v) => ({ ...v, dataset_val_count: e.target.value }))}
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-0.5">Test 數量</label>
                        <input
                          type="number"
                          className="w-full border rounded p-2 text-sm"
                          value={editFields.dataset_test_count}
                          onChange={(e) => setEditFields((v) => ({ ...v, dataset_test_count: e.target.value }))}
                        />
                      </div>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Kaggle dataset URL</label>
                      <input
                        className="w-full border rounded p-2 text-sm"
                        value={editFields.kaggle_dataset_url}
                        onChange={(e) => setEditFields((v) => ({ ...v, kaggle_dataset_url: e.target.value }))}
                      />
                    </div>
                    <button
                      disabled={updateFieldsMut.isPending}
                      onClick={() => {
                        const payload: Record<string, string | number | undefined> = {
                          class_list: editFields.class_list || undefined,
                          dataset_count: editFields.dataset_count || undefined,
                          dataset_source: editFields.dataset_source || undefined,
                          kaggle_dataset_url: editFields.kaggle_dataset_url || undefined,
                          dataset_val_count: editFields.dataset_val_count ? parseInt(editFields.dataset_val_count) : undefined,
                          dataset_test_count: editFields.dataset_test_count ? parseInt(editFields.dataset_test_count) : undefined,
                        }
                        updateFieldsMut.mutate(payload)
                      }}
                      className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
                    >
                      {updateFieldsMut.isPending ? '儲存中...' : '儲存變更'}
                    </button>
                  </div>
                )}
                <textarea
                  rows={2}
                  className="w-full border rounded p-2 text-sm mb-2"
                  placeholder="請說明補件後修改了什麼..."
                  value={resubmitNote}
                  onChange={(e) => setResubmitNote(e.target.value)}
                />
                <button
                  disabled={resubmitMut.isPending || !resubmitNote.trim()}
                  onClick={() => resubmitMut.mutate(resubmitNote.trim())}
                  className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {resubmitMut.isPending ? '送出中...' : '補件 resubmit'}
                </button>
              </div>
            </div>
          )}

          {/* 資料集阻塞警示框 */}
          {sub.dataset_status && sub.dataset_status !== 'ready' && (
            <div className="bg-orange-50 border border-orange-300 rounded p-4 mb-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-orange-600 font-bold text-sm">尚無法訓練</span>
                <span className="text-xs font-medium bg-orange-200 text-orange-800 px-2 py-0.5 rounded">
                  {sub.dataset_status}
                </span>
              </div>
              {sub.blocked_reason && (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{sub.blocked_reason}</p>
              )}
            </div>
          )}

          <InfoCard sub={sub} />

          {/* 狀態機操作 */}
          {actions.length > 0 && (
            <div className="bg-white rounded shadow p-5 mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">狀態操作</h3>
              {actionError && (
                <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-3 py-2 mb-3">
                  {actionError}
                </div>
              )}
              <textarea
                className="w-full border rounded p-2 text-sm mb-3"
                rows={2}
                placeholder="備注（審核意見、拒絕原因等）"
                value={noteInput}
                onChange={(e) => setNoteInput(e.target.value)}
              />
              <div className="flex gap-2">
                {actions.map(({ action, label, color }) => (
                  <button
                    key={action}
                    disabled={actionMut.isPending}
                    onClick={() => actionMut.mutate({ action, note: noteInput || undefined })}
                    className={`px-4 py-2 text-sm rounded font-medium ${COLOR_MAP[color]} disabled:opacity-50`}
                  >
                    {actionMut.isPending ? '處理中...' : label}
                  </button>
                ))}
                {sub.status === 'submitted' && (
                  <button
                    disabled={actionMut.isPending || rejectMut.isPending}
                    onClick={() => setShowRejectModal(true)}
                    className="px-4 py-2 text-sm rounded font-medium bg-red-600 hover:bg-red-700 text-white disabled:opacity-50"
                  >
                    退件
                  </button>
                )}
              </div>
            </div>
          )}

          {/* 模型版本 */}
          <div className="bg-white rounded shadow p-5">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">相關模型版本</h3>
            {!versions || versions.length === 0 ? (
              <p className="text-sm text-gray-400">尚無模型版本記錄</p>
            ) : (
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">版本</th>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">訓練日期</th>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">mAP50</th>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">Pass/Fail</th>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">狀態</th>
                    <th className="text-left px-3 py-2 text-gray-500 font-medium">驗收</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {versions.map((v) => (
                    <tr key={v.id} className="hover:bg-gray-50">
                      <td className="px-3 py-2 font-mono">{v.version}</td>
                      <td className="px-3 py-2 text-gray-500">{v.train_date || '-'}</td>
                      <td className="px-3 py-2">{v.map50_actual ?? v.map50 ?? '-'}</td>
                      <td className="px-3 py-2">
                        {v.pass_fail ? (
                          <span className={`text-xs font-bold ${
                            v.pass_fail === 'pass' ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {v.pass_fail.toUpperCase()}
                          </span>
                        ) : '-'}
                      </td>
                      <td className="px-3 py-2"><StatusBadge status={v.status} /></td>
                      <td className="px-3 py-2">
                        <Link
                          to={`/registry/${v.id}/accept`}
                          className="text-indigo-600 text-xs hover:underline"
                        >
                          驗收
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}

      {tab === 'kaggle' && (
        <div className="bg-white rounded shadow p-5 space-y-4">
          <h3 className="text-sm font-semibold text-gray-700">Kaggle Kernel 訓練</h3>

          {sub.kaggle_kernel_slug ? (
            <>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              <Item label="Kernel" value={sub.kaggle_kernel_slug} mono />
              <Item label="版本" value={sub.kaggle_kernel_version?.toString()} mono />
              <Item label="狀態" value={sub.kaggle_status} />
              <Item
                label="最後更新"
                value={
                  sub.kaggle_status_updated_at
                    ? new Date(sub.kaggle_status_updated_at).toLocaleString('zh-TW')
                    : undefined
                }
              />
              <Item
                label="開始訓練"
                value={
                  sub.training_started_at
                    ? new Date(sub.training_started_at).toLocaleString('zh-TW')
                    : undefined
                }
              />
              <Item
                label="訓練完成"
                value={
                  sub.training_completed_at
                    ? new Date(sub.training_completed_at).toLocaleString('zh-TW')
                    : undefined
                }
              />
              <Item
                label="GPU 時數"
                value={
                  sub.gpu_seconds
                    ? `${Math.floor(sub.gpu_seconds / 3600)}h ${Math.floor((sub.gpu_seconds % 3600) / 60)}m`
                    : undefined
                }
              />
              <Item
                label="估算成本"
                value={sub.estimated_cost_usd != null ? `$${sub.estimated_cost_usd.toFixed(4)}` : undefined}
              />
              <Item
                label="預算上限"
                value={sub.max_budget_usd != null ? `$${sub.max_budget_usd.toFixed(2)}` : undefined}
              />
              <Item
                label="重試次數"
                value={`${sub.retry_count ?? 0} / ${sub.max_retries ?? 2}`}
              />
            </dl>
            {sub.max_budget_usd && sub.estimated_cost_usd != null && (
              <BudgetBar current={sub.estimated_cost_usd} budget={sub.max_budget_usd} exceeded={!!sub.budget_exceeded_notified} />
            )}
            </>
          ) : (
            <p className="text-sm text-gray-500">尚未綁定 Kaggle kernel。</p>
          )}

          {(sub.status === 'approved' || sub.status === 'training') && (
            <div className="border-t pt-4 space-y-2">
              <input
                type="text"
                placeholder="kernel slug，例：boardgamegroup/mh-2026-001"
                className="w-full border rounded p-2 text-sm font-mono"
                value={kernelSlug}
                onChange={(e) => setKernelSlug(e.target.value)}
              />
              <input
                type="number"
                placeholder="版本號（可選）"
                className="w-full border rounded p-2 text-sm"
                value={kernelVersion}
                onChange={(e) => setKernelVersion(e.target.value)}
              />
              <button
                disabled={attachMut.isPending || !kernelSlug.trim()}
                onClick={() =>
                  attachMut.mutate({
                    slug: kernelSlug.trim(),
                    version: kernelVersion ? Number(kernelVersion) : undefined,
                  })
                }
                className="px-4 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
              >
                {attachMut.isPending ? '綁定中...' : '綁定 Kernel 並開始訓練'}
              </button>
            </div>
          )}

          {sub.kaggle_kernel_slug && (
            <div className="border-t pt-4 flex gap-2">
              <button
                disabled={refreshKaggleMut.isPending}
                onClick={() => refreshKaggleMut.mutate()}
                className="px-4 py-2 text-sm bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
              >
                {refreshKaggleMut.isPending ? '刷新中...' : '手動刷新 Kaggle 狀態'}
              </button>
              <a
                href={`https://www.kaggle.com/code/${sub.kaggle_kernel_slug}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 text-sm border rounded text-indigo-600 hover:bg-indigo-50"
              >
                開 Kaggle 頁面 ↗
              </a>
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="bg-white rounded shadow p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">審核軌跡</h3>
          {!history || history.length === 0 ? (
            <p className="text-sm text-gray-400">尚無紀錄</p>
          ) : (
            <ol className="space-y-3">
              {history.map((h) => (
                <li key={h.id} className="border-l-2 border-indigo-200 pl-4 pb-3">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="font-mono text-sm font-semibold text-indigo-700">
                      {h.action}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(h.created_at).toLocaleString('zh-TW')}
                    </span>
                    {h.actor && (
                      <span className="text-xs text-gray-400">by {h.actor}</span>
                    )}
                  </div>
                  {h.reasons && h.reasons.length > 0 && (
                    <ul className="text-xs text-gray-600 list-disc list-inside mt-1">
                      {h.reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  )}
                  {h.note && (
                    <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{h.note}</p>
                  )}
                  {h.meta && Object.keys(h.meta).length > 0 && (
                    <pre className="text-xs text-gray-500 mt-1 bg-gray-50 rounded p-1 overflow-x-auto">
                      {JSON.stringify(h.meta, null, 2)}
                    </pre>
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {showRejectModal && (
        <RejectModal
          submission={sub}
          onConfirm={(reasons, note) => rejectMut.mutate({ reasons, note })}
          onCancel={() => setShowRejectModal(false)}
          isPending={rejectMut.isPending}
        />
      )}
    </div>
  )
}

function InfoCard({ sub }: { sub: Submission }) {
  return (
    <div className="bg-white rounded shadow p-5 mb-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">
        {sub.req_name || sub.product}
      </h2>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
        <Item label="需求單號" value={sub.req_no} mono />
        <Item label="產品" value={sub.product} />
        <Item label="業務公司" value={sub.company} />
        <Item label="提交人" value={sub.submitter} />
        <Item label="優先度" value={sub.priority} mono />
        <Item label="模型類型" value={sub.model_type} />
        <Item label="mAP50 目標" value={sub.map50_target?.toString()} />
        <Item label="架構" value={sub.arch} />
        <Item label="資料來源" value={sub.dataset_source} />
        <Item label="標注格式" value={sub.label_format} />
        <Item label="預計交付" value={sub.expected_delivery} />
        <Item label="建立時間" value={new Date(sub.created_at).toLocaleString('zh-TW')} />
      </dl>
      {sub.purpose && (
        <div className="mt-4">
          <span className="text-xs font-medium text-gray-500 uppercase">業務描述</span>
          <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{sub.purpose}</p>
        </div>
      )}
      {sub.class_list && (
        <div className="mt-3">
          <span className="text-xs font-medium text-gray-500 uppercase">類別清單</span>
          <p className="text-sm text-gray-700 mt-1 font-mono">{sub.class_list}</p>
        </div>
      )}
      {sub.reviewer_note && (
        <div className="mt-3 bg-yellow-50 border border-yellow-200 rounded p-3">
          <span className="text-xs font-medium text-yellow-700">審核意見</span>
          <p className="text-sm text-gray-700 mt-1">{sub.reviewer_note}</p>
        </div>
      )}
    </div>
  )
}

function Item({
  label,
  value,
  mono,
}: {
  label: string
  value?: string | null
  mono?: boolean
}) {
  return (
    <div>
      <dt className="text-xs text-gray-400 uppercase font-medium">{label}</dt>
      <dd className={`text-gray-800 mt-0.5 ${mono ? 'font-mono text-xs' : ''}`}>
        {value || <span className="text-gray-300">-</span>}
      </dd>
    </div>
  )
}

function BudgetBar({
  current,
  budget,
  exceeded,
}: {
  current: number
  budget: number
  exceeded: boolean
}) {
  const pct = Math.min(100, (current / budget) * 100)
  const overPct = current > budget ? Math.min(100, ((current - budget) / budget) * 100) : 0
  const color =
    current >= budget ? 'bg-red-500' : pct >= 70 ? 'bg-yellow-500' : 'bg-green-500'
  return (
    <div className="mt-4">
      <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
        <span>預算使用率</span>
        <span>
          ${current.toFixed(2)} / ${budget.toFixed(2)}
          {exceeded && <span className="text-red-600 font-semibold ml-2">已超預算</span>}
        </span>
      </div>
      <div className="h-3 bg-gray-100 rounded overflow-hidden relative">
        <div className={`${color} h-full transition-all`} style={{ width: `${pct}%` }} />
        {overPct > 0 && (
          <div
            className="bg-red-700 h-full absolute top-0 right-0"
            style={{ width: `${overPct}%` }}
            title={`超出預算 $${(current - budget).toFixed(2)}`}
          />
        )}
      </div>
    </div>
  )
}

function parseReasons(raw: string | null): string[] {
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === 'string')
  } catch {
    // 若舊格式是單行文字，包成 array
    return [raw]
  }
  return []
}
