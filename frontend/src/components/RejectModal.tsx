import { useState } from 'react'
import type { Submission } from '../api/client'

export const REJECT_REASONS = [
  'expected_delivery 未填',
  '資料集數量不足',
  '資料集 test set 未規劃',
  'submitter 非真實聯絡人',
  'map50_95_target 未填',
  'model_type 格式不符',
  '架構選型需重新評估',
]

interface RejectModalProps {
  submission: Submission
  onConfirm: (reasons: string[], note: string) => void
  onCancel: () => void
  isPending: boolean
}

export default function RejectModal({ submission, onConfirm, onCancel, isPending }: RejectModalProps) {
  const [checked, setChecked] = useState<string[]>([])
  const [otherChecked, setOtherChecked] = useState(false)
  const [otherText, setOtherText] = useState('')

  const toggleReason = (reason: string) => {
    setChecked((prev) =>
      prev.includes(reason) ? prev.filter((r) => r !== reason) : [...prev, reason],
    )
  }

  const buildReasons = (): string[] => {
    const list = [...checked]
    if (otherChecked) list.push('其他')
    return list
  }

  const canConfirm = checked.length > 0 || (otherChecked && otherText.trim())

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">退件確認</h3>
        <p className="text-sm text-gray-500 mb-4">
          {submission.req_no} — {submission.req_name || submission.product}
        </p>

        <p className="text-sm font-medium text-gray-700 mb-2">退件原因（可複選）</p>
        <div className="space-y-2 mb-4">
          {REJECT_REASONS.map((reason) => (
            <label key={reason} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={checked.includes(reason)}
                onChange={() => toggleReason(reason)}
                className="rounded"
              />
              {reason}
            </label>
          ))}
          <label className="flex items-start gap-2 text-sm text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={otherChecked}
              onChange={(e) => setOtherChecked(e.target.checked)}
              className="rounded mt-0.5"
            />
            其他（自由文字）
          </label>
        </div>

        {otherChecked && (
          <textarea
            rows={3}
            placeholder="補充說明..."
            className="w-full border rounded p-2 text-sm mb-4"
            value={otherText}
            onChange={(e) => setOtherText(e.target.value)}
          />
        )}

        {!canConfirm && (
          <p className="text-xs text-red-500 mb-3">請至少選擇一個退件原因</p>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={isPending}
            className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-50 disabled:opacity-50"
          >
            取消
          </button>
          <button
            type="button"
            disabled={!canConfirm || isPending}
            onClick={() => onConfirm(buildReasons(), otherText.trim())}
            className="px-4 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
          >
            {isPending ? '處理中...' : '確認退件'}
          </button>
        </div>
      </div>
    </div>
  )
}
