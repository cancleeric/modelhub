const STATUS_STYLES: Record<string, string> = {
  draft:     'bg-gray-100 text-gray-700',
  submitted: 'bg-blue-100 text-blue-700',
  approved:  'bg-green-100 text-green-700',
  rejected:  'bg-red-100 text-red-700',
  training:  'bg-yellow-100 text-yellow-700',
  trained:   'bg-purple-100 text-purple-700',
  accepted:  'bg-emerald-100 text-emerald-800',
  failed:    'bg-red-200 text-red-900',
  // model version statuses
  active:              'bg-green-100 text-green-700',
  retired:             'bg-gray-200 text-gray-600',
  testing:             'bg-yellow-100 text-yellow-700',
  pending_review:      'bg-blue-100 text-blue-700',
  pending_acceptance:  'bg-orange-100 text-orange-700',
}

const STATUS_LABELS: Record<string, string> = {
  draft:          '草稿',
  submitted:      '已提交',
  approved:       '已核准',
  rejected:       '已拒絕',
  training:       '訓練中',
  trained:        '訓練完成',
  accepted:       '已驗收',
  failed:         '驗收失敗',
  active:              '使用中',
  retired:             '已退役',
  testing:             '測試中',
  pending_review:      '待審核',
  pending_acceptance:  '待驗收',
}

interface Props {
  status: string
}

export default function StatusBadge({ status }: Props) {
  const cls = STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-600'
  const label = STATUS_LABELS[status] ?? status
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {label}
    </span>
  )
}
