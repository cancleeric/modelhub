import { Badge } from '../design-system/Badge'
import type { ComponentProps } from 'react'

type BadgeVariant = ComponentProps<typeof Badge>['variant']

// 工作流狀態 → Badge variant（domain-specific mapping 保留在此）
const STATUS_VARIANT: Record<string, BadgeVariant> = {
  draft:              'neutral',
  submitted:          'info',
  approved:           'success',
  rejected:           'error',
  training:           'warning',
  trained:            'info',
  accepted:           'success',
  failed:             'error',
  // model version statuses
  active:             'success',
  retired:            'neutral',
  testing:            'warning',
  pending_review:     'info',
  pending_acceptance: 'warning',
}

const STATUS_LABELS: Record<string, string> = {
  draft:              '草稿',
  submitted:          '已提交',
  approved:           '已核准',
  rejected:           '已拒絕',
  training:           '訓練中',
  trained:            '訓練完成',
  accepted:           '已驗收',
  failed:             '驗收失敗',
  active:             '使用中',
  retired:            '已退役',
  testing:            '測試中',
  pending_review:     '待審核',
  pending_acceptance: '待驗收',
}

interface Props {
  status: string
}

export default function StatusBadge({ status }: Props) {
  const variant = STATUS_VARIANT[status] ?? 'neutral'
  const label = STATUS_LABELS[status] ?? status
  return <Badge variant={variant}>{label}</Badge>
}
