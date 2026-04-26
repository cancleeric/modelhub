/**
 * ResourceHealthWidget.tsx — 資源健康 Widget（Sprint 24 Task 24-2）
 *
 * 每 60 秒 refresh /api/health/system-status，顯示：
 * - Kaggle 本週剩餘配額（進度條，< 5h 變紅）
 * - Lightning 本月剩餘配額（進度條，< 3h 變紅）
 * - 訓練中/等待中任務數
 * - Poller 最後執行時間（> 5 分鐘標紅）
 *
 * 整合到 StatsPage。
 */

import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

interface SystemStatus {
  kaggle_remaining_hours: number | null
  lightning_remaining_hours: number | null
  kaggle_poller_last_at: string | null
  lightning_poller_last_at: string | null
  api_server: string
  active_trainings: number
  queue_status?: {
    waiting_count: number
    running_count: number
  }
  poller_health?: {
    kaggle: 'ok' | 'warning' | 'critical'
    lightning: 'ok' | 'warning' | 'critical'
  }
  success_rate_24h?: number | null
}

function pollerAgeLabel(isoStr: string | null): { label: string; isStale: boolean } {
  if (!isoStr) return { label: '從未執行', isStale: true }
  const ms = Date.now() - new Date(isoStr).getTime()
  const mins = Math.floor(ms / 60000)
  return {
    label: mins === 0 ? '剛執行' : `${mins} 分鐘前`,
    isStale: mins >= 5,
  }
}

function QuotaBar({
  label,
  remaining,
  limit,
  warnThreshold,
  colorWhenWarn = 'bg-red-500',
  colorNormal = 'bg-green-500',
}: {
  label: string
  remaining: number | null
  limit: number
  warnThreshold: number
  colorWhenWarn?: string
  colorNormal?: string
}) {
  if (remaining == null) {
    return (
      <div>
        <div className="text-xs text-gray-500 mb-1">{label}</div>
        <span className="text-xs text-gray-400">無資料</span>
      </div>
    )
  }
  const safeRemaining = remaining ?? 0
  const pct = Math.min(100, (safeRemaining / limit) * 100)
  const isWarn = safeRemaining < warnThreshold
  const barColor = isWarn ? colorWhenWarn : colorNormal

  return (
    <div>
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="h-2 bg-gray-100 rounded overflow-hidden mb-1">
        <div className={`${barColor} h-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-medium ${isWarn ? 'text-red-600' : 'text-gray-700'}`}>
        {safeRemaining.toFixed(1)} h 剩餘
        {isWarn && ' ⚠ 即將耗盡'}
      </span>
    </div>
  )
}

function PollerStatusDot({ health }: { health: 'ok' | 'warning' | 'critical' | undefined }) {
  if (!health) return null
  const colorMap = { ok: 'bg-green-500', warning: 'bg-orange-400', critical: 'bg-red-500' }
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full mr-1 ${colorMap[health]}`}
      title={health}
    />
  )
}

export default function ResourceHealthWidget() {
  const { data: status, isLoading } = useQuery<SystemStatus>({
    queryKey: ['system-status-widget'],
    queryFn: () => api.get<SystemStatus>('/api/health/system-status').then((r) => r.data),
    refetchInterval: 60 * 1000, // 每 60 秒 refresh
  })

  if (isLoading || !status) {
    return (
      <div className="bg-white rounded shadow p-5 mb-6 animate-pulse h-32" />
    )
  }

  const kagglePoller = pollerAgeLabel(status.kaggle_poller_last_at)
  const lightningPoller = pollerAgeLabel(status.lightning_poller_last_at)

  return (
    <div className="bg-white rounded shadow p-5 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-700">資源健康狀態</h3>
        <span className="text-xs text-gray-400">每 60 秒更新</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
        {/* Kaggle 配額 */}
        <QuotaBar
          label="Kaggle 本週剩餘"
          remaining={status.kaggle_remaining_hours}
          limit={30}
          warnThreshold={5}
        />

        {/* Lightning 配額 */}
        <QuotaBar
          label="Lightning 本月剩餘"
          remaining={status.lightning_remaining_hours}
          limit={22}
          warnThreshold={3}
          colorNormal="bg-blue-500"
        />

        {/* 任務數 */}
        <div>
          <div className="text-xs text-gray-500 mb-1">任務狀態</div>
          <div className="flex gap-3 flex-wrap">
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded bg-blue-100 text-blue-700">
              訓練中 {status.active_trainings}
            </span>
            {status.queue_status && (
              <>
                <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded bg-orange-100 text-orange-700">
                  等待中 {status.queue_status.waiting_count}
                </span>
                <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded bg-purple-100 text-purple-700">
                  派發中 {status.queue_status.running_count}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Kaggle Poller */}
        <div>
          <div className="text-xs text-gray-500 mb-1">
            <PollerStatusDot health={status.poller_health?.kaggle} />
            Kaggle Poller
          </div>
          <span className={`text-xs font-medium ${kagglePoller.isStale ? 'text-red-600' : 'text-gray-700'}`}>
            {kagglePoller.label}
          </span>
        </div>

        {/* Lightning Poller */}
        <div>
          <div className="text-xs text-gray-500 mb-1">
            <PollerStatusDot health={status.poller_health?.lightning} />
            Lightning Poller
          </div>
          <span className={`text-xs font-medium ${lightningPoller.isStale ? 'text-red-600' : 'text-gray-700'}`}>
            {lightningPoller.label}
          </span>
        </div>

        {/* 24h 訓練成功率 */}
        <div>
          <div className="text-xs text-gray-500 mb-1">近 24h 訓練成功率</div>
          {status.success_rate_24h != null ? (
            <span
              className={`text-sm font-semibold ${
                status.success_rate_24h >= 80
                  ? 'text-green-600'
                  : status.success_rate_24h >= 50
                  ? 'text-orange-500'
                  : 'text-red-600'
              }`}
            >
              {status.success_rate_24h.toFixed(1)}%
            </span>
          ) : (
            <span className="text-xs text-gray-400">無資料</span>
          )}
        </div>
      </div>
    </div>
  )
}
