import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { submissionsApi, registryApi, type Submission, type ModelVersion } from '../api/client'

function startOfMonth(): Date {
  const d = new Date()
  d.setDate(1)
  d.setHours(0, 0, 0, 0)
  return d
}

function groupBy<T>(arr: T[], keyFn: (x: T) => string): Record<string, number> {
  const m: Record<string, number> = {}
  arr.forEach((x) => {
    const k = keyFn(x) || '(未填)'
    m[k] = (m[k] ?? 0) + 1
  })
  return m
}

/** 取得近 N 週的週一 Date（由舊到新） */
function getWeekStarts(n: number): Date[] {
  const now = new Date()
  const day = now.getDay()
  const thisMonday = new Date(now)
  thisMonday.setDate(now.getDate() - (day === 0 ? 6 : day - 1))
  thisMonday.setHours(0, 0, 0, 0)
  const result: Date[] = []
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(thisMonday)
    d.setDate(thisMonday.getDate() - i * 7)
    result.push(d)
  }
  return result
}

function isoWeekLabel(d: Date): string {
  const m = d.getMonth() + 1
  const day = d.getDate()
  return `${m}/${day}`
}

interface ResourceQuota {
  kaggle: {
    weekly_limit_hours: number
    used_hours_this_week: number
    remaining_hours_this_week: number
    total_gpu_hours_all_time: number
  }
  lightning: {
    monthly_limit_hours: number
    used_hours_this_month: number
    remaining_hours_this_month: number
    is_free_tier: boolean
  }
}

export default function StatsPage() {
  const { data: subs } = useQuery({
    queryKey: ['submissions-all'],
    queryFn: () => submissionsApi.list(),
  })
  const { data: versions } = useQuery({
    queryKey: ['registry', 'all'],
    queryFn: () => registryApi.list(),
  })
  const { data: quota } = useQuery<ResourceQuota>({
    queryKey: ['resource-quota'],
    queryFn: () => api.get<ResourceQuota>('/api/health/resource-quota').then((r) => r.data),
    refetchInterval: 5 * 60 * 1000, // 每 5 分鐘更新
  })

  const stats = useMemo(() => {
    const s = subs ?? []
    const v = versions ?? []
    const m0 = startOfMonth()
    const monthSubs = s.filter((x) => new Date(x.created_at) >= m0)
    const monthTrained = s.filter((x) => {
      const t = x.training_completed_at
      return t && new Date(t) >= m0
    })
    const monthAccepted = v.filter((x) => {
      const t = x.accepted_at
      return t && new Date(t) >= m0
    })

    const totalGpuSec = s.reduce((acc, x) => acc + (x.gpu_seconds || 0), 0)
    const totalCost = s.reduce((acc, x) => acc + (x.estimated_cost_usd || 0), 0)

    const leadTimes = v
      .filter((x) => x.accepted_at)
      .map((x) => {
        const sub = s.find((y) => y.req_no === x.req_no)
        if (!sub) return null
        return (new Date(x.accepted_at!).getTime() - new Date(sub.created_at).getTime()) / 1000 / 3600
      })
      .filter((x): x is number => x != null)
    const avgLeadHours = leadTimes.length > 0
      ? leadTimes.reduce((a, b) => a + b, 0) / leadTimes.length
      : null

    // 訓練資源分布
    const resourceDist: Record<string, number> = { kaggle: 0, lightning: 0, local_mps: 0, unknown: 0 }
    s.forEach((x) => {
      const r = x.training_resource
      if (!r) {
        resourceDist.unknown += 1
      } else if (r === 'kaggle') {
        resourceDist.kaggle += 1
      } else if (r === 'lightning') {
        resourceDist.lightning += 1
      } else if (r === 'local_mps') {
        resourceDist.local_mps += 1
      } else {
        // ssh@host 等歸為 unknown
        resourceDist.unknown += 1
      }
    })

    // 驗收通過率
    const acceptedCount = s.filter((x) => x.status === 'accepted').length
    const failedCount = s.filter((x) => x.status === 'training_failed' || x.status === 'failed').length
    const passRateDenom = acceptedCount + failedCount
    const passRate = passRateDenom > 0 ? (acceptedCount / passRateDenom) * 100 : null

    return {
      monthSubs: monthSubs.length,
      monthTrained: monthTrained.length,
      monthAccepted: monthAccepted.length,
      totalGpuSec,
      totalCost,
      avgLeadHours,
      byProduct: groupBy<Submission>(s, (x) => x.product),
      byCompany: groupBy<Submission>(s, (x) => x.company),
      byStatus: groupBy<Submission>(s, (x) => x.status),
      totalSubs: s.length,
      totalVersions: v.length,
      currentCount: v.filter((x: ModelVersion) => x.is_current).length,
      resourceDist,
      passRate,
    }
  }, [subs, versions])

  // P2-5: 近 12 週趨勢資料（前端計算）
  const weeklyTrend = useMemo(() => {
    const s = subs ?? []
    const v = versions ?? []
    const weekStarts = getWeekStarts(12)
    return weekStarts.map((weekStart, idx) => {
      const weekEnd = idx + 1 < weekStarts.length
        ? weekStarts[idx + 1]
        : new Date(weekStart.getTime() + 7 * 24 * 3600 * 1000)

      const newCount = s.filter((x) => {
        const t = new Date(x.created_at)
        return t >= weekStart && t < weekEnd
      }).length

      const trainedCount = s.filter((x) => {
        if (!x.training_completed_at) return false
        const t = new Date(x.training_completed_at)
        return t >= weekStart && t < weekEnd
      }).length

      const acceptedCount = v.filter((x) => {
        if (!x.accepted_at) return false
        const t = new Date(x.accepted_at)
        return t >= weekStart && t < weekEnd
      }).length

      return {
        week: isoWeekLabel(weekStart),
        新增工單: newCount,
        訓練完成: trainedCount,
        驗收通過: acceptedCount,
      }
    })
  }, [subs, versions])

  if (!subs || !versions) {
    return <p className="text-gray-500">載入中...</p>
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">統計儀表板</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <KpiCard label="本月新增" value={stats.monthSubs} sub="needs submitted" tone="indigo" />
        <KpiCard label="本月訓練完成" value={stats.monthTrained} sub="training → trained" tone="blue" />
        <KpiCard label="本月驗收通過" value={stats.monthAccepted} sub="model accepted" tone="green" />
        <KpiCard
          label="平均交期"
          value={stats.avgLeadHours != null ? `${stats.avgLeadHours.toFixed(1)}h` : '-'}
          sub="submit → accept"
          tone="purple"
        />
        <KpiCard
          label="累計 GPU"
          value={`${Math.floor(stats.totalGpuSec / 3600)}h`}
          sub={`估算 $${stats.totalCost.toFixed(2)}`}
          tone="amber"
        />
        <KpiCard label="總需求單" value={stats.totalSubs} sub="all time" tone="slate" />
        <KpiCard label="模型版本" value={stats.totalVersions} sub={`當前 ${stats.currentCount} 個`} tone="slate" />
        <KpiCard label="活躍狀態" value={Object.keys(stats.byStatus).length} sub="status kinds" tone="slate" />
        <KpiCard
          label="驗收通過率"
          value={stats.passRate != null ? `${stats.passRate.toFixed(1)}%` : '-'}
          sub="accepted / (accepted + failed)"
          tone="green"
        />
      </div>

      {/* P2-5: 近 12 週趨勢折線圖 */}
      <div className="bg-white rounded shadow p-5 mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">近 12 週趨勢</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={weeklyTrend} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="week" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="新增工單"
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="訓練完成"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="驗收通過"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <BarCard title="按產品分布" data={stats.byProduct} />
        <BarCard title="按業務公司分布" data={stats.byCompany} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <BarCard title="按狀態分布" data={stats.byStatus} />
        <BarCard title="訓練資源分布" data={stats.resourceDist} />
      </div>

      {/* 資源配額卡片 */}
      {quota && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <QuotaCard
            title="Kaggle 免費配額（本週）"
            used={quota.kaggle.used_hours_this_week}
            limit={quota.kaggle.weekly_limit_hours}
            remaining={quota.kaggle.remaining_hours_this_week}
            unit="h/週"
            warnThreshold={5}
          />
          <QuotaCard
            title="Lightning AI 免費配額（本月）"
            used={quota.lightning.used_hours_this_month}
            limit={quota.lightning.monthly_limit_hours}
            remaining={quota.lightning.remaining_hours_this_month}
            unit="h/月"
            warnThreshold={3}
          />
        </div>
      )}
    </div>
  )
}

function KpiCard({
  label, value, sub, tone,
}: { label: string; value: number | string; sub: string; tone: string }) {
  const toneMap: Record<string, string> = {
    indigo: 'bg-indigo-100 text-indigo-700',
    blue:   'bg-blue-100 text-blue-700',
    green:  'bg-green-100 text-green-700',
    purple: 'bg-purple-100 text-purple-700',
    amber:  'bg-amber-100 text-amber-700',
    slate:  'bg-slate-100 text-slate-700',
  }
  return (
    <div className="bg-white rounded shadow-sm p-4">
      <div className={`inline-block text-xs font-medium px-2 py-0.5 rounded ${toneMap[tone]}`}>
        {label}
      </div>
      <div className="text-2xl font-bold text-gray-900 mt-2">{value}</div>
      <div className="text-xs text-gray-500 mt-0.5">{sub}</div>
    </div>
  )
}

function QuotaCard({
  title,
  used,
  limit,
  remaining,
  unit,
  warnThreshold,
}: {
  title: string
  used: number
  limit: number
  remaining: number
  unit: string
  warnThreshold: number
}) {
  const pct = Math.min(100, (used / limit) * 100)
  const isWarn = remaining < warnThreshold
  const barColor = isWarn ? 'bg-red-500' : pct >= 70 ? 'bg-yellow-500' : 'bg-green-500'
  return (
    <div className="bg-white rounded shadow p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{title}</h3>
      <div className="flex justify-between text-sm text-gray-600 mb-2">
        <span>已用 <strong>{used.toFixed(1)}</strong> {unit}</span>
        <span className={isWarn ? 'text-red-600 font-semibold' : 'text-gray-600'}>
          剩餘 <strong>{remaining.toFixed(1)}</strong> / {limit} h
        </span>
      </div>
      <div className="h-3 bg-gray-100 rounded overflow-hidden">
        <div className={`${barColor} h-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      {isWarn && (
        <p className="text-xs text-red-600 mt-2 font-medium">
          配額即將耗盡，請注意排程！
        </p>
      )}
    </div>
  )
}

function BarCard({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = Math.max(1, ...entries.map(([, v]) => v))
  return (
    <div className="bg-white rounded shadow p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{title}</h3>
      {entries.length === 0 ? (
        <p className="text-sm text-gray-400">無資料</p>
      ) : (
        <div className="space-y-2">
          {entries.map(([k, v]) => (
            <div key={k} className="flex items-center gap-3">
              <div className="w-32 text-xs text-gray-600 truncate">{k}</div>
              <div className="flex-1 bg-gray-100 rounded h-4 overflow-hidden">
                <div
                  className="bg-indigo-500 h-full"
                  style={{ width: `${(v / max) * 100}%` }}
                />
              </div>
              <div className="w-10 text-right text-sm font-mono text-gray-700">{v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
