import {
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts'

interface SparklinePoint {
  v: number
}

interface KpiCardProps {
  label: string
  value: number | string
  sub?: string
  /** tone 對應 indigo/blue/green/purple/amber/slate */
  tone?: 'indigo' | 'blue' | 'green' | 'purple' | 'amber' | 'slate'
  /** delta：正數顯示綠色箭頭，負數紅色，undefined 不顯示 */
  delta?: number
  /** sparkline 資料點，最多 12 個 */
  sparkline?: SparklinePoint[]
}

const TONE_LABEL: Record<string, string> = {
  indigo: 'text-indigo-600',
  blue:   'text-blue-600',
  green:  'text-emerald-600',
  purple: 'text-purple-600',
  amber:  'text-amber-600',
  slate:  'text-slate-600',
}

const TONE_SPARKLINE: Record<string, string> = {
  indigo: '#6366f1',
  blue:   '#3b82f6',
  green:  '#10b981',
  purple: '#a855f7',
  amber:  '#f59e0b',
  slate:  '#64748b',
}

export function KpiCard({
  label,
  value,
  sub,
  tone = 'indigo',
  delta,
  sparkline,
}: KpiCardProps) {
  const valueColor = TONE_LABEL[tone] ?? TONE_LABEL.indigo
  const sparkColor = TONE_SPARKLINE[tone] ?? TONE_SPARKLINE.indigo

  return (
    <div className="bg-white rounded-xl p-4 border border-gray-100 flex flex-col gap-1" style={{ boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
      <div className="text-xs text-gray-500 font-medium">{label}</div>

      <div className="flex items-end justify-between gap-2 mt-1">
        <div className={`text-3xl font-bold ${valueColor} leading-none`}>{value}</div>

        {/* Sparkline（選配） */}
        {sparkline && sparkline.length > 1 && (
          <div className="w-20 h-10 flex-shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sparkline}>
                <Line
                  type="monotone"
                  dataKey="v"
                  stroke={sparkColor}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Delta 箭頭 */}
      {delta !== undefined && (
        <div className={`text-xs font-medium flex items-center gap-0.5 ${delta >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
          <span>{delta >= 0 ? '↑' : '↓'}</span>
          <span>{Math.abs(delta)}</span>
          <span className="text-gray-400 font-normal ml-1">vs 上週</span>
        </div>
      )}

      {/* Sub 說明 */}
      {sub && <div className="text-[11px] text-gray-400">{sub}</div>}
    </div>
  )
}
