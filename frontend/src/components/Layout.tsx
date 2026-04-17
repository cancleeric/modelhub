import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getCachedUserInfo, logout } from '../auth'
import { versionApi } from '../api/client'

const NAV = [
  { to: '/',               label: '需求單列表' },
  { to: '/submit',         label: '+ 新增需求' },
  { to: '/review',         label: 'CTO 審核' },
  { to: '/acceptance',     label: '驗收佇列' },
  { to: '/registry',       label: '模型清冊' },
  { to: '/stats',          label: '統計' },
  { to: '/admin/api-keys', label: 'API Keys' },
  { to: '/predict',        label: '推論測試' },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  const user = getCachedUserInfo()
  const displayName = user?.name ?? user?.preferred_username ?? user?.email ?? '未知用戶'
  const { data: ver } = useQuery({
    queryKey: ['version'],
    queryFn: () => versionApi.get(),
    staleTime: 60_000,
  })

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-indigo-700 text-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
          <span className="font-bold text-lg tracking-wide flex items-baseline gap-2">
            ModelHub
            {ver && (
              <span
                className="text-xs font-mono text-indigo-200"
                title={`build=${ver.build} commit=${ver.commit}`}
              >
                v{ver.version}
              </span>
            )}
          </span>
          <nav className="flex gap-4 flex-1">
            {NAV.map((n) => (
              <Link
                key={n.to}
                to={n.to}
                className={`text-sm px-3 py-1 rounded hover:bg-indigo-600 transition-colors ${
                  pathname === n.to ? 'bg-indigo-800 font-semibold' : ''
                }`}
              >
                {n.label}
              </Link>
            ))}
          </nav>
          <div className="flex items-center gap-3 ml-auto">
            <span className="text-sm text-indigo-200">{displayName}</span>
            <button
              onClick={() => logout()}
              className="text-xs px-3 py-1.5 bg-indigo-800 hover:bg-indigo-900 rounded transition-colors"
            >
              登出
            </button>
          </div>
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">{children}</main>
    </div>
  )
}
