import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getCachedUserInfo, logout } from '../auth'
import { versionApi } from '../api/client'
import {
  Cpu,
  FilePlus,
  List,
  CheckSquare,
  ClipboardCheck,
  Database,
  BarChart2,
  Key,
  Zap,
  Clock,
  Menu,
  X,
  LogOut,
  type LucideIcon,
} from 'lucide-react'

// NAV 分群：提交 / 審核 / 模型 / 系統
const NAV_GROUPS: {
  label: string
  items: { to: string; label: string; Icon: LucideIcon }[]
}[] = [
  {
    label: '提交',
    items: [
      { to: '/',       label: '需求單列表', Icon: List },
      { to: '/submit', label: '新增需求',   Icon: FilePlus },
    ],
  },
  {
    label: '審核',
    items: [
      { to: '/review',     label: 'CTO 審核',  Icon: CheckSquare },
      { to: '/acceptance', label: '驗收佇列',  Icon: ClipboardCheck },
    ],
  },
  {
    label: '模型',
    items: [
      { to: '/registry', label: '模型清冊',  Icon: Database },
      { to: '/stats',    label: '統計',      Icon: BarChart2 },
      { to: '/predict',  label: '推論測試',  Icon: Zap },
    ],
  },
  {
    label: '系統',
    items: [
      { to: '/queue',          label: '訓練隊列', Icon: Clock },
      { to: '/admin/api-keys', label: 'API Keys', Icon: Key },
    ],
  },
]

function SidebarContent({
  pathname,
  displayName,
  ver,
  onClose,
}: {
  pathname: string
  displayName: string
  ver?: { version: string; build: string; commit: string }
  onClose?: () => void
}) {
  // active match：/ 精確比對，其他前綴比對
  function isActive(to: string) {
    if (to === '/') return pathname === '/'
    return pathname.startsWith(to)
  }

  return (
    <div className="flex flex-col h-full bg-indigo-900 text-white w-[220px]">
      {/* Logo header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-indigo-700">
        <Link to="/" className="flex items-center gap-2" onClick={onClose}>
          <Cpu size={22} />
          <div className="leading-tight">
            <span className="font-bold text-base tracking-wide">ModelHub</span>
            {ver && (
              <span
                className="block text-[10px] font-mono text-indigo-300"
                title={`build=${ver.build} commit=${ver.commit}`}
              >
                v{ver.version}
              </span>
            )}
          </div>
        </Link>
        {/* 只在 mobile drawer 模式顯示關閉按鈕 */}
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-indigo-700 transition-colors"
            aria-label="關閉選單"
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Nav groups */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="mb-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400 px-3 mb-1">
              {group.label}
            </div>
            {group.items.map(({ to, label, Icon }) => (
              <Link
                key={to}
                to={to}
                onClick={onClose}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors mb-0.5 ${
                  isActive(to)
                    ? 'bg-indigo-700 text-white font-semibold'
                    : 'text-indigo-200 hover:bg-indigo-800 hover:text-white'
                }`}
              >
                <Icon size={18} />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        ))}
      </nav>

      {/* User footer */}
      <div className="px-4 py-3 border-t border-indigo-700">
        <div className="text-xs text-indigo-300 truncate mb-2">{displayName}</div>
        <button
          onClick={() => logout()}
          className="flex items-center gap-2 text-xs px-3 py-1.5 bg-indigo-800 hover:bg-indigo-700 rounded transition-colors w-full"
        >
          <LogOut size={14} />
          登出
        </button>
      </div>
    </div>
  )
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const user = getCachedUserInfo()
  const displayName = user?.name ?? user?.preferred_username ?? user?.email ?? '未知用戶'
  const { data: ver } = useQuery({
    queryKey: ['version'],
    queryFn: () => versionApi.get(),
    staleTime: 60_000,
  })

  return (
    <div className="min-h-screen flex">
      {/* Desktop sidebar（固定，768px 以上顯示） */}
      <aside className="hidden md:flex flex-shrink-0 h-screen sticky top-0">
        <SidebarContent pathname={pathname} displayName={displayName} ver={ver} />
      </aside>

      {/* Mobile: hamburger button + drawer overlay */}
      <div className="md:hidden">
        {/* 頂部 bar（mobile only） */}
        <header className="fixed top-0 left-0 right-0 z-30 bg-indigo-900 text-white flex items-center px-4 py-3 shadow">
          <button
            onClick={() => setDrawerOpen(true)}
            className="p-1 rounded hover:bg-indigo-700 transition-colors mr-3"
            aria-label="開啟選單"
            data-testid="hamburger-btn"
          >
            <Menu size={22} />
          </button>
          <div className="flex items-center gap-2">
            <Cpu size={20} />
            <span className="font-bold tracking-wide">ModelHub</span>
          </div>
        </header>

        {/* Drawer overlay */}
        {drawerOpen && (
          <>
            {/* 背景遮罩 */}
            <div
              className="fixed inset-0 z-40 bg-black/50"
              onClick={() => setDrawerOpen(false)}
            />
            {/* Drawer */}
            <div className="fixed top-0 left-0 z-50 h-full shadow-xl">
              <SidebarContent
                pathname={pathname}
                displayName={displayName}
                ver={ver}
                onClose={() => setDrawerOpen(false)}
              />
            </div>
          </>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* mobile top bar 佔位（避免內容被 fixed header 遮住） */}
        <div className="md:hidden h-14" />
        <main className="flex-1 px-4 py-6 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
