import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getCachedUserInfo, logout } from '../auth'
import { versionApi } from '../api/client'

// lucide-react icons（inline SVG fallback — 避免等 M1 安裝依賴）
function IconCpu({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  )
}
function IconFilePlus({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="12" y1="18" x2="12" y2="12" />
      <line x1="9" y1="15" x2="15" y2="15" />
    </svg>
  )
}
function IconList({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  )
}
function IconCheckSquare({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 11 12 14 22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </svg>
  )
}
function IconClipboardCheck({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2" />
      <rect x="9" y="3" width="6" height="4" rx="1" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  )
}
function IconDatabase({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  )
}
function IconBarChart2({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  )
}
function IconKey({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21 2-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  )
}
function IconZap({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  )
}
function IconClock({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}
function IconMenu({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  )
}
function IconX({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}
function IconLogOut({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  )
}

// NAV 分群：提交 / 審核 / 模型 / 系統
const NAV_GROUPS = [
  {
    label: '提交',
    items: [
      { to: '/',       label: '需求單列表', Icon: IconList },
      { to: '/submit', label: '新增需求',   Icon: IconFilePlus },
    ],
  },
  {
    label: '審核',
    items: [
      { to: '/review',     label: 'CTO 審核',  Icon: IconCheckSquare },
      { to: '/acceptance', label: '驗收佇列',  Icon: IconClipboardCheck },
    ],
  },
  {
    label: '模型',
    items: [
      { to: '/registry', label: '模型清冊',  Icon: IconDatabase },
      { to: '/stats',    label: '統計',      Icon: IconBarChart2 },
      { to: '/predict',  label: '推論測試',  Icon: IconZap },
    ],
  },
  {
    label: '系統',
    items: [
      { to: '/queue',          label: '訓練隊列', Icon: IconClock },
      { to: '/admin/api-keys', label: 'API Keys', Icon: IconKey },
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
          <IconCpu size={22} />
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
            <IconX size={20} />
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
          <IconLogOut size={14} />
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
            <IconMenu size={22} />
          </button>
          <div className="flex items-center gap-2">
            <IconCpu size={20} />
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
