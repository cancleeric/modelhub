import { Link, useLocation } from 'react-router-dom'

const NAV = [
  { to: '/',          label: '需求單列表' },
  { to: '/submit',    label: '+ 新增需求' },
  { to: '/review',    label: 'CTO 審核' },
  { to: '/registry',  label: '模型清冊' },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-indigo-700 text-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
          <span className="font-bold text-lg tracking-wide">ModelHub</span>
          <nav className="flex gap-4">
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
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">{children}</main>
    </div>
  )
}
