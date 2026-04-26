import { login } from '../auth'
import { Cpu, GitBranch, Shield, Zap } from 'lucide-react'

function ProductLogo({ size = 40 }: { size?: number }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className="rounded-xl bg-white/20 flex items-center justify-center"
        style={{ width: size, height: size }}
      >
        <Cpu size={size * 0.55} className="text-white" />
      </div>
      <div>
        <div className="text-2xl font-bold text-white tracking-wide leading-tight">ModelHub</div>
        <div className="text-indigo-200 text-xs">颶核科技 AI 模型版本管理平台</div>
      </div>
    </div>
  )
}

const FEATURES = [
  { Icon: GitBranch, text: '模型版本追蹤與清冊管理' },
  { Icon: Shield,    text: 'CTO 審核 + QA 驗收雙重把關' },
  { Icon: Zap,       text: 'Kaggle / Lightning 訓練資源排程' },
]

export default function LoginPage() {
  return (
    <div className="min-h-screen flex">
      {/* 左欄：品牌 hero */}
      <div className="hidden md:flex flex-1 flex-col justify-center px-12 bg-gradient-to-br from-indigo-700 via-indigo-600 to-violet-600">
        <ProductLogo size={48} />
        <p className="mt-6 text-indigo-100 text-sm leading-relaxed max-w-xs">
          集中管理訓練需求、模型版本與驗收紀錄，讓 AI 能力穩定流入每一個產品。
        </p>
        <div className="mt-8 space-y-3">
          {FEATURES.map(({ Icon, text }) => (
            <div key={text} className="flex items-center gap-3 text-indigo-100 text-sm">
              <div className="w-7 h-7 rounded-lg bg-white/15 flex items-center justify-center flex-shrink-0">
                <Icon size={14} />
              </div>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 右欄：登入表單 */}
      <div className="flex flex-1 items-center justify-center bg-gray-50 px-6">
        <div className="w-full max-w-sm">
          {/* Mobile only logo */}
          <div className="md:hidden flex justify-center mb-8">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center">
                <Cpu size={22} className="text-white" />
              </div>
              <span className="text-xl font-bold text-indigo-700 tracking-wide">ModelHub</span>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-8">
            <h2 className="text-lg font-semibold text-gray-800 mb-1">歡迎回來</h2>
            <p className="text-sm text-gray-500 mb-6">使用颶風集團 LIDS 統一身份認證登入</p>
            <button
              onClick={() => login()}
              className="w-full py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
            >
              以 LIDS 帳號登入
            </button>
            <p className="text-xs text-gray-400 text-center mt-4">
              登入即代表您同意颶風集團資訊安全政策
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
