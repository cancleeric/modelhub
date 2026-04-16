import { login } from '../auth'

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 to-indigo-100">
      <div className="bg-white rounded-xl shadow-md p-10 flex flex-col items-center gap-6 w-80">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-indigo-700 tracking-wide">ModelHub</h1>
          <p className="text-sm text-gray-500 mt-1">模型版本管理平台</p>
        </div>
        <button
          onClick={() => login()}
          className="w-full py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
        >
          以 LIDS 帳號登入
        </button>
        <p className="text-xs text-gray-400 text-center">
          使用颶風集團 LIDS 統一身份認證
        </p>
      </div>
    </div>
  )
}
