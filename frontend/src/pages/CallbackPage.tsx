import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { handleCallback } from '../auth'

export default function CallbackPage() {
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const code = params.get('code')
    const state = params.get('state')
    const errorParam = params.get('error')

    if (errorParam) {
      setError(`LIDS 授權失敗：${params.get('error_description') ?? errorParam}`)
      return
    }

    if (!code || !state) {
      setError('缺少 OAuth callback 參數')
      return
    }

    handleCallback(code, state)
      .then(() => navigate('/', { replace: true }))
      .catch((err: Error) => setError(err.message))
  }, [navigate])

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <h2 className="text-red-700 font-semibold mb-2">登入失敗</h2>
          <p className="text-red-600 text-sm">{error}</p>
          <button
            onClick={() => (window.location.href = '/login')}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded text-sm hover:bg-red-700"
          >
            返回登入
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <p className="text-gray-500 text-sm">正在完成登入...</p>
      </div>
    </div>
  )
}
