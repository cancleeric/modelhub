/**
 * auth.ts — LIDS PKCE OAuth 認證
 * client_id: modelhub-dev (public client)
 * redirect_uri: http://localhost:3950/callback
 */

const LIDS_URL = 'http://localhost:8073'
const CLIENT_ID = 'modelhub-dev'
const REDIRECT_URI = 'http://localhost:3950/callback'
const SCOPES = 'openid profile email'

// --- PKCE helpers ---

function base64URLEncode(buffer: ArrayBuffer): string {
  return btoa(String.fromCharCode(...new Uint8Array(buffer)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '')
}

async function sha256(plain: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder()
  const data = encoder.encode(plain)
  return crypto.subtle.digest('SHA-256', data)
}

function generateCodeVerifier(): string {
  const array = new Uint8Array(64)
  crypto.getRandomValues(array)
  return base64URLEncode(array.buffer)
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const hash = await sha256(verifier)
  return base64URLEncode(hash)
}

function generateState(): string {
  const array = new Uint8Array(16)
  crypto.getRandomValues(array)
  return base64URLEncode(array.buffer)
}

// --- Public API ---

export async function login(): Promise<void> {
  const verifier = generateCodeVerifier()
  const challenge = await generateCodeChallenge(verifier)
  const state = generateState()

  sessionStorage.setItem('pkce_verifier', verifier)
  sessionStorage.setItem('oauth_state', state)

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    scope: SCOPES,
    state,
    code_challenge: challenge,
    code_challenge_method: 'S256',
  })

  window.location.href = `${LIDS_URL}/connect/authorize?${params.toString()}`
}

export async function handleCallback(code: string, state: string): Promise<void> {
  const savedState = sessionStorage.getItem('oauth_state')
  if (state !== savedState) {
    throw new Error('OAuth state mismatch — potential CSRF attack')
  }

  const verifier = sessionStorage.getItem('pkce_verifier')
  if (!verifier) {
    throw new Error('Missing PKCE verifier')
  }

  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    code,
    code_verifier: verifier,
  })

  const resp = await fetch(`${LIDS_URL}/connect/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}))
    throw new Error((err as { error_description?: string }).error_description ?? 'Token exchange failed')
  }

  const tokens = await resp.json() as {
    access_token: string
    id_token?: string
    expires_in?: number
  }

  localStorage.setItem('modelhub_access_token', tokens.access_token)
  if (tokens.id_token) {
    localStorage.setItem('modelhub_id_token', tokens.id_token)
  }

  sessionStorage.removeItem('pkce_verifier')
  sessionStorage.removeItem('oauth_state')

  // 取 userinfo 並快取
  await fetchAndCacheUserInfo(tokens.access_token)
}

export function getToken(): string | null {
  return localStorage.getItem('modelhub_access_token')
}

export function isAuthenticated(): boolean {
  return !!getToken()
}

export function logout(): void {
  const idToken = localStorage.getItem('modelhub_id_token')
  localStorage.removeItem('modelhub_access_token')
  localStorage.removeItem('modelhub_id_token')
  localStorage.removeItem('modelhub_userinfo')

  if (idToken) {
    const params = new URLSearchParams({
      post_logout_redirect_uri: 'http://localhost:3950',
      id_token_hint: idToken,
    })
    window.location.href = `${LIDS_URL}/connect/logout?${params.toString()}`
  } else {
    window.location.href = '/'
  }
}

export interface UserInfo {
  sub: string
  name?: string
  email?: string
  preferred_username?: string
}

async function fetchAndCacheUserInfo(token: string): Promise<UserInfo> {
  const resp = await fetch(`${LIDS_URL}/connect/userinfo`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!resp.ok) throw new Error('Failed to fetch userinfo')
  const info = await resp.json() as UserInfo
  localStorage.setItem('modelhub_userinfo', JSON.stringify(info))
  return info
}

export function getCachedUserInfo(): UserInfo | null {
  const raw = localStorage.getItem('modelhub_userinfo')
  if (!raw) return null
  try {
    return JSON.parse(raw) as UserInfo
  } catch {
    return null
  }
}
