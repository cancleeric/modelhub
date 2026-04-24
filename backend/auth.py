"""
auth.py — LIDS JWT + API Key 認證

- Bearer token → 打 LIDS userinfo 驗證
- X-Api-Key → 先查 DB (api_keys table，Sprint 7.1)，找不到 fallback env bootstrap key
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Optional

import httpx
from cachetools import TTLCache
from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("modelhub.auth")

# P1-2: module-level TTLCache，60 秒有效期，最多 512 個 token
_TOKEN_CACHE: TTLCache = TTLCache(maxsize=512, ttl=60)

_BOOTSTRAP_KEY_RAW = os.getenv("MODELHUB_API_KEY")

if not _BOOTSTRAP_KEY_RAW:
    logger.warning(
        "MODELHUB_API_KEY environment variable is not set. "
        "Bootstrap API key authentication is DISABLED. "
        "Only DB-backed API keys will be accepted."
    )

# 明確拒絕已知的預設 dev key（即使 env 被設成這個值）
_KNOWN_INSECURE_KEYS = {"modelhub-dev-key-2026"}


def _verify_api_key_db(x_api_key: str) -> Optional[dict]:  # P3-26: dict | None → Optional[dict]
    """查 DB api_keys table，成功回 {sub,name}，失敗回 None。"""
    if not x_api_key:
        return None
    try:
        from models import SessionLocal, ApiKey
    except Exception:
        return None
    db = SessionLocal()
    try:
        row = db.query(ApiKey).filter(ApiKey.key == x_api_key, ApiKey.disabled == False).first()  # noqa: E712
        if not row:
            return None
        row.last_used_at = datetime.utcnow()
        db.commit()
        return {"sub": f"api_key:{row.id}", "name": row.name}
    finally:
        db.close()


def verify_api_key(x_api_key: str) -> Optional[dict]:  # P3-26: dict | None → Optional[dict]
    """回 userinfo dict 或 None。先查 DB，fallback env bootstrap key（拒絕 insecure 預設值）。"""
    if not x_api_key:
        return None
    # 拒絕已知不安全預設值
    if x_api_key in _KNOWN_INSECURE_KEYS:
        logger.warning(
            "Rejected authentication attempt using known insecure default API key. "
            "Set MODELHUB_API_KEY to a secure value."
        )
        return None
    hit = _verify_api_key_db(x_api_key)
    if hit:
        return hit
    # fallback: env bootstrap key（僅在有設定且非 insecure 時才允許）
    if _BOOTSTRAP_KEY_RAW and x_api_key == _BOOTSTRAP_KEY_RAW:
        return {"sub": "api_key:bootstrap", "name": "bootstrap"}
    return None


async def get_api_key(x_api_key: str = Header(None)) -> str:
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# LIDS userinfo endpoint（容器內走 squid_dev_network，本機開發走 localhost）
LIDS_USERINFO_URL = os.getenv(
    "LIDS_USERINFO_URL",
    "http://squid-lids-dev:8073/connect/userinfo",
)
# fallback for local dev outside docker
LIDS_USERINFO_URL_FALLBACK = "http://localhost:8073/connect/userinfo"

bearer_scheme = HTTPBearer(auto_error=False)


async def _verify_lids_token(token: str) -> dict:
    """
    P1-4: 共用 LIDS token 驗證邏輯。
    成功回傳 userinfo dict，失敗拋 HTTPException(401)。
    P1-2: 加入 TTLCache，同一 token 60 秒內不重複打 LIDS。
    """
    key = hashlib.sha256(token.encode()).hexdigest()[:32]
    if key in _TOKEN_CACHE:
        logger.debug("_verify_lids_token: cache hit key=%s", key)
        return _TOKEN_CACHE[key]

    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in [LIDS_USERINFO_URL, LIDS_USERINFO_URL_FALLBACK]:
            try:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 200:
                    userinfo = resp.json()
                    _TOKEN_CACHE[key] = userinfo
                    return userinfo
                # 401/403 直接拒絕，不再 fallback
                if resp.status_code in (401, 403):
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
            except httpx.ConnectError:
                continue  # 嘗試 fallback URL
            except HTTPException:
                raise
            except Exception:
                continue
    raise HTTPException(status_code=401, detail="Unable to verify token (LIDS unreachable)")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    """
    驗證 Bearer token，回傳 userinfo dict。
    token 無效或缺少時拋 401。
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return await _verify_lids_token(credentials.credentials)


async def get_current_user_or_api_key(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    x_api_key: str = Header(None),
) -> dict:
    """
    接受 LIDS Bearer token 或 API Key (X-Api-Key header)。
    機器對機器呼叫（如 AICAD pipeline）用 API Key 即可。
    回傳 userinfo dict 或 {"sub": "api_key", "name": "service_account"}。
    """
    # 優先嘗試 API Key（DB → bootstrap env）
    if x_api_key is not None:
        hit = verify_api_key(x_api_key)
        if hit:
            return hit
        raise HTTPException(status_code=401, detail="Invalid API key")

    # fallback: LIDS Bearer token
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return await _verify_lids_token(credentials.credentials)


# 方便在 router 直接 import 用
CurrentUser = Depends(get_current_user)
CurrentUserOrApiKey = Depends(get_current_user_or_api_key)

# P3-3: claim-based role check
# MODELHUB_ROLE_CLAIM: userinfo 中的 claim key（預設 modelhub_role）
# SKIP_ROLE_CHECK: 設為 "true" 時跳過 role 檢查（dev 環境 LIDS 未設 claim 時用）
_ROLE_CLAIM_KEY = os.getenv("MODELHUB_ROLE_CLAIM", "modelhub_role")
_SKIP_ROLE_CHECK = os.getenv("SKIP_ROLE_CHECK", "false").lower() == "true"


def assert_role(role: str, current_user: dict) -> None:
    """
    P2-24: 同步版 role check，供 endpoint 函式體內（非 Depends）使用。
    邏輯與 require_role Depends 完全一致：
      - SKIP_ROLE_CHECK=true 時略過
      - API Key 使用者（sub 以 "api_key:" 開頭）略過
      - 否則要求 MODELHUB_ROLE_CLAIM == role
    """
    if _SKIP_ROLE_CHECK:
        return
    is_api_key = str((current_user or {}).get("sub", "")).startswith("api_key:")
    if is_api_key:
        return
    user_role = (current_user or {}).get(_ROLE_CLAIM_KEY)
    if user_role != role:
        raise HTTPException(
            status_code=403,
            detail=f"approve requires role '{role}', current='{user_role}'",
        )


def require_role(role: str):
    """
    Dependency factory：要求 userinfo 中 MODELHUB_ROLE_CLAIM == role。
    SKIP_ROLE_CHECK=true 時略過（dev 環境用）。
    API Key 使用者略過 role 檢查（機器對機器）。
    """
    async def _check(
        credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
        x_api_key: str = Header(None),
    ) -> dict:
        if _SKIP_ROLE_CHECK:
            return {}
        # API Key 使用者略過 role 檢查
        if x_api_key is not None:
            hit = verify_api_key(x_api_key)
            if hit:
                return hit
            raise HTTPException(status_code=401, detail="Invalid API key")
        if credentials is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        # 取 userinfo（P1-4: 呼叫共用 _verify_lids_token）
        token = credentials.credentials
        userinfo = await _verify_lids_token(token)
        user_role = userinfo.get(_ROLE_CLAIM_KEY)
        if user_role != role:
            raise HTTPException(
                status_code=403,
                detail=f"Requires role '{role}', current='{user_role}'",
            )
        return userinfo
    return Depends(_check)
