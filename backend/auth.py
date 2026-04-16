"""
auth.py — LIDS JWT 輕量認證

對 LIDS userinfo endpoint 驗證 Bearer token。
READ endpoint 免驗，WRITE endpoint 必須帶 token。
"""

import os
import httpx
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# LIDS userinfo endpoint（容器內走 squid_dev_network，本機開發走 localhost）
LIDS_USERINFO_URL = os.getenv(
    "LIDS_USERINFO_URL",
    "http://squid-lids-dev:8073/connect/userinfo",
)
# fallback for local dev outside docker
LIDS_USERINFO_URL_FALLBACK = "http://localhost:8073/connect/userinfo"

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    """
    驗證 Bearer token，回傳 userinfo dict。
    token 無效或缺少時拋 401。
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in [LIDS_USERINFO_URL, LIDS_USERINFO_URL_FALLBACK]:
            try:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 200:
                    return resp.json()
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


# 方便在 router 直接 import 用
CurrentUser = Depends(get_current_user)
