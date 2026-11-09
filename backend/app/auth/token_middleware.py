"""
Token 校验中间件（FastAPI Dependency）。

使用方式：
    from auth import require_user
    @app.get("/some/protected/endpoint")
    async def handler(user_id: str = Depends(require_user)):
        ...

设计原则：
- 不强制所有现有端点接入。现有 web 端用 "frontend-user" 不带 token，照常工作。
- 只有小程序前端调用的端点显式 Depends(require_user) 时才校验。
- 失败统一返回 401 + JSON {detail, code}，便于小程序前端识别跳登录。
"""

from __future__ import annotations

from typing import Optional
from fastapi import Depends, HTTPException, Header, status

from .token_store import get_store


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """从 'Bearer xxx' 提取 token。"""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def require_user(authorization: Optional[str] = Header(None)) -> str:
    """
    强校验：必须有合法 Bearer token，返回 user_id。
    无 token / 失效 / 格式错 → 401。
    """
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_token", "message": "未携带 token，请先登录"},
        )
    payload = get_store().validate(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "token 已失效，请重新登录"},
        )
    return payload["user_id"]


def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """
    弱校验：有合法 token 用 token 对应的 user_id；
    无 token 时回退到 "frontend-user"（兼容现有 web 前端，逐步迁移用）。
    """
    token = _extract_bearer(authorization)
    if token:
        payload = get_store().validate(token)
        if payload:
            return payload["user_id"]
    return "frontend-user"
