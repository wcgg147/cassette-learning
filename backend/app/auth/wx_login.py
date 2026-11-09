"""
微信小程序登录端点。

流程：
  1. 小程序调 wx.login() 拿到临时 code
  2. POST /auth/wx_login { code } → 本端点
  3. 本端点调微信 code2session 接口换 openid + session_key
  4. 颁发本系统自定义 token，存入 SQLite
  5. 返回 { token, user_id, is_new_user }
  6. 小程序后续所有请求带 Authorization: Bearer <token>

向后兼容：
- 现有 web 前端不调用本端点，仍用硬编码 "frontend-user"，不受影响。
- 小程序用户的 user_id 格式为 "wx_<openid前12位>"，与 "frontend-user" 隔离。
- **设备身份登录**（开发模式）用 "dev:<uuid4>" 作为 code，user_id 直接取该 UUID。
  早期版本固定返回同一个 "dev_openid_001"（→ 所有用户共用一份记忆），已停用。

环境变量（在 产品主文件/api/.env 中配置）：
- WX_APPID     微信小程序 AppID
- WX_SECRET    微信小程序 AppSecret
- WX_DEV_MODE  1 = 允许设备身份登录（**只应在上线前的测试阶段开启**）
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, status
from pydantic import BaseModel

from config import (
    WX_APPID, WX_SECRET,
    WX_MINIAPP_APPID, WX_MINIAPP_SECRET,
    WX_ANDROID_APPID, WX_ANDROID_SECRET,
)
from .token_store import get_store
from .token_middleware import _extract_bearer


router = APIRouter(prefix="/auth", tags=["auth"])

# 微信 code2session 接口（小程序，官方固定地址）
_WX_CODE2SESSION_URL = "https://api.weixin.qq.com/sns/jscode2session"
# 微信移动应用 access_token 接口
_WX_APP_ACCESS_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"

# ── 设备身份登录（开发模式）──────────────────────────────────────────────
# 客户端发 code="dev:<uuid4>"，其中 uuid4 由 APP 首次启动时按
# 设备标识（ANDROID_ID）确定性派生并本地持久化 —— 因此**每台设备一个身份**，
# 同一设备卸载重装后仍是同一个人。
# 早期版本固定用 openid="dev_openid_001"，导致所有用户共用一份记忆，已停用。
DEV_CODE_PREFIX = "dev:"


def _parse_device_id(code: str) -> str:
    """
    从 code="dev:<uuid>" 中取出并校验设备 UUID。
    非法时抛 400 —— 宁可登录失败，也不能让身份退化成"所有人共用一个"。
    """
    raw = (code or "")[len(DEV_CODE_PREFIX):].strip()
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_device_id",
                    "message": '设备标识必须是合法 UUID，形如 code="dev:<uuid4>"'},
        )


class WxLoginReq(BaseModel):
    code: str
    platform: str = "miniapp"  # "miniapp" | "android"


class WxLoginResp(BaseModel):
    token: str
    user_id: str
    is_new_user: bool


def _call_wx_api(url: str) -> dict:
    """通用微信接口调用。"""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "wx_network_error",
                    "message": f"调用微信接口失败: {e.reason}"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "wx_parse_error", "message": f"解析微信响应失败: {e}"},
        )

    if "errcode" in data and data["errcode"] != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "wx_api_error",
                    "message": f"微信接口返回错误: errcode={data.get('errcode')}, "
                               f"errmsg={data.get('errmsg', '')}"},
        )
    return data


def _call_wx_code2session(code: str) -> dict:
    """
    调用微信小程序 jscode2session 接口。
    返回 {openid, session_key, [unionid]} 或抛出异常。
    """
    if not WX_MINIAPP_APPID or not WX_MINIAPP_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "wx_not_configured",
                    "message": "后端未配置微信小程序 AppID / Secret，无法登录"},
        )

    url = (
        f"{_WX_CODE2SESSION_URL}"
        f"?appid={WX_MINIAPP_APPID}&secret={WX_MINIAPP_SECRET}"
        f"&js_code={code}&grant_type=authorization_code"
    )
    data = _call_wx_api(url)
    if "openid" not in data or "session_key" not in data:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "wx_invalid_response",
                    "message": "微信响应缺少 openid 或 session_key"},
        )
    return data


def _call_wx_app_access_token(code: str) -> dict:
    """
    调用微信移动应用 access_token 接口。
    返回 {access_token, expires_in, refresh_token, openid, scope} 或抛出异常。
    """
    if not WX_ANDROID_APPID or not WX_ANDROID_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "wx_android_not_configured",
                    "message": "后端未配置微信 Android 移动应用 AppID / Secret，无法登录"},
        )

    url = (
        f"{_WX_APP_ACCESS_TOKEN_URL}"
        f"?appid={WX_ANDROID_APPID}&secret={WX_ANDROID_SECRET}"
        f"&code={code}&grant_type=authorization_code"
    )
    data = _call_wx_api(url)
    if "openid" not in data:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "wx_invalid_response",
                    "message": "微信响应缺少 openid"},
        )
    return data


@router.post("/wx_login", response_model=WxLoginResp)
def wx_login(req: WxLoginReq):
    """
    微信登录入口（小程序 / Android APP）。

    请求体:
      { "code": "...", "platform": "miniapp" | "android" }
    响应:
      { "token": "...", "user_id": "wx_xxxxxxxxxxxx", "is_new_user": true/false }

    错误码:
    - 400 wx_api_error                 code 无效或过期
    - 500 wx_not_configured            后端未配置微信 AppID/Secret
    - 500 wx_android_not_configured    后端未配置 Android 移动应用
    - 502 wx_network_error             调用微信接口网络失败

    开发模式（WX_DEV_MODE=1）:
      传 code="dev:<uuid4>" 即可跳过微信接口，直接颁发 token。
      user_id 就是那个 UUID —— **每台设备一个身份**，互不串记忆。
      生产环境切勿开启；正式上线前必须换成真实微信登录。

    错误码（开发模式）:
    - 400 bad_device_id                code 不是 "dev:<合法 UUID>"
    - 400 legacy_dev_login_disabled    旧版 code="dev"（会让所有人共用身份）
    """
    if os.getenv("WX_DEV_MODE") == "1":
        if (req.code or "").startswith(DEV_CODE_PREFIX):
            device_id = _parse_device_id(req.code)
            issued = get_store().issue(
                openid=f"dev_{device_id}",
                session_key="dev_session_key",
                user_id=device_id,
            )
            return WxLoginResp(
                token=issued["token"],
                user_id=issued["user_id"],
                is_new_user=issued["is_new_user"],
            )
        if req.code == "dev":
            # 旧版 APP 会走到这里。**故意不兼容**：那个分支固定把所有人映射到
            # 同一个 user_id，会让多用户共享记忆与画像，比登录失败更糟。
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "legacy_dev_login_disabled",
                        "message": '旧的 code="dev" 已停用（会导致所有用户共用同一份记忆），'
                                   '请更新到使用 code="dev:<设备UUID>" 的新版 APP'},
            )

    platform = req.platform.lower().strip() if req.platform else "miniapp"
    if platform == "android":
        wx_data = _call_wx_app_access_token(req.code)
        issued = get_store().issue(
            openid=wx_data["openid"],
            session_key=wx_data.get("refresh_token", ""),
        )
    else:
        wx_data = _call_wx_code2session(req.code)
        issued = get_store().issue(
            openid=wx_data["openid"],
            session_key=wx_data.get("session_key", ""),
        )
    return WxLoginResp(
        token=issued["token"],
        user_id=issued["user_id"],
        is_new_user=issued["is_new_user"],
    )


@router.get("/me")
def me(authorization: Optional[str] = Header(None)):
    """
    校验当前 token 是否有效，返回 user_id。
    用于小程序启动时检查登录态。

    响应: { "user_id": "wx_xxxxxxxxxxxx" }
    失败: 401
    """
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_token", "message": "未携带 token"},
        )
    payload = get_store().validate(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "token 已失效"},
        )
    return {"user_id": payload["user_id"]}


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """
    注销当前 token。
    """
    token = _extract_bearer(authorization)
    if token:
        get_store().revoke(token)
    return {"status": "ok"}
