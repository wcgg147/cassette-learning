"""
微信小程序登录鉴权模块。

提供：
- POST /auth/wx_login    微信 code 换 token
- GET  /auth/me          校验 token，返回当前用户
- 可选中间件 require_user  注入 user_id 到请求上下文

向后兼容：现有 web 端使用硬编码 "frontend-user"，不经过本模块，互不干扰。
"""

from .wx_login import router as auth_router
from .token_middleware import require_user, get_current_user_id

__all__ = ["auth_router", "require_user", "get_current_user_id"]
