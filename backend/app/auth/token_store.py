"""
Token 存储 - SQLite 实现。

复用现有 memory.db（位于 产品主文件/data/memory.db），新增一张 wx_tokens 表。
零外部依赖，与现有 L2SQLiteMemory 共享同一个数据库文件。

字段说明：
- token        本系统颁发的自定义 token（uuid4），作为后续请求的 Bearer 凭证
- openid       微信用户的 openid（来自 code2session）
- user_id      本系统内的用户标识。默认 "wx_" + openid 前 12 位；
               设备身份登录（见 wx_login.py）直接用设备 UUID，因为
               Postgres 侧的 user_id 列是 uuid 类型。
- session_key  微信会话密钥（如需调用微信其他接口会用到，暂存）
- created_at   颁发时间（ISO 字符串）
- expires_at   过期时间（ISO 字符串）

注意：session_key 有敏感性，不可下发给小程序前端。
"""

from __future__ import annotations

import sqlite3
import uuid
import os as _os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


# 默认 token 有效期 30 天（与微信 refresh_token 默认有效期对齐）
TOKEN_TTL_DAYS = 30


def _default_db_path() -> str:
    """定位 memory.db，与 L2SQLiteMemory 保持一致。"""
    api_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    product_root = _os.path.dirname(api_dir)
    return str(Path(product_root) / "data" / "memory.db")


class TokenStore:
    """token CRUD。所有方法同步，因为 SQLite 同步即可，且避免与 L2 的 async 连接冲突。"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or _default_db_path()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_table(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS wx_tokens (
                    token        TEXT PRIMARY KEY,
                    openid       TEXT NOT NULL,
                    user_id      TEXT NOT NULL,
                    session_key  TEXT,
                    created_at   TEXT NOT NULL,
                    expires_at   TEXT NOT NULL
                );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_wx_tokens_openid  ON wx_tokens(openid);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_wx_tokens_user_id ON wx_tokens(user_id);")

    def issue(self, openid: str, session_key: str,
              user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        为给定 openid 颁发一个新 token。
        同一 openid 可有多条有效 token（多设备登录），不主动清理。
        返回 {token, user_id, created_at, expires_at, is_new_user}。

        user_id 显式传入时优先使用（**必须是 UUID 字符串**）：
        设备身份登录（code="dev:<uuid4>"）把设备 UUID 直接当 user_id，
        因为 Postgres 侧所有 user_id 列都是 uuid 类型，而
        默认推导出的 "wx_xxx" 字符串无法写入。
        """
        if user_id is None:
            user_id = "wx_" + openid[:12] if len(openid) >= 12 else "wx_" + openid
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=TOKEN_TTL_DAYS)
        token = uuid.uuid4().hex

        with self._conn() as c:
            # 是否新用户：当前 openid 是否有任何历史 token
            row = c.execute(
                "SELECT COUNT(*) AS cnt FROM wx_tokens WHERE openid = ?;",
                (openid,)
            ).fetchone()
            is_new_user = int(row["cnt"]) == 0

            c.execute(
                """INSERT INTO wx_tokens (token, openid, user_id, session_key, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?);""",
                (token, openid, user_id, session_key,
                 now.isoformat(), expires.isoformat())
            )
        return {
            "token": token,
            "user_id": user_id,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "is_new_user": is_new_user,
        }

    def validate(self, token: str) -> Optional[Dict[str, Any]]:
        """校验 token，返回 {openid, user_id} 或 None。"""
        if not token:
            return None
        with self._conn() as c:
            row = c.execute(
                """SELECT openid, user_id, expires_at FROM wx_tokens WHERE token = ?;""",
                (token,)
            ).fetchone()
        if not row:
            return None
        try:
            exp = datetime.fromisoformat(row["expires_at"])
            if exp < datetime.now(timezone.utc):
                return None
        except Exception:
            return None
        return {"openid": row["openid"], "user_id": row["user_id"]}

    def revoke(self, token: str) -> bool:
        """注销 token（登出）。"""
        with self._conn() as c:
            cur = c.execute("DELETE FROM wx_tokens WHERE token = ?;", (token,))
            return cur.rowcount > 0


# 单例（与 main.py 的全局对象风格一致）
_store: Optional[TokenStore] = None


def get_store() -> TokenStore:
    global _store
    if _store is None:
        _store = TokenStore()
    return _store
