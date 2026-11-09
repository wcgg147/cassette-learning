import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

try:
    from ..memory.provider import maybe_await
except ImportError:
    from memory.provider import maybe_await


@dataclass
class StyleParameters:
    """用户的说话风格参数"""
    user_id: str = ""

    # 语气相关
    formality: float = 0.5       # 0=非常随意, 1=非常正式
    warmth: float = 0.5          # 0=冷淡, 1=热情
    humor: float = 0.3           # 0=严肃, 1=幽默
    directness: float = 0.5      # 0=委婉, 1=直接

    # 教学相关
    correction_style: str = "recast"  # recast, explicit, elicitation
    error_tolerance: float = 0.5      # 0=立即纠错, 1=不纠错
    challenge_level: float = 0.5      # 0=简单问题, 1=困难问题

    # 呈现相关
    preferred_length: int = 30        # 字符数偏好
    language_mix: float = 0.5         # 0=全母语, 1=全目标语言
    example_frequency: float = 0.5    # 0=少给例子, 1=多给例子

    # 策略相关
    opening_style: str = "suggest"    # suggest, ask, state
    followup_style: str = "expand"    # expand, challenge, support

    # 元数据
    version: int = 0
    updated_at: float = 0.0
    feedback_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleParameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "StyleParameters":
        return cls.from_dict(json.loads(json_str))


class ParameterStore:
    """风格参数存储（委托给 L2 后端，SQLite / Postgres 都可）"""

    def __init__(self, l2_memory=None):
        self.l2 = l2_memory

    async def get(self, user_id: str) -> StyleParameters:
        """获取用户风格参数"""
        if not self.l2:
            return StyleParameters(user_id=user_id)

        try:
            # 走 L2 后端的方法而不是直连 sqlite3 —— 直连 _get_conn() 在
            # Postgres 后端上会抛 AttributeError，然后被 except 吞掉，
            # 结果就是「风格参数永远读不出来」，且不报错。
            data = await maybe_await(self.l2.get_style_parameters(user_id))
            if data:
                clean = {k: v for k, v in data.items() if k != "user_id"}
                return StyleParameters.from_dict({"user_id": user_id, **clean})
        except Exception:
            pass

        return StyleParameters(user_id=user_id)

    async def save(self, params: StyleParameters) -> bool:
        """保存用户风格参数"""
        if not self.l2:
            return False

        try:
            data = params.to_dict()
            data.pop("user_id", None)
            return bool(await maybe_await(
                self.l2.save_style_parameters(params.user_id, data)
            ))
        except Exception:
            return False

    async def get_history(self, user_id: str, limit: int = 10) -> list:
        """获取参数变更历史（从 L3 archive 读取）"""
        return []
