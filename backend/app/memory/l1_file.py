"""
L1 Memory Layer -- Local File Storage (替代Redis)

用JSON文件存储用户画像，零外部依赖。
路径结构：./data/profiles/{user_id}.json

适用于开发和测试环境，不需要安装Redis。
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


class L1FileProfile:
    """
    L1用户画像存储 - 本地文件版。
    
    替代L1RedisProfile，使用JSON文件存储用户画像，无需Redis。
    
    数据结构：
    - stable层: 稳定的用户画像数据（CEFR等级、学习风格等）
    - session层: 会话级临时数据（最近弱点、复习队列等）
    """

    PREFIX_PROFILE = "user:profile"
    PREFIX_STABLE = "user:profile:stable"

    def __init__(self, base_path: str = "./data/profiles"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, user_id: str) -> Path:
        return self._base / f"{user_id}_session.json"

    def _stable_path(self, user_id: str) -> Path:
        return self._base / f"{user_id}_stable.json"

    async def _read_json(self, filepath: Path) -> dict:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def _write_json(self, filepath: Path, data: dict):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def get_frozen_snapshot(self, user_id: str) -> dict:
        profile = await self._read_json(self._profile_path(user_id))
        stable = await self._read_json(self._stable_path(user_id))
        merged = {**stable, **profile}

        return {
            "cefr": merged.get("cefr", ""),
            "recent_weakness": merged.get("recent_weakness", ""),
            "preferred_topics": json.loads(merged.get("preferred_topics", "[]")),
            "learning_style": merged.get("learning_style", "balanced"),
            "communication_style": merged.get("communication_style", ""),
            "review_queue": merged.get("review_queue", ""),
            "confidence_score": float(merged.get("confidence_score", "0.5")),
            "updated_at": merged.get("updated_at", ""),
        }

    async def update_profile(self, user_id: str, profile_data: dict):
        filepath = self._profile_path(user_id)
        # 本函数是整档覆盖写：未传的字段一律回默认值。communication_style
        # 只有会话末的 profile_evolution 会算，而 main.py / session_manager
        # 的画像写入不带它——不保留就会把刚算出的标签抹成空串。
        existing = await self._read_json(filepath)
        data = {
            "cefr": profile_data.get("cefr", "B1"),
            "recent_weakness": profile_data.get("recent_weakness", ""),
            "preferred_topics": json.dumps(profile_data.get("preferred_topics", [])),
            "learning_style": profile_data.get("learning_style", "balanced"),
            "communication_style": profile_data.get(
                "communication_style", existing.get("communication_style", "")
            ),
            "review_queue": profile_data.get("review_queue", ""),
            "confidence_score": str(profile_data.get("confidence_score", 0.5)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._write_json(filepath, data)

    async def update_stable(self, user_id: str, stable_data: dict):
        filepath = self._stable_path(user_id)
        data = await self._read_json(filepath)
        data.update(stable_data)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._write_json(filepath, data)

    async def get_stable_snapshot(self, user_id: str) -> dict:
        return await self._read_json(self._stable_path(user_id))

    async def seed_profile(self, user_id: str, profile_data: dict):
        await self.update_stable(user_id, profile_data)
        await self.update_profile(user_id, profile_data)

    async def clear_user_profile(self, user_id: str) -> int:
        """彻底删除该用户的 L1 画像：session + stable 两份文件（plans/0923-04 §4.5）。

        只删这两个精确路径，不碰目录里别的文件。返回删除的文件数；
        异常原样上抛，由调用方决定删除流程是否算失败。
        """
        removed = 0
        for path in (self._profile_path(user_id), self._stable_path(user_id)):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        return removed

    async def close(self):
        pass