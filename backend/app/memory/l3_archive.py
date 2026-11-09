"""
L3 Archive Layer -- 可插拔的会话归档存储

本层提供可插拔的对象存储后端，支持云原生部署和水平扩展。

参考架构：
- Halldyll Memory Model: 云原生部署（Kubernetes/RunPods），支持水平扩展
- MemoryOS: L3层采用对象存储是生产环境的标配

后端类型：
1. LocalFileBackend - 本地文件系统（开发/测试环境）
2. S3Backend - S3/MinIO 对象存储（生产环境）

配置方式：通过 ARCHIVE_BACKEND 环境变量指定
- ARCHIVE_BACKEND=local（默认）
- ARCHIVE_BACKEND=s3
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path


class ArchiveBackend:
    """
    可插拔的归档后端抽象基类。

    所有后端实现必须继承此类并实现以下方法：
    - archive(): 归档会话数据
    - retrieve(): 检索已归档的会话
    - list_sessions(): 列出用户的所有会话
    """

    def archive(
        self,
        user_id: str,
        session_id: str,
        transcript: list[dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        归档会话数据。

        Args:
            user_id: 用户ID
            session_id: 会话ID
            transcript: 对话记录
            metadata: 元数据

        Returns:
            归档路径/键
        """
        raise NotImplementedError

    def retrieve(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        检索已归档的会话。

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            会话数据（包含 transcript 和 metadata），未找到返回 None
        """
        raise NotImplementedError

    def list_sessions(self, user_id: str) -> list[Dict[str, str]]:
        """
        列出用户的所有归档会话。

        Args:
            user_id: 用户ID

        Returns:
            会话列表，每个元素包含 session_id 和 archived_at
        """
        raise NotImplementedError

    def delete_user_data(self, user_id: str) -> int:
        raise NotImplementedError

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        删除单个会话的归档。返回是否真的删掉了东西（幂等：不存在返回 False）。

        删一盘会话时必须连带调用 —— 数据库行清了而磁盘/对象存储里还躺着原始
        转写，等于没删干净。
        """
        raise NotImplementedError


class LocalFileBackend(ArchiveBackend):
    """
    本地文件系统归档后端。

    路径结构：{base_path}/{user_id}/{year}/{month}/{session_id}.json

    适用于开发和测试环境。
    """

    def __init__(self, base_path: str = "./data/archives"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _session_path(self, user_id: str, session_id: str) -> Path:
        now = datetime.now(timezone.utc)
        return self._base / user_id / str(now.year) / f"{now.month:02d}" / f"{session_id}.json"

    def _get_existing_path(self, user_id: str, session_id: str) -> Optional[Path]:
        if user_id:
            base = self._base / user_id
            if base.exists():
                for json_file in base.rglob(f"{session_id}.json"):
                    return json_file
        for json_file in self._base.rglob(f"{session_id}.json"):
            return json_file
        return None

    def archive(
        self,
        user_id: str,
        session_id: str,
        transcript: list[dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        filepath = self._session_path(user_id, session_id)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        archive = {
            "session_id": session_id,
            "user_id": user_id,
            "transcript": transcript,
            "metadata": metadata or {},
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(archive, f, ensure_ascii=False, indent=2)
        return str(filepath)

    def retrieve(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        filepath = self._get_existing_path(user_id, session_id)
        if filepath and filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_sessions(self, user_id: str) -> list[Dict[str, str]]:
        sessions = []
        base = self._base / user_id
        if not base.exists():
            return sessions
        for json_file in sorted(base.rglob("*.json"), reverse=True):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", json_file.stem),
                        "archived_at": data.get("archived_at", ""),
                    })
            except (json.JSONDecodeError, OSError):
                continue
        return sessions

    def delete_user_data(self, user_id: str) -> int:
        # Reject path-shaped identifiers before composing a filesystem path.
        if not user_id or Path(user_id).name != user_id or user_id in {".", ".."}:
            raise ValueError("invalid user_id")
        target = (self._base / user_id).resolve()
        base = self._base.resolve()
        if base not in target.parents or not target.exists():
            return 0
        count = sum(1 for _ in target.rglob("*.json"))
        shutil.rmtree(target)
        return count

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除一盘会话的原始转写 JSON（按 session_id 递归找，找到就删）。

        两个标识都必须做路径形状校验：它们会被拼进文件系统路径，
        形如 "../.." 的值可以直接越出归档根目录。
        """
        for value in (user_id, session_id):
            if not value or Path(value).name != value or value in {".", ".."}:
                raise ValueError("invalid identifier")
        target = self._get_existing_path(user_id, session_id)
        if not target or not target.exists():
            return False
        target.unlink()
        return True

    def cleanup_old_archives(self, max_age_hours: int = 24) -> int:
        """自动清理超过 N 小时的旧测试会话归档，防止磁盘占用过大"""
        removed_count = 0
        now = datetime.now(timezone.utc)
        for json_file in self._base.rglob("*.json"):
            try:
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                if (now - mtime).total_seconds() > max_age_hours * 3600:
                    json_file.unlink()
                    removed_count += 1
            except Exception:
                continue
        return removed_count


class S3Backend(ArchiveBackend):
    """
    S3/MinIO 对象存储归档后端。

    键结构：archives/{user_id}/{year}/{month}/{session_id}.json

    适用于生产环境，支持水平扩展和高可用性。

    配置环境变量：
    - S3_ENDPOINT_URL: S3 端点 URL（可选，用于 MinIO 等兼容服务）
    - S3_BUCKET_NAME: S3 存储桶名称（默认：llm-os-archives）
    - AWS_ACCESS_KEY_ID: AWS 访问密钥 ID
    - AWS_SECRET_ACCESS_KEY: AWS 秘密访问密钥
    """

    def __init__(self):
        self._bucket_name = os.getenv("S3_BUCKET_NAME", "llm-os-archives")
        self._endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import boto3
                self._session = boto3.session.Session()
            except ImportError:
                raise RuntimeError(
                    "boto3 is required for S3Backend. Install with: pip install boto3"
                )
        return self._session

    def _get_client(self):
        session = self._get_session()
        config = {}
        if self._endpoint_url:
            config["endpoint_url"] = self._endpoint_url
        return session.client("s3", **config)

    def _object_key(self, user_id: str, session_id: str) -> str:
        now = datetime.now(timezone.utc)
        return f"archives/{user_id}/{now.year}/{now.month:02d}/{session_id}.json"

    def _parse_object_key(self, key: str) -> Optional[Dict[str, str]]:
        """解析对象键，提取 user_id 和 session_id"""
        parts = key.strip("/").split("/")
        if len(parts) >= 5 and parts[0] == "archives":
            return {
                "user_id": parts[1],
                "year": parts[2],
                "month": parts[3],
                "session_id": parts[4].replace(".json", ""),
            }
        return None

    def archive(
        self,
        user_id: str,
        session_id: str,
        transcript: list[dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._get_client()
        key = self._object_key(user_id, session_id)
        archive = {
            "session_id": session_id,
            "user_id": user_id,
            "transcript": transcript,
            "metadata": metadata or {},
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        client.put_object(
            Bucket=self._bucket_name,
            Key=key,
            Body=json.dumps(archive, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self._bucket_name}/{key}"

    def retrieve(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        prefix = f"archives/{user_id}/"

        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(f"/{session_id}.json"):
                        response = client.get_object(Bucket=self._bucket_name, Key=obj["Key"])
                        content = response["Body"].read().decode("utf-8")
                        return json.loads(content)
            return None
        except client.exceptions.NoSuchBucket:
            return None
        except client.exceptions.NoSuchKey:
            return None

    def list_sessions(self, user_id: str) -> list[Dict[str, str]]:
        client = self._get_client()
        prefix = f"archives/{user_id}/"
        sessions = []

        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    parsed = self._parse_object_key(obj["Key"])
                    if parsed:
                        sessions.append({
                            "session_id": parsed["session_id"],
                            "archived_at": obj["LastModified"].isoformat(),
                        })
            return sorted(sessions, key=lambda x: x["archived_at"], reverse=True)
        except client.exceptions.NoSuchBucket:
            return []

    def delete_user_data(self, user_id: str) -> int:
        client = self._get_client()
        prefix = f"archives/{user_id}/"
        deleted = 0
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            for index in range(0, len(keys), 1000):
                batch = keys[index:index + 1000]
                if batch:
                    client.delete_objects(Bucket=self._bucket_name, Delete={"Objects": batch})
                    deleted += len(batch)
        return deleted

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除一盘会话的归档对象。键里带年月，所以按前缀列表找后缀匹配。"""
        client = self._get_client()
        prefix = f"archives/{user_id}/"
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(f"/{session_id}.json"):
                        client.delete_object(Bucket=self._bucket_name, Key=obj["Key"])
                        return True
        except Exception:
            return False
        return False


class L3Archive:
    """
    L3 归档管理 - 统一的归档入口。

    根据环境变量 ARCHIVE_BACKEND 自动选择后端实现：
    - local（默认）: LocalFileBackend
    - s3: S3Backend

    用法：
        archive = L3Archive()
        archive.archive(user_id, session_id, transcript, metadata)
        session = archive.retrieve(user_id, session_id)
    """

    def __init__(self, backend: Optional[ArchiveBackend] = None):
        """
        初始化归档管理器。

        Args:
            backend: 自定义后端实现（可选）。如果不提供，将根据环境变量选择。
        """
        if backend:
            self._backend = backend
        else:
            backend_type = os.getenv("ARCHIVE_BACKEND", "local").lower()
            if backend_type == "s3":
                self._backend = S3Backend()
            else:
                self._backend = LocalFileBackend()

    async def archive_session(
        self,
        user_id: str,
        session_id: str,
        transcript: list[dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        归档会话。

        Args:
            user_id: 用户ID
            session_id: 会话ID
            transcript: 对话记录
            metadata: 元数据

        Returns:
            归档路径/键
        """
        return self._backend.archive(user_id, session_id, transcript, metadata)

    def archive(
        self,
        user_id: str,
        session_id: str,
        transcript: list[dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """同步归档别名"""
        return self._backend.archive(user_id, session_id, transcript, metadata)

    async def retrieve_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        检索会话。

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            会话数据（包含 transcript 和 metadata），未找到返回 None
        """
        return self._backend.retrieve(user_id, session_id)

    async def list_sessions(self, user_id: str) -> list[Dict[str, str]]:
        """
        列出用户的所有归档会话。

        Args:
            user_id: 用户ID

        Returns:
            会话列表，每个元素包含 session_id 和 archived_at
        """
        return self._backend.list_sessions(user_id)

    def cleanup_old_archives(self, max_age_hours: int = 24) -> int:
        """清理超过 N 小时的旧归档"""
        if hasattr(self._backend, "cleanup_old_archives"):
            return self._backend.cleanup_old_archives(max_age_hours=max_age_hours)
        return 0

    async def delete_user_data(self, user_id: str) -> int:
        return self._backend.delete_user_data(user_id)

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除单个会话的归档（随一盘磁带的删除一起调用）。"""
        return self._backend.delete_session(user_id, session_id)
