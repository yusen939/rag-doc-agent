from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from redis.asyncio import Redis


class MemorySessionStore:
    def __init__(self, ttl_seconds: int = 21600) -> None:
        self._sessions: dict[str, tuple[float, list[dict[str, Any]]]] = defaultdict(
            lambda: (0.0, [])
        )
        self.ttl_seconds = ttl_seconds

    async def get(self, session_id: str) -> list[dict[str, Any]]:
        expires_at, messages = self._sessions.get(session_id, (0.0, []))
        if expires_at < time.time():
            self._sessions.pop(session_id, None)
            return []
        return list(messages)

    async def append(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        _, existing = self._sessions.get(session_id, (0.0, []))
        self._sessions[session_id] = (
            time.time() + self.ttl_seconds,
            [*existing, *messages],
        )

    async def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def close(self) -> None:
        self._sessions.clear()


class RedisSessionStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 21600) -> None:
        self._redis = redis
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int) -> "RedisSessionStore":
        return cls(Redis.from_url(url, decode_responses=True), ttl_seconds)

    def _key(self, session_id: str) -> str:
        return f"rag-doc-agent:session:{session_id}"

    async def get(self, session_id: str) -> list[dict[str, Any]]:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    async def append(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        existing = await self.get(session_id)
        payload = [*existing, *messages]
        await self._redis.set(
            self._key(session_id),
            json.dumps(payload, ensure_ascii=False),
            ex=self.ttl_seconds,
        )

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def close(self) -> None:
        await self._redis.aclose()
