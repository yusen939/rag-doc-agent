from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout: float = 120.0,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.num_ctx = num_ctx
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            return response
        except httpx.ConnectError as exc:
            raise OllamaError(
                "无法连接 Ollama，请确认已启动 `ollama serve` 并且地址配置正确"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaError(f"Ollama 请求失败（{exc.response.status_code}）：{detail}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama 请求超时，请检查模型是否已加载") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._request(
                "POST",
                "/api/embed",
                json={"model": self.embedding_model, "input": texts},
            )
            payload = response.json()
            embeddings = payload.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except OllamaError:
            # Older Ollama versions expose /api/embeddings for a single prompt.
            pass

        embeddings: list[list[float]] = []
        for text in texts:
            response = await self._request(
                "POST",
                "/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            )
            embedding = response.json().get("embedding")
            if not embedding:
                raise OllamaError("Ollama 未返回向量，请确认 Embedding 模型已下载")
            embeddings.append(embedding)
        return embeddings

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        response = await self._request("POST", "/api/chat", json=payload)
        return response.json().get("message", {})

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools

        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = chunk.get("message", {})
                    content = message.get("content", "")
                    if content:
                        yield {"type": "token", "content": content}
                    if chunk.get("done"):
                        yield {"type": "done"}
                        break
        except httpx.ConnectError as exc:
            raise OllamaError(
                "无法连接 Ollama，请确认已启动 `ollama serve` 并且地址配置正确"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaError(f"Ollama 请求失败（{exc.response.status_code}）：{detail}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama 请求超时，请检查模型是否已加载") from exc
