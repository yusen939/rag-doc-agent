from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .knowledge_base import KnowledgeBase, RetrievedChunk
from .ollama import OllamaClient, OllamaError
from .sessions import MemorySessionStore, RedisSessionStore
from .tools import DocumentToolbox, ToolExecutionError

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。规则：
1. 回答知识库相关问题时，必须先调用 search_documents 工具，不得凭记忆编造文档内容。
2. 回答只能依据工具返回的资料。资料不足时明确说“文档中未找到相关信息”。
3. 引用资料时使用 [1]、[2] 这样的编号，编号必须对应当前工具结果中的 citation。
4. 涉及算术时调用 calculator，不要心算。
5. 使用简洁、准确的中文回答，先给结论，再给依据。
"""


class DocumentAgent:
    def __init__(
        self,
        ollama: OllamaClient,
        knowledge_base: KnowledgeBase,
        sessions: MemorySessionStore | RedisSessionStore,
        history_messages: int = 10,
    ) -> None:
        self.ollama = ollama
        self.knowledge_base = knowledge_base
        self.sessions = sessions
        self.history_messages = history_messages
        self.toolbox = DocumentToolbox(knowledge_base)

    async def stream(
        self, question: str, session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        question = question.strip()
        if not question:
            yield {"event": "error", "data": {"message": "问题不能为空"}}
            return

        history = await self.sessions.get(session_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-self.history_messages :],
            {"role": "user", "content": question},
        ]

        collected_sources: dict[str, RetrievedChunk] = {}
        answer = ""
        used_tool = False

        try:
            yield {"event": "status", "data": {"message": "正在分析问题"}}

            decision: dict[str, Any] = {}
            for _ in range(2):
                decision = await self.ollama.chat(
                    messages, tools=self.toolbox.definitions
                )
                tool_calls = decision.get("tool_calls") or []
                if not tool_calls:
                    break

                used_tool = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": decision.get("content", "") or "",
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name = str(function.get("name", ""))
                    arguments = self._parse_arguments(function.get("arguments"))
                    yield {
                        "event": "status",
                        "data": {"message": self._tool_status(name)},
                    }
                    sources: list[RetrievedChunk] = []
                    try:
                        result = await self.toolbox.execute(name, arguments)
                        content = result.content
                        sources = result.sources
                    except (ToolExecutionError, OllamaError) as exc:
                        content = f"工具调用失败：{exc}"
                    for source in sources:
                        collected_sources[source.id] = source
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": content,
                        }
                    )

                if collected_sources:
                    yield {
                        "event": "sources",
                        "data": self._serialize_sources(collected_sources.values()),
                    }

            if decision.get("content") and not used_tool:
                if await self.knowledge_base.vector_store.count() > 0:
                    # Never trust a document answer produced without retrieval.
                    # Re-run the answer with retrieved context before returning it.
                    async for event in self._fallback_answer(
                        question,
                        session_id,
                        history,
                        status_message="正在核对文档依据",
                    ):
                        yield event
                    return
                answer = str(decision["content"])
                for index in range(0, len(answer), 6):
                    yield {
                        "event": "token",
                        "data": {"content": answer[index : index + 6]},
                    }
            else:
                yield {"event": "status", "data": {"message": "正在生成回答"}}
                async for chunk in self.ollama.chat_stream(messages):
                    if chunk["type"] == "token":
                        answer += chunk["content"]
                        yield {
                            "event": "token",
                            "data": {"content": chunk["content"]},
                        }

            if not answer.strip() and decision.get("content"):
                answer = str(decision["content"])
                yield {"event": "token", "data": {"content": answer}}

            if not answer.strip():
                answer = "模型未返回有效内容，请稍后重试。"
                yield {"event": "token", "data": {"content": answer}}

            await self.sessions.append(
                session_id,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )
            yield {
                "event": "done",
                "data": {
                    "session_id": session_id,
                    "sources": self._serialize_sources(collected_sources.values()),
                },
            }
        except OllamaError as exc:
            if not used_tool and self._tool_unsupported(exc):
                async for event in self._fallback_answer(
                    question,
                    session_id,
                    history,
                    status_message="当前模型不支持工具调用，改用检索增强模式",
                ):
                    yield event
            else:
                yield {"event": "error", "data": {"message": str(exc)}}

    async def _fallback_answer(
        self,
        question: str,
        session_id: str,
        history: list[dict[str, Any]],
        status_message: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Provide retrieval-only answers for models without tool support."""
        yield {
            "event": "status",
            "data": {"message": status_message},
        }
        try:
            chunks = await self.knowledge_base.search(question, top_k=4)
        except OllamaError as exc:
            yield {"event": "error", "data": {"message": str(exc)}}
            return
        if chunks:
            context = "\n\n".join(
                f"[{index}] 文件：{chunk.filename}"
                f"{f'，第 {chunk.page} 页' if chunk.page else ''}\n{chunk.text}"
                for index, chunk in enumerate(chunks, start=1)
            )
        else:
            context = "知识库中没有检索到相关内容。"

        fallback_system = (
            "你是文档问答助手。只能依据下方检索资料回答；"
            "资料不足时明确说明。引用时使用 [1]、[2] 编号。"
        )
        fallback_messages = [
            {"role": "system", "content": fallback_system},
            *history[-self.history_messages :],
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索资料：\n{context}",
            },
        ]
        yield {
            "event": "sources",
            "data": self._serialize_sources(chunks),
        }
        answer = ""
        async for chunk in self.ollama.chat_stream(fallback_messages):
            if chunk["type"] != "token":
                continue
            answer += chunk["content"]
            yield {"event": "token", "data": {"content": chunk["content"]}}
        if not answer.strip():
            answer = "模型未返回有效内容，请稍后重试。"
            yield {"event": "token", "data": {"content": answer}}
        await self.sessions.append(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )
        yield {
            "event": "done",
            "data": {
                "session_id": session_id,
                "sources": self._serialize_sources(chunks),
            },
        }

    @staticmethod
    def _tool_unsupported(error: OllamaError) -> bool:
        message = str(error).lower()
        return "tool" in message and (
            "not support" in message
            or "does not support" in message
            or "unsupported" in message
        )

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _tool_status(name: str) -> str:
        return {
            "search_documents": "正在检索知识库",
            "list_documents": "正在读取文档列表",
            "calculator": "正在计算结果",
        }.get(name, "正在调用工具")

    @staticmethod
    def _serialize_sources(sources: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": source.id,
                "filename": source.filename,
                "page": source.page,
                "score": source.score,
                "excerpt": source.text[:180],
            }
            for source in sources
        ]
