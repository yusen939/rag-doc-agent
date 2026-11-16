import asyncio

from app.services.agent import DocumentAgent
from app.services.knowledge_base import RetrievedChunk
from app.services.sessions import MemorySessionStore


class FakeOllama:
    async def chat(self, messages, tools=None):
        return {"content": "模型没有检索就给出的错误答案", "tool_calls": []}

    async def chat_stream(self, messages):
        assert "检索资料" in messages[-1]["content"]
        yield {"type": "token", "content": "根据文档，标准版为 29 元。"}
        yield {"type": "done"}


class FakeVectorStore:
    async def count(self):
        return 1


class FakeKnowledgeBase:
    vector_store = FakeVectorStore()

    async def search(self, query, top_k=None):
        return [
            RetrievedChunk(
                id="doc:0",
                text="标准版价格为 29 元每人每月。",
                filename="handbook.md",
                page=None,
                score=0.9,
            )
        ]


def test_agent_never_returns_unretrieved_document_answer():
    async def scenario():
        agent = DocumentAgent(
            FakeOllama(),
            FakeKnowledgeBase(),
            MemorySessionStore(),
        )
        events = [
            event
            async for event in agent.stream("标准版多少钱？", "test-session")
        ]
        tokens = "".join(
            event["data"]["content"]
            for event in events
            if event["event"] == "token"
        )
        assert "29 元" in tokens
        assert "错误答案" not in tokens
        assert any(event["event"] == "sources" for event in events)
        assert any(event["event"] == "done" for event in events)

    asyncio.run(scenario())
