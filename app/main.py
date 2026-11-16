from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .services.agent import DocumentAgent
from .services.documents import UnsupportedDocumentError
from .services.knowledge_base import DocumentTooLargeError, KnowledgeBase
from .services.ollama import OllamaClient
from .services.reranker import LexicalReranker
from .services.sessions import MemorySessionStore, RedisSessionStore
from .services.vector_store import create_vector_store

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.ollama = OllamaClient(
            settings.ollama_base_url,
            settings.chat_model,
            settings.embedding_model,
            settings.request_timeout,
            settings.ollama_num_ctx,
        )
        self.vector_store = create_vector_store(settings)
        self.knowledge_base = KnowledgeBase(
            upload_dir=settings.upload_dir,
            registry_path=settings.registry_path,
            vector_store=self.vector_store,
            ollama=self.ollama,
            reranker=LexicalReranker(),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_n=settings.rerank_top_n,
            max_file_mb=settings.max_file_mb,
        )
        self.sessions = MemorySessionStore(settings.session_ttl_seconds)
        self.agent = DocumentAgent(
            self.ollama,
            self.knowledge_base,
            self.sessions,
            settings.history_messages,
        )

    async def startup(self) -> None:
        if self.settings.redis_url:
            candidate = RedisSessionStore.from_url(
                self.settings.redis_url, self.settings.session_ttl_seconds
            )
            try:
                await candidate._redis.ping()
            except Exception as exc:
                logger.warning("Redis unavailable, using in-memory sessions: %s", exc)
                await candidate.close()
            else:
                self.sessions = candidate
                self.agent.sessions = candidate

    async def close(self) -> None:
        await self.ollama.close()
        await self.sessions.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container(runtime_settings)
        app.state.container = container
        await container.startup()
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(
        title="RAG Document Agent",
        version="1.0.0",
        description="A local-first RAG question answering service with Ollama.",
        lifespan=lifespan,
    )
    if runtime_settings.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        container: Container = request.app.state.container
        return {
            "status": "ok",
            "chat_model": container.settings.chat_model,
            "embedding_model": container.settings.embedding_model,
            "vector_backend": container.settings.vector_backend,
            "documents": len(container.knowledge_base.list_documents()),
            "vectors": await container.vector_store.count(),
        }

    @app.get("/api/documents")
    async def list_documents(request: Request) -> dict[str, Any]:
        container: Container = request.app.state.container
        return {
            "documents": [
                {
                    "id": item.id,
                    "filename": item.filename,
                    "size": item.size,
                    "chunks": item.chunks,
                    "created_at": item.created_at,
                }
                for item in container.knowledge_base.list_documents()
            ]
        }

    @app.post("/api/documents", status_code=201)
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile, File(description="PDF、TXT 或 Markdown 文件")],
    ) -> dict[str, Any]:
        container: Container = request.app.state.container
        content = await file.read()
        try:
            document = await container.knowledge_base.ingest(file.filename or "document", content)
        except (DocumentTooLargeError, UnsupportedDocumentError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to ingest document")
            raise HTTPException(
                status_code=502,
                detail=f"文档处理失败，请检查 Ollama 模型是否可用：{exc}",
            ) from exc
        return {
            "id": document.id,
            "filename": document.filename,
            "size": document.size,
            "chunks": document.chunks,
            "created_at": document.created_at,
        }

    @app.delete("/api/documents/{document_id}")
    async def delete_document(request: Request, document_id: str) -> dict[str, bool]:
        container: Container = request.app.state.container
        deleted = await container.knowledge_base.delete(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"deleted": True}

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        container: Container = request.app.state.container

        async def event_stream() -> AsyncIterator[str]:
            async for event in container.agent.stream(body.question, body.session_id):
                data = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {data}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/api/sessions/{session_id}")
    async def clear_session(request: Request, session_id: str) -> dict[str, bool]:
        container: Container = request.app.state.container
        await container.sessions.clear(session_id)
        return {"cleared": True}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(UnsupportedDocumentError)
    async def unsupported_document_handler(
        request: Request, exc: UnsupportedDocumentError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
