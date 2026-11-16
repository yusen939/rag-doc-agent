from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .documents import UnsupportedDocumentError, extract_segments
from .ollama import OllamaClient
from .reranker import LexicalReranker, RankedChunk
from .vector_store import VectorRecord, VectorStore


class DocumentTooLargeError(ValueError):
    pass


@dataclass(slots=True)
class DocumentInfo:
    id: str
    filename: str
    size: int
    chunks: int
    created_at: str


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    filename: str
    page: int | None
    score: float


class KnowledgeBase:
    def __init__(
        self,
        upload_dir: Path,
        registry_path: Path,
        vector_store: VectorStore,
        ollama: OllamaClient,
        reranker: LexicalReranker,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        retrieve_top_k: int = 8,
        rerank_top_n: int = 4,
        max_file_mb: int = 20,
    ) -> None:
        self.upload_dir = upload_dir
        self.registry_path = registry_path
        self.vector_store = vector_store
        self.ollama = ollama
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_n = rerank_top_n
        self.max_file_mb = max_file_mb
        self._lock = asyncio.Lock()
        self._documents: dict[str, DocumentInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in payload.get("documents", []):
            try:
                document = DocumentInfo(**item)
            except TypeError:
                continue
            self._documents[document.id] = document

    async def _persist_registry(self) -> None:
        payload = {
            "version": 1,
            "documents": [asdict(item) for item in self._documents.values()],
        }
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.registry_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.registry_path)

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(
            self._documents.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )

    async def ingest(self, filename: str, content: bytes) -> DocumentInfo:
        if len(content) > self.max_file_mb * 1024 * 1024:
            raise DocumentTooLargeError(f"文件不能超过 {self.max_file_mb} MB")
        safe_name = Path(filename or "document").name
        if not safe_name:
            raise ValueError("文件名不能为空")

        segments = extract_segments(safe_name, content)
        chunks: list[tuple[str, int | None, int]] = []
        for segment in segments:
            for text in chunk_text(
                segment.text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            ):
                chunks.append((text, segment.page, len(chunks)))
        if not chunks:
            raise UnsupportedDocumentError("文档中没有可用于检索的文本")

        document_id = uuid.uuid4().hex
        stored_name = f"{document_id}_{re.sub(r'[^A-Za-z0-9._-]+', '_', safe_name)}"
        stored_path = self.upload_dir / stored_name
        stored_path.write_bytes(content)

        try:
            embeddings = await self.ollama.embed([item[0] for item in chunks])
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding 返回数量与文本块数量不一致")

            records: list[VectorRecord] = []
            for (text, page, index), embedding in zip(chunks, embeddings):
                metadata: dict[str, Any] = {
                    "document_id": document_id,
                    "filename": safe_name,
                    "chunk_index": index,
                }
                if page is not None:
                    metadata["page"] = page
                records.append(
                    VectorRecord(
                        id=f"{document_id}:{index}",
                        text=text,
                        embedding=embedding,
                        metadata=metadata,
                    )
                )
            await self.vector_store.add(records)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        info = DocumentInfo(
            id=document_id,
            filename=safe_name,
            size=len(content),
            chunks=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        async with self._lock:
            self._documents[document_id] = info
            await self._persist_registry()
        return info

    async def delete(self, document_id: str) -> bool:
        async with self._lock:
            document = self._documents.get(document_id)
            if not document:
                return False
            await self.vector_store.delete_document(document_id)
            for path in self.upload_dir.glob(f"{document_id}_*"):
                path.unlink(missing_ok=True)
            self._documents.pop(document_id, None)
            await self._persist_registry()
            return True

    async def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        if await self.vector_store.count() == 0:
            return []
        embedding = (await self.ollama.embed([query]))[0]
        candidates = await self.vector_store.search(
            embedding, top_k or self.retrieve_top_k
        )
        ranked = self.reranker.rerank(
            query,
            [
                RankedChunk(
                    id=hit.id,
                    text=hit.text,
                    metadata=hit.metadata,
                    dense_score=hit.score,
                    rerank_score=hit.score,
                )
                for hit in candidates
            ],
            top_k or self.rerank_top_n,
        )
        return [
            RetrievedChunk(
                id=item.id,
                text=item.text,
                filename=str(item.metadata.get("filename", "未知文件")),
                page=item.metadata.get("page"),
                score=round(item.rerank_score, 4),
            )
            for item in ranked
        ]
