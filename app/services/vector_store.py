from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore(Protocol):
    async def add(self, records: list[VectorRecord]) -> None: ...

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def count(self) -> int: ...


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalVectorStore:
    """Small persistent vector store backed by a JSON file.

    Suitable for a single-process demo. The interface is intentionally the same
    as the Chroma adapter so the application can switch backends via env vars.
    """

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._records: dict[str, VectorRecord] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in payload.get("records", []):
            record = VectorRecord(
                id=item["id"],
                text=item["text"],
                embedding=[float(value) for value in item["embedding"]],
                metadata=item.get("metadata", {}),
            )
            self._records[record.id] = record

    async def _persist(self) -> None:
        payload = {
            "version": 1,
            "records": [
                {
                    "id": record.id,
                    "text": record.text,
                    "embedding": record.embedding,
                    "metadata": record.metadata,
                }
                for record in self._records.values()
            ],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.index_path)

    async def add(self, records: list[VectorRecord]) -> None:
        async with self._lock:
            for record in records:
                self._records[record.id] = record
            await self._persist()

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        async with self._lock:
            records = list(self._records.values())
        hits = [
            SearchHit(
                id=record.id,
                text=record.text,
                metadata=record.metadata,
                score=_cosine(embedding, record.embedding),
            )
            for record in records
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, top_k)]

    async def delete_document(self, document_id: str) -> None:
        async with self._lock:
            self._records = {
                key: record
                for key, record in self._records.items()
                if record.metadata.get("document_id") != document_id
            }
            await self._persist()

    async def count(self) -> int:
        async with self._lock:
            return len(self._records)


class ChromaVectorStore:
    """Optional ChromaDB adapter.

    Install `requirements-vector.txt` and set VECTOR_BACKEND=chroma to enable it.
    Chroma's client is synchronous, so calls are moved to a worker thread.
    """

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "使用 Chroma 需要先安装 requirements-vector.txt"
            ) from exc

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        def _add() -> None:
            self._collection.add(
                ids=[record.id for record in records],
                documents=[record.text for record in records],
                embeddings=[record.embedding for record in records],
                metadatas=[record.metadata for record in records],
            )

        await asyncio.to_thread(_add)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        def _query() -> dict[str, Any]:
            return self._collection.query(
                query_embeddings=[embedding],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )

        result = await asyncio.to_thread(_query)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchHit(
                id=item_id,
                text=document,
                metadata=metadata or {},
                score=1.0 - float(distance),
            )
            for item_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(
            self._collection.delete,
            where={"document_id": document_id},
        )

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)


def create_vector_store(settings: Any) -> VectorStore:
    if settings.vector_backend == "chroma":
        return ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    if settings.vector_backend != "local":
        raise ValueError(f"未知的 VECTOR_BACKEND: {settings.vector_backend}")
    return LocalVectorStore(settings.index_path)
