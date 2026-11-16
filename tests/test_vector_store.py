import asyncio

from app.services.vector_store import LocalVectorStore, VectorRecord


def test_local_vector_store_add_search_delete(tmp_path):
    async def scenario():
        store = LocalVectorStore(tmp_path / "index.json")
        await store.add(
            [
                VectorRecord(
                    id="doc:0",
                    text="退款政策",
                    embedding=[1.0, 0.0],
                    metadata={"document_id": "doc", "filename": "a.md"},
                ),
                VectorRecord(
                    id="doc:1",
                    text="部署说明",
                    embedding=[0.0, 1.0],
                    metadata={"document_id": "doc", "filename": "a.md"},
                ),
            ]
        )
        hits = await store.search([0.9, 0.1], top_k=1)
        assert hits[0].id == "doc:0"
        assert await store.count() == 2

        reloaded = LocalVectorStore(tmp_path / "index.json")
        assert await reloaded.count() == 2

        await reloaded.delete_document("doc")
        assert await reloaded.count() == 0

    asyncio.run(scenario())
