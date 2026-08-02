from edupaper_agent.pdf import Chunk
from edupaper_agent.storage import ChromaStore


def test_store_add_query_and_delete(test_settings) -> None:
    store = ChromaStore(
        persist_dir=test_settings.chroma_dir,
        collection_name=test_settings.collection_name,
    )
    chunks = [
        Chunk(
            id="doc-1:p1:c0",
            text="形成性评价能够为学生提供及时反馈。",
            metadata={"document_id": "doc-1", "source": "paper.pdf", "page": 1},
        ),
        Chunk(
            id="doc-2:p1:c0",
            text="学校经费投入与区域教育公平相关。",
            metadata={"document_id": "doc-2", "source": "policy.pdf", "page": 1},
        ),
    ]
    assert store.add_chunks(chunks) == 2
    hits = store.query("学生反馈", top_k=2, document_ids=["doc-1"])
    assert hits
    assert all(hit.metadata["document_id"] == "doc-1" for hit in hits)
    store.delete_document("doc-1")
    assert store.count() == 1
