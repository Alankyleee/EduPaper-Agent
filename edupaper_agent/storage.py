from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .embeddings import HashingEmbedder
from .pdf import Chunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None

    @property
    def score(self) -> float | None:
        if self.distance is None:
            return None
        return max(0.0, min(1.0, 1.0 - float(self.distance)))


class ChromaStore:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        embedder: HashingEmbedder | None = None,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding": "hashing-char-ngram-v1"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=self.embedder.embed_documents(texts),
        )
        return len(chunks)

    def query(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if self.collection.count() == 0:
            return []

        where: dict[str, Any] | None = None
        if document_ids:
            unique_ids = sorted(set(document_ids))
            where = (
                {"document_id": unique_ids[0]}
                if len(unique_ids) == 1
                else {"document_id": {"$in": unique_ids}}
            )

        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            SearchHit(
                id=item_id,
                text=document or "",
                metadata=metadata or {},
                distance=distance,
            )
            for item_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        ]

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self.collection.count()
