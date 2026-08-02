from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbedder:
    """Offline multilingual character n-gram embeddings.

    This is intentionally deterministic and does not download a model. It makes the
    repository runnable without an API key. For production, replace it with a stronger
    embedding model while keeping the same interface.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
