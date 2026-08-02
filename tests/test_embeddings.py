from edupaper_agent.embeddings import HashingEmbedder


def test_hashing_embeddings_are_deterministic() -> None:
    embedder = HashingEmbedder(dimensions=128)
    first = embedder.embed_query("教育公平")
    second = embedder.embed_query("教育公平")
    assert first == second
    assert len(first) == 128
