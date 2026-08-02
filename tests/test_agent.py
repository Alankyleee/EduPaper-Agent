from edupaper_agent.arxiv_client import ArxivClient
from edupaper_agent.models import QueryRequest
from edupaper_agent.services import ServiceContainer


class NoNetworkArxiv(ArxivClient):
    def search(self, query: str, *, max_results: int = 5):
        raise AssertionError("arXiv should not be called in this test")


def test_agent_returns_trace_without_model(test_settings) -> None:
    services = ServiceContainer(test_settings, arxiv_client=NoNetworkArxiv())
    response = services.query(
        QueryRequest(
            query="文档中的研究方法是什么？",
            allow_arxiv_search=False,
        )
    )
    assert response.answer
    assert response.trace
    assert any(event.node == "retrieve_local" for event in response.trace)
