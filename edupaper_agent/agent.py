from __future__ import annotations

from typing import Any, Literal, TypedDict
import uuid

from langgraph.graph import END, START, StateGraph

from .arxiv_client import ArxivClient
from .config import Settings
from .llm import LLMClient
from .models import QueryMode
from .storage import ChromaStore


class AgentState(TypedDict, total=False):
    run_id: str
    query: str
    mode: str
    top_k: int
    document_ids: list[str] | None
    allow_arxiv_search: bool
    local_evidence: list[dict[str, Any]]
    arxiv_evidence: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    need_arxiv: bool
    answer: str
    citations: list[dict[str, Any]]
    warnings: list[str]
    trace: list[dict[str, Any]]


def _append_trace(
    state: AgentState, node: str, **detail: Any
) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, "detail": detail}]


def _token_overlap(query: str, text: str) -> float:
    query_chars = {char.lower() for char in query if not char.isspace()}
    text_chars = {char.lower() for char in text if not char.isspace()}
    if not query_chars:
        return 0.0
    return len(query_chars & text_chars) / len(query_chars)


def build_agent_graph(
    *,
    settings: Settings,
    store: ChromaStore,
    arxiv_client: ArxivClient,
    llm: LLMClient,
):
    def initialize(state: AgentState) -> AgentState:
        return {
            "run_id": state.get("run_id") or str(uuid.uuid4()),
            "warnings": [],
            "trace": _append_trace(state, "initialize", mode=state.get("mode", "qa")),
        }

    def retrieve_local(state: AgentState) -> AgentState:
        hits = store.query(
            state["query"],
            top_k=state.get("top_k", settings.default_top_k),
            document_ids=state.get("document_ids"),
        )
        evidence = [
            {
                "text": hit.text,
                "source": hit.metadata.get("source", "uploaded_document"),
                "page": hit.metadata.get("page"),
                "section": hit.metadata.get("section"),
                "document_id": hit.metadata.get("document_id"),
                "score": hit.score,
                "url": None,
            }
            for hit in hits
        ]
        return {
            "local_evidence": evidence,
            "trace": _append_trace(
                state,
                "retrieve_local",
                count=len(evidence),
                best_score=evidence[0].get("score") if evidence else None,
            ),
        }

    def grade_local_evidence(state: AgentState) -> AgentState:
        local = state.get("local_evidence", [])
        best_score = float(local[0].get("score") or 0.0) if local else 0.0
        overlap = max(
            (_token_overlap(state["query"], item.get("text", "")) for item in local),
            default=0.0,
        )
        weak = not local or (best_score < settings.local_score_threshold and overlap < 0.35)
        need_arxiv = bool(
            weak
            and state.get("allow_arxiv_search", True)
            and settings.enable_arxiv
        )
        warnings = list(state.get("warnings", []))
        if weak and not need_arxiv:
            warnings.append("本地检索证据较弱，且未启用 arXiv 补充检索。")
        return {
            "need_arxiv": need_arxiv,
            "warnings": warnings,
            "trace": _append_trace(
                state,
                "grade_local_evidence",
                best_score=round(best_score, 4),
                character_overlap=round(overlap, 4),
                need_arxiv=need_arxiv,
            ),
        }

    def route_after_grade(state: AgentState) -> Literal["search_arxiv", "assemble_evidence"]:
        return "search_arxiv" if state.get("need_arxiv") else "assemble_evidence"

    def search_arxiv(state: AgentState) -> AgentState:
        warnings = list(state.get("warnings", []))
        error: str | None = None
        try:
            papers = arxiv_client.search(
                state["query"], max_results=settings.arxiv_max_results
            )
        except Exception as exc:  # network failures should be visible to the caller
            papers = []
            error = str(exc)
            warnings.append(f"arXiv 检索失败：{error}")

        evidence = [
            {
                "text": paper.summary,
                "source": paper.title,
                "page": None,
                "section": "abstract",
                "document_id": None,
                "score": None,
                "url": paper.entry_url or paper.pdf_url,
            }
            for paper in papers
        ]
        return {
            "arxiv_evidence": evidence,
            "warnings": warnings,
            "trace": _append_trace(
                state, "search_arxiv", count=len(evidence), error=error
            ),
        }

    def assemble_evidence(state: AgentState) -> AgentState:
        local = state.get("local_evidence", [])
        external = state.get("arxiv_evidence", [])
        evidence: list[dict[str, Any]] = []
        for index, item in enumerate(local, start=1):
            evidence.append({**item, "citation_id": f"L{index}"})
        for index, item in enumerate(external, start=1):
            evidence.append({**item, "citation_id": f"A{index}"})

        evidence = evidence[:12]
        return {
            "evidence": evidence,
            "trace": _append_trace(
                state,
                "assemble_evidence",
                local_count=len(local),
                arxiv_count=len(external),
                total=len(evidence),
            ),
        }

    def generate_answer(state: AgentState) -> AgentState:
        mode = QueryMode(state.get("mode", QueryMode.QA.value))
        answer = llm.generate(
            query=state["query"], mode=mode, evidence=state.get("evidence", [])
        )
        citations = [
            {
                "citation_id": item["citation_id"],
                "source": item["source"],
                "page": item.get("page"),
                "section": item.get("section"),
                "snippet": " ".join(item["text"].split())[:360],
                "url": item.get("url"),
                "document_id": item.get("document_id"),
                "score": item.get("score"),
            }
            for item in state.get("evidence", [])
        ]
        return {
            "answer": answer,
            "citations": citations,
            "trace": _append_trace(
                state,
                "generate_answer",
                model_configured=llm.available,
                citation_count=len(citations),
            ),
        }

    def verify_answer(state: AgentState) -> AgentState:
        warnings = list(state.get("warnings", []))
        citations = state.get("citations", [])
        answer = state.get("answer", "")
        if citations and not any(
            f"[{citation['citation_id']}]" in answer for citation in citations
        ):
            warnings.append("生成结果未在正文中使用证据编号，请结合 citations 字段核验。")
        if not citations:
            warnings.append("本次回答没有可用引用。")
        return {
            "warnings": warnings,
            "trace": _append_trace(
                state,
                "verify_answer",
                has_inline_citation=any(
                    f"[{citation['citation_id']}]" in answer for citation in citations
                ),
                warning_count=len(warnings),
            ),
        }

    builder = StateGraph(AgentState)
    builder.add_node("initialize", initialize)
    builder.add_node("retrieve_local", retrieve_local)
    builder.add_node("grade_local_evidence", grade_local_evidence)
    builder.add_node("search_arxiv", search_arxiv)
    builder.add_node("assemble_evidence", assemble_evidence)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("verify_answer", verify_answer)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "retrieve_local")
    builder.add_edge("retrieve_local", "grade_local_evidence")
    builder.add_conditional_edges(
        "grade_local_evidence",
        route_after_grade,
        {
            "search_arxiv": "search_arxiv",
            "assemble_evidence": "assemble_evidence",
        },
    )
    builder.add_edge("search_arxiv", "assemble_evidence")
    builder.add_edge("assemble_evidence", "generate_answer")
    builder.add_edge("generate_answer", "verify_answer")
    builder.add_edge("verify_answer", END)
    return builder.compile()
