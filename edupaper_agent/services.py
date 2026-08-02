from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

from .agent import build_agent_graph
from .arxiv_client import ArxivClient
from .config import Settings
from .llm import LLMClient
from .models import DocumentRecord, QueryRequest, QueryResponse, TraceEvent
from .pdf import PdfParserName, parse_pdf
from .registry import DocumentRegistry
from .storage import ChromaStore


class ServiceContainer:
    def __init__(
        self,
        settings: Settings,
        *,
        arxiv_client: ArxivClient | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.registry = DocumentRegistry(settings.sqlite_path)
        self.store = ChromaStore(
            persist_dir=settings.chroma_dir,
            collection_name=settings.collection_name,
        )
        self.arxiv_client = arxiv_client or ArxivClient()
        self.llm = LLMClient(settings)
        self.graph = build_agent_graph(
            settings=settings,
            store=self.store,
            arxiv_client=self.arxiv_client,
            llm=self.llm,
        )

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def ingest_pdf(
        self,
        path: Path,
        *,
        original_name: str,
        parser: PdfParserName | str | None = None,
    ) -> tuple[DocumentRecord, bool, list[str]]:
        file_sha = self.sha256(path)
        existing = self.registry.find_by_sha256(file_sha)
        if existing:
            path.unlink(missing_ok=True)
            return existing, True, []

        document_id = str(uuid.uuid4())
        parsed = parse_pdf(
            path,
            document_id=document_id,
            source_name=original_name,
            parser=parser or self.settings.pdf_parser,
            options=self.settings.pdf_parser_options(),
        )
        self.store.add_chunks(parsed.chunks)
        try:
            record = self.registry.add(
                document_id=document_id,
                filename=original_name,
                sha256=file_sha,
                page_count=parsed.page_count,
                chunk_count=len(parsed.chunks),
                parser=parsed.parser.value,
                file_path=path,
            )
        except Exception:
            self.store.delete_document(document_id)
            raise
        return record, False, parsed.warnings

    def delete_document(self, document_id: str) -> bool:
        record = self.registry.delete(document_id)
        if record is None:
            return False
        self.store.delete_document(document_id)
        Path(record.file_path).unlink(missing_ok=True)
        return True

    def list_documents(self) -> list[DocumentRecord]:
        return self.registry.list()

    def query(self, request: QueryRequest) -> QueryResponse:
        state = self.graph.invoke(
            {
                "query": request.query,
                "mode": request.mode.value,
                "top_k": request.top_k,
                "document_ids": request.document_ids,
                "allow_arxiv_search": request.allow_arxiv_search,
            }
        )
        response = QueryResponse(
            run_id=state["run_id"],
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            used_arxiv=bool(state.get("arxiv_evidence")),
            warnings=state.get("warnings", []),
            trace=[TraceEvent.model_validate(item) for item in state.get("trace", [])],
        )
        self._save_run(request=request, response=response)
        return response

    def _save_run(self, *, request: QueryRequest, response: QueryResponse) -> None:
        path = self.settings.data_dir / "runs" / f"{response.run_id}.json"
        payload = {
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_run(self, run_id: str) -> dict | None:
        path = self.settings.data_dir / "runs" / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
