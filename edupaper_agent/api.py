from __future__ import annotations

from pathlib import Path
import re
import uuid

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from .config import Settings, get_settings
from .models import (
    DeleteResponse,
    DocumentRecord,
    HealthResponse,
    PaperResult,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from .pdf import ParserUnavailableError, PdfParserName, parser_availability
from .services import ServiceContainer


_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+")


def _safe_filename(filename: str) -> str:
    cleaned = _FILENAME_RE.sub("_", Path(filename).name).strip("._")
    return cleaned[:120] or "document.pdf"


async def _save_upload(
    file: UploadFile, *, upload_dir: Path, max_bytes: int
) -> Path:
    suffix = ".pdf"
    destination = upload_dir / f"{uuid.uuid4()}{suffix}"
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="上传文件超过大小限制")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return destination


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    services = ServiceContainer(app_settings)

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="可追溯的教育论文研读、RAG 检索与综述生成 Agent。",
    )
    app.state.services = services

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        svc: ServiceContainer = request.app.state.services
        return HealthResponse(
            status="ok",
            version=app_settings.app_version,
            document_count=svc.registry.count(),
            chunk_count=svc.store.count(),
            model_configured=svc.llm.available,
            default_pdf_parser=app_settings.pdf_parser.value,
            parser_availability=parser_availability(app_settings.pdf_parser_options()),
        )

    @app.post("/v1/documents/upload", response_model=UploadResponse)
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        parser: PdfParserName | None = Query(
            default=None,
            description="PDF 解析器：auto、pymupdf、mineru 或 paddleocr",
        ),
    ) -> UploadResponse:
        filename = _safe_filename(file.filename or "document.pdf")
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
        if file.content_type not in {None, "application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail="文件 Content-Type 不是 PDF")

        svc: ServiceContainer = request.app.state.services
        path = await _save_upload(
            file,
            upload_dir=app_settings.data_dir / "uploads",
            max_bytes=app_settings.max_upload_mb * 1024 * 1024,
        )
        try:
            document, duplicated, warnings = svc.ingest_pdf(
                path,
                original_name=filename,
                parser=parser,
            )
        except ParserUnavailableError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"文档入库失败：{exc}") from exc
        return UploadResponse(
            document=document,
            duplicated=duplicated,
            warnings=warnings,
        )

    @app.get("/v1/documents", response_model=list[DocumentRecord])
    def list_documents(request: Request) -> list[DocumentRecord]:
        svc: ServiceContainer = request.app.state.services
        return svc.list_documents()

    @app.delete("/v1/documents/{document_id}", response_model=DeleteResponse)
    def delete_document(document_id: str, request: Request) -> DeleteResponse:
        svc: ServiceContainer = request.app.state.services
        deleted = svc.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="文档不存在")
        return DeleteResponse(document_id=document_id, deleted=True)

    @app.post("/v1/agent/query", response_model=QueryResponse)
    def query_agent(payload: QueryRequest, request: Request) -> QueryResponse:
        svc: ServiceContainer = request.app.state.services
        unknown_ids = [
            document_id
            for document_id in payload.document_ids or []
            if svc.registry.get(document_id) is None
        ]
        if unknown_ids:
            raise HTTPException(
                status_code=404,
                detail={"message": "部分文档不存在", "document_ids": unknown_ids},
            )
        try:
            return svc.query(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc

    @app.get("/v1/papers/search", response_model=list[PaperResult])
    def search_papers(
        request: Request,
        q: str = Query(min_length=2, max_length=500),
        limit: int = Query(default=5, ge=1, le=10),
    ) -> list[PaperResult]:
        svc: ServiceContainer = request.app.state.services
        if not app_settings.enable_arxiv:
            raise HTTPException(status_code=503, detail="arXiv 检索已禁用")
        try:
            return svc.arxiv_client.search(q, max_results=limit)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"arXiv 检索失败：{exc}") from exc

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> dict:
        svc: ServiceContainer = request.app.state.services
        payload = svc.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="运行记录不存在")
        return payload

    @app.get("/v1/graph/mermaid")
    def graph_mermaid(request: Request) -> dict[str, str]:
        svc: ServiceContainer = request.app.state.services
        return {"mermaid": svc.graph.get_graph().draw_mermaid()}

    return app


app = create_app()
