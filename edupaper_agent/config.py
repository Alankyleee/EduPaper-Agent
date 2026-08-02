from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

from .pdf import PdfParserName, PdfParserOptions

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parser_order(raw: str) -> tuple[PdfParserName, ...]:
    parsers: list[PdfParserName] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parser = PdfParserName(item)
        if parser not in parsers:
            parsers.append(parser)
    return tuple(parsers) or (PdfParserName.MINERU, PdfParserName.PADDLEOCR)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "EduPaper Agent"
    app_version: str = "0.3.0"
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    sqlite_path: Path = Path("./data/edupaper.db")
    collection_name: str = "education_papers"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model_name: str = "gpt-4o-mini"

    max_upload_mb: int = 50
    max_context_chars: int = 16_000
    default_top_k: int = 6
    local_score_threshold: float = 0.20
    arxiv_max_results: int = 5
    enable_arxiv: bool = True

    pdf_parser: PdfParserName = PdfParserName.AUTO
    pdf_fallback_order: tuple[PdfParserName, ...] = (
        PdfParserName.MINERU,
        PdfParserName.PADDLEOCR,
    )
    pdf_min_native_chars_per_page: int = 80
    pdf_min_native_page_coverage: float = 0.50

    mineru_command: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_timeout_seconds: int = 900

    paddleocr_device: str = "cpu"
    paddleocr_use_doc_orientation_classify: bool = True
    paddleocr_use_doc_unwarping: bool = False
    paddleocr_use_textline_orientation: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("EDUPAPER_DATA_DIR", "./data"))
        return cls(
            app_name=os.getenv("APP_NAME", "EduPaper Agent"),
            app_version=os.getenv("APP_VERSION", "0.3.0"),
            data_dir=data_dir,
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma"))),
            sqlite_path=Path(os.getenv("SQLITE_PATH", str(data_dir / "edupaper.db"))),
            collection_name=os.getenv("CHROMA_COLLECTION", "education_papers"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "16000")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "6")),
            local_score_threshold=float(os.getenv("LOCAL_SCORE_THRESHOLD", "0.20")),
            arxiv_max_results=int(os.getenv("ARXIV_MAX_RESULTS", "5")),
            enable_arxiv=_env_bool("ENABLE_ARXIV", True),
            pdf_parser=PdfParserName(os.getenv("PDF_PARSER", "auto").strip().lower()),
            pdf_fallback_order=_parser_order(
                os.getenv("PDF_FALLBACK_ORDER", "mineru,paddleocr")
            ),
            pdf_min_native_chars_per_page=int(
                os.getenv("PDF_MIN_NATIVE_CHARS_PER_PAGE", "80")
            ),
            pdf_min_native_page_coverage=float(
                os.getenv("PDF_MIN_NATIVE_PAGE_COVERAGE", "0.50")
            ),
            mineru_command=os.getenv("MINERU_COMMAND", "mineru"),
            mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
            mineru_timeout_seconds=int(os.getenv("MINERU_TIMEOUT_SECONDS", "900")),
            paddleocr_device=os.getenv("PADDLEOCR_DEVICE", "cpu"),
            paddleocr_use_doc_orientation_classify=_env_bool(
                "PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY", True
            ),
            paddleocr_use_doc_unwarping=_env_bool(
                "PADDLEOCR_USE_DOC_UNWARPING", False
            ),
            paddleocr_use_textline_orientation=_env_bool(
                "PADDLEOCR_USE_TEXTLINE_ORIENTATION", True
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs").mkdir(parents=True, exist_ok=True)

    def pdf_parser_options(self) -> PdfParserOptions:
        return PdfParserOptions(
            fallback_order=self.pdf_fallback_order,
            min_native_chars_per_page=self.pdf_min_native_chars_per_page,
            min_native_page_coverage=self.pdf_min_native_page_coverage,
            mineru_command=self.mineru_command,
            mineru_backend=self.mineru_backend,
            mineru_timeout_seconds=self.mineru_timeout_seconds,
            paddleocr_device=self.paddleocr_device,
            paddleocr_use_doc_orientation_classify=(
                self.paddleocr_use_doc_orientation_classify
            ),
            paddleocr_use_doc_unwarping=self.paddleocr_use_doc_unwarping,
            paddleocr_use_textline_orientation=self.paddleocr_use_textline_orientation,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings
