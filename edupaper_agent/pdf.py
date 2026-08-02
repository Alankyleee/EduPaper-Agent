from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import fitz


class PdfParserName(str, Enum):
    AUTO = "auto"
    PYMUPDF = "pymupdf"
    MINERU = "mineru"
    PADDLEOCR = "paddleocr"


class ParserUnavailableError(RuntimeError):
    """Raised when an optional document parser is not installed or executable."""


@dataclass(frozen=True, slots=True)
class PdfParserOptions:
    fallback_order: tuple[PdfParserName, ...] = (
        PdfParserName.MINERU,
        PdfParserName.PADDLEOCR,
    )
    min_native_chars_per_page: int = 80
    min_native_page_coverage: float = 0.50

    mineru_command: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_timeout_seconds: int = 900

    paddleocr_device: str = "cpu"
    paddleocr_use_doc_orientation_classify: bool = True
    paddleocr_use_doc_unwarping: bool = False
    paddleocr_use_textline_orientation: bool = True


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    page_count: int
    chunks: list[Chunk]
    parser: PdfParserName
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, *, chunk_size: int = 1_200, overlap: int = 180) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    sentence_boundaries = "。！？!?；;\n"

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end
        if proposed_end < len(text):
            lower_bound = start + chunk_size // 2
            candidates = [
                text.rfind(mark, lower_bound, proposed_end)
                for mark in sentence_boundaries
            ]
            boundary = max(candidates)
            if boundary >= lower_bound:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def parser_availability(options: PdfParserOptions | None = None) -> dict[str, bool]:
    opts = options or PdfParserOptions()
    return {
        PdfParserName.PYMUPDF.value: True,
        PdfParserName.MINERU.value: shutil.which(opts.mineru_command) is not None,
        PdfParserName.PADDLEOCR.value: importlib.util.find_spec("paddleocr") is not None,
    }


def _meaningful_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _native_text_is_sufficient(page_texts: list[str], options: PdfParserOptions) -> bool:
    if not page_texts:
        return False
    threshold = max(1, options.min_native_chars_per_page)
    page_counts = [_meaningful_char_count(text) for text in page_texts]
    covered_pages = sum(count >= threshold for count in page_counts)
    coverage = covered_pages / len(page_counts)
    total = sum(page_counts)
    minimum_total = max(120, int(len(page_texts) * threshold * 0.45))
    return total >= minimum_total and coverage >= options.min_native_page_coverage


def _read_page_count(path: Path) -> int:
    try:
        with fitz.open(path) as document:
            return document.page_count
    except Exception as exc:  # pragma: no cover - library-specific error paths
        raise ValueError(f"无法打开 PDF：{exc}") from exc


def _extract_pymupdf_pages(path: Path) -> list[str]:
    try:
        document = fitz.open(path)
    except Exception as exc:  # pragma: no cover - library-specific error paths
        raise ValueError(f"无法打开 PDF：{exc}") from exc

    with document:
        if document.page_count == 0:
            raise ValueError("PDF 不包含任何页面")
        return [clean_text(page.get_text("text")) for page in document]


def _build_document(
    page_texts: list[str],
    *,
    document_id: str,
    source_name: str,
    parser: PdfParserName,
    chunk_size: int,
    overlap: int,
    warnings: Iterable[str] = (),
) -> ParsedDocument:
    chunks: list[Chunk] = []
    for page_index, page_text in enumerate(page_texts):
        for chunk_index, text in enumerate(
            split_text(page_text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    id=f"{document_id}:p{page_index + 1}:c{chunk_index}",
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "source": source_name,
                        "page": page_index + 1,
                        "chunk_index": chunk_index,
                        "parser": parser.value,
                    },
                )
            )

    if not chunks:
        raise ValueError(f"{parser.value} 未提取到可检索文本")

    return ParsedDocument(
        page_count=len(page_texts),
        chunks=chunks,
        parser=parser,
        warnings=list(warnings),
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (_as_text(item) for item in value) if part)
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        return "\n".join(part for part in (_as_text(item) for item in value.values()) if part)
    return str(value)


def _mineru_block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", ""))
    if block_type in {"header", "footer", "page_number", "aside_text"}:
        return ""

    if block_type == "text":
        text = _as_text(block.get("text"))
        level = int(block.get("text_level") or 0)
        return f"{'#' * min(level, 6)} {text}" if level > 0 and text else text

    if block_type == "equation":
        equation = _as_text(
            block.get("text")
            or block.get("equation")
            or block.get("latex")
            or block.get("content")
        )
        return f"$$\n{equation}\n$$" if equation else ""

    if block_type == "list":
        return _as_text(block.get("list_items") or block.get("text"))

    if block_type == "code":
        caption = _as_text(block.get("code_caption"))
        body = _as_text(block.get("code_body") or block.get("text"))
        pieces = [caption, f"```\n{body}\n```" if body else ""]
        return "\n".join(piece for piece in pieces if piece)

    if block_type == "table":
        caption = _as_text(block.get("table_caption"))
        body = _as_text(
            block.get("table_body")
            or block.get("content")
            or block.get("text")
            or block.get("html")
        )
        footnote = _as_text(block.get("table_footnote"))
        return "\n".join(piece for piece in (caption, body, footnote) if piece)

    if block_type in {"image", "chart"}:
        caption = _as_text(
            block.get(f"{block_type}_caption")
            or block.get("caption")
            or block.get("text")
        )
        content = _as_text(block.get("content"))
        footnote = _as_text(block.get(f"{block_type}_footnote"))
        return "\n".join(piece for piece in (caption, content, footnote) if piece)

    return _as_text(block.get("text") or block.get("content"))


def _flatten_mineru_v2_span_list(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten_mineru_v2_span_list(item) for item in value)
    if isinstance(value, dict):
        if "content" in value and isinstance(value["content"], str):
            return value["content"]
        if "children" in value:
            return _flatten_mineru_v2_span_list(value["children"])
        return "\n".join(
            part for part in (_flatten_mineru_v2_span_list(item) for item in value.values()) if part
        )
    return ""


def _mineru_v2_block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", ""))
    content = block.get("content") or {}
    if block_type.startswith("page_") and block_type != "page_footnote":
        return ""
    if block_type == "title":
        text = _flatten_mineru_v2_span_list(content.get("title_content"))
        level = int(content.get("level") or 1)
        return f"{'#' * min(level, 6)} {text}" if text else ""
    if block_type == "paragraph":
        return _flatten_mineru_v2_span_list(content.get("paragraph_content"))
    if block_type == "equation_interline":
        equation = _flatten_mineru_v2_span_list(content.get("math_content"))
        return f"$$\n{equation}\n$$" if equation else ""
    if block_type in {"list", "index"}:
        return _as_text(content.get("list_items"))
    if block_type in {"code", "algorithm"}:
        body = _flatten_mineru_v2_span_list(
            content.get("code_content") or content.get("algorithm_content")
        )
        return f"```\n{body}\n```" if body else ""
    return _flatten_mineru_v2_span_list(content)


def _read_mineru_output(output_dir: Path, page_count: int) -> list[str]:
    legacy_files = [
        path
        for path in output_dir.rglob("*_content_list.json")
        if not path.name.endswith("_content_list_v2.json")
    ]
    if legacy_files:
        content = json.loads(legacy_files[0].read_text(encoding="utf-8"))
        page_blocks: dict[int, list[str]] = defaultdict(list)
        for block in content:
            if not isinstance(block, dict):
                continue
            page_index = int(block.get("page_idx") or 0)
            text = clean_text(_mineru_block_text(block))
            if text:
                page_blocks[page_index].append(text)
        return ["\n\n".join(page_blocks.get(index, [])) for index in range(page_count)]

    v2_files = list(output_dir.rglob("*_content_list_v2.json"))
    if v2_files:
        content = json.loads(v2_files[0].read_text(encoding="utf-8"))
        pages: list[str] = []
        for page in content:
            page_text = "\n\n".join(
                text
                for text in (
                    clean_text(_mineru_v2_block_text(block))
                    for block in page
                    if isinstance(block, dict)
                )
                if text
            )
            pages.append(page_text)
        if len(pages) < page_count:
            pages.extend([""] * (page_count - len(pages)))
        return pages[:page_count]

    markdown_files = sorted(output_dir.rglob("*.md"))
    if markdown_files:
        text = clean_text(markdown_files[0].read_text(encoding="utf-8"))
        return [text] + [""] * max(0, page_count - 1)

    raise ValueError("MinerU 运行完成，但未找到 content_list.json 或 Markdown 输出")


def _extract_mineru_pages(
    path: Path,
    *,
    page_count: int,
    options: PdfParserOptions,
) -> list[str]:
    executable = shutil.which(options.mineru_command)
    if executable is None:
        raise ParserUnavailableError(
            "未检测到 MinerU CLI。请安装 `mineru[all]`，并确认 `mineru` 命令可用。"
        )

    with tempfile.TemporaryDirectory(prefix="edupaper-mineru-") as temp_dir:
        output_dir = Path(temp_dir) / "output"
        command = [
            executable,
            "-p",
            str(path),
            "-o",
            str(output_dir),
            "-b",
            options.mineru_backend,
            "--dump-content-list",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=options.mineru_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"MinerU 解析超时（{options.mineru_timeout_seconds} 秒）"
            ) from exc

        if result.returncode != 0:
            detail = clean_text(result.stderr or result.stdout)[-2_000:]
            raise ValueError(f"MinerU 解析失败：{detail or '未知错误'}")
        return _read_mineru_output(output_dir, page_count)


@lru_cache(maxsize=4)
def _load_paddle_pipeline(device: str):
    try:
        from paddleocr import PPStructureV3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ParserUnavailableError(
            "未安装 PaddleOCR 文档解析依赖。请安装 `paddleocr[doc-parser]` 及对应推理引擎。"
        ) from exc
    return PPStructureV3(device=device)


def _paddle_result_json(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    return payload if isinstance(payload, dict) else {}


def _extract_paddleocr_pages(
    path: Path,
    *,
    page_count: int,
    options: PdfParserOptions,
) -> list[str]:
    if importlib.util.find_spec("paddleocr") is None:
        raise ParserUnavailableError(
            "未安装 PaddleOCR。请安装 `paddleocr[doc-parser]` 及对应推理引擎。"
        )

    pipeline = _load_paddle_pipeline(options.paddleocr_device)
    try:
        output = pipeline.predict(
            input=str(path),
            use_doc_orientation_classify=options.paddleocr_use_doc_orientation_classify,
            use_doc_unwarping=options.paddleocr_use_doc_unwarping,
            use_textline_orientation=options.paddleocr_use_textline_orientation,
        )
    except Exception as exc:  # pragma: no cover - optional dependency runtime
        raise ValueError(f"PaddleOCR 解析失败：{exc}") from exc

    page_texts = [""] * page_count
    for fallback_index, result in enumerate(output):
        markdown = getattr(result, "markdown", None)
        if callable(markdown):
            markdown = markdown()
        markdown = markdown if isinstance(markdown, dict) else {}
        text = clean_text(_as_text(markdown.get("markdown_texts")))

        result_json = _paddle_result_json(result)
        raw_page_index = result_json.get("page_index", fallback_index)
        try:
            page_index = int(raw_page_index)
        except (TypeError, ValueError):
            page_index = fallback_index
        if 0 <= page_index < page_count:
            page_texts[page_index] = text
        elif fallback_index < page_count:
            page_texts[fallback_index] = text

    return page_texts


def _parse_with_engine(
    engine: PdfParserName,
    path: Path,
    *,
    document_id: str,
    source_name: str,
    chunk_size: int,
    overlap: int,
    options: PdfParserOptions,
    warnings: Iterable[str] = (),
) -> ParsedDocument:
    if engine == PdfParserName.PYMUPDF:
        pages = _extract_pymupdf_pages(path)
    else:
        page_count = _read_page_count(path)
        if engine == PdfParserName.MINERU:
            pages = _extract_mineru_pages(path, page_count=page_count, options=options)
        elif engine == PdfParserName.PADDLEOCR:
            pages = _extract_paddleocr_pages(path, page_count=page_count, options=options)
        else:  # pragma: no cover - protected by enum
            raise ValueError(f"不支持的 PDF 解析器：{engine.value}")

    return _build_document(
        pages,
        document_id=document_id,
        source_name=source_name,
        parser=engine,
        chunk_size=chunk_size,
        overlap=overlap,
        warnings=warnings,
    )


def parse_pdf(
    path: Path,
    *,
    document_id: str,
    source_name: str,
    parser: PdfParserName | str = PdfParserName.AUTO,
    options: PdfParserOptions | None = None,
    chunk_size: int = 1_200,
    overlap: int = 180,
) -> ParsedDocument:
    selected = PdfParserName(parser)
    opts = options or PdfParserOptions()

    if selected != PdfParserName.AUTO:
        return _parse_with_engine(
            selected,
            path,
            document_id=document_id,
            source_name=source_name,
            chunk_size=chunk_size,
            overlap=overlap,
            options=opts,
        )

    native_pages = _extract_pymupdf_pages(path)
    if _native_text_is_sufficient(native_pages, opts):
        return _build_document(
            native_pages,
            document_id=document_id,
            source_name=source_name,
            parser=PdfParserName.PYMUPDF,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    warnings = [
        "PyMuPDF 提取到的原生文本较少，已尝试使用版面/OCR 解析器。"
    ]
    failures: list[str] = []
    for fallback in opts.fallback_order:
        if fallback in {PdfParserName.AUTO, PdfParserName.PYMUPDF}:
            continue
        try:
            return _parse_with_engine(
                fallback,
                path,
                document_id=document_id,
                source_name=source_name,
                chunk_size=chunk_size,
                overlap=overlap,
                options=opts,
                warnings=warnings,
            )
        except (ParserUnavailableError, ValueError) as exc:
            failures.append(f"{fallback.value}: {exc}")

    if any(_meaningful_char_count(text) for text in native_pages):
        warnings.extend(failures)
        warnings.append("OCR/版面解析不可用，已退回稀疏的 PyMuPDF 文本。")
        return _build_document(
            native_pages,
            document_id=document_id,
            source_name=source_name,
            parser=PdfParserName.PYMUPDF,
            chunk_size=chunk_size,
            overlap=overlap,
            warnings=warnings,
        )

    detail = "；".join(failures) or "未配置可用的 OCR/版面解析器"
    raise ValueError(f"PDF 未提取到文本。请安装 MinerU 或 PaddleOCR。详情：{detail}")
