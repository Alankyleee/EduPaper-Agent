from pathlib import Path

import fitz

from edupaper_agent import pdf
from edupaper_agent.pdf import PdfParserName, PdfParserOptions, parse_pdf


def _make_blank_pdf(path: Path, pages: int = 1) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    document.save(path)
    document.close()


def test_auto_falls_back_to_mineru_for_scanned_pdf(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _make_blank_pdf(path, pages=2)

    monkeypatch.setattr(
        pdf,
        "_extract_mineru_pages",
        lambda *_args, **_kwargs: ["第一页 OCR 文本。" * 20, "第二页 OCR 文本。" * 20],
    )

    parsed = parse_pdf(
        path,
        document_id="doc-ocr",
        source_name="scan.pdf",
        options=PdfParserOptions(fallback_order=(PdfParserName.MINERU,)),
    )

    assert parsed.parser == PdfParserName.MINERU
    assert parsed.page_count == 2
    assert parsed.chunks
    assert parsed.chunks[0].metadata["parser"] == "mineru"
    assert parsed.warnings


def test_mineru_content_list_keeps_page_and_structured_content(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    content = [
        {"type": "text", "text": "研究背景", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "第一段正文", "page_idx": 0},
        {"type": "table", "table_body": "|A|B|\n|-|-|\n|1|2|", "page_idx": 1},
        {"type": "equation", "text": "y=ax+b", "page_idx": 1},
    ]
    (output / "paper_content_list.json").write_text(
        __import__("json").dumps(content, ensure_ascii=False), encoding="utf-8"
    )

    pages = pdf._read_mineru_output(output, page_count=2)

    assert "# 研究背景" in pages[0]
    assert "第一段正文" in pages[0]
    assert "|A|B|" in pages[1]
    assert "y=ax+b" in pages[1]


def test_explicit_paddleocr_parser_with_mock_pipeline(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _make_blank_pdf(path)

    class Result:
        markdown = {"markdown_texts": "# 标题\n\nPaddleOCR 解析正文。" * 20}
        json = {"page_index": 0}

    class Pipeline:
        def predict(self, **_kwargs):
            return [Result()]

    monkeypatch.setattr(pdf.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(pdf, "_load_paddle_pipeline", lambda _device: Pipeline())

    parsed = parse_pdf(
        path,
        document_id="doc-paddle",
        source_name="scan.pdf",
        parser=PdfParserName.PADDLEOCR,
    )

    assert parsed.parser == PdfParserName.PADDLEOCR
    assert parsed.chunks
    assert parsed.chunks[0].metadata["page"] == 1
