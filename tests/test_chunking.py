from pathlib import Path

import fitz

from edupaper_agent.pdf import parse_pdf, split_text


def test_split_text_produces_non_empty_bounded_chunks() -> None:
    text = "这是一个用于测试文本切分的句子。" * 100
    chunks = split_text(text, chunk_size=180, overlap=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 181 for chunk in chunks)


def test_parse_pdf_keeps_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Research method and findings. " * 30)
    doc.save(pdf_path)
    doc.close()

    parsed = parse_pdf(pdf_path, document_id="doc-1", source_name="sample.pdf")
    assert parsed.page_count == 1
    assert parsed.chunks
    assert parsed.chunks[0].metadata["page"] == 1
    assert parsed.chunks[0].metadata["document_id"] == "doc-1"
