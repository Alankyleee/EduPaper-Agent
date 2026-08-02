from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from edupaper_agent.api import create_app


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Education research method and sample. " * 40)
    doc.save(path)
    doc.close()


def test_health_and_pdf_upload(test_settings, tmp_path: Path) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)
    with pdf_path.open("rb") as file:
        response = client.post(
            "/v1/documents/upload",
            files={"file": ("paper.pdf", file, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["document"]["chunk_count"] >= 1
    assert response.json()["document"]["parser"] == "pymupdf"


def test_query_validation(test_settings) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    response = client.post("/v1/agent/query", json={"query": "x"})
    assert response.status_code == 422
