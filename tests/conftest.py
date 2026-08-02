from pathlib import Path

import pytest

from edupaper_agent.config import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        sqlite_path=tmp_path / "data" / "edupaper.db",
        collection_name="test_collection",
        openai_api_key=None,
        enable_arxiv=False,
    )
    settings.ensure_directories()
    return settings
