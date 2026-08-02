from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .models import DocumentRecord


class DocumentRegistry:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    parser TEXT NOT NULL DEFAULT 'pymupdf',
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "parser" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN parser TEXT NOT NULL DEFAULT 'pymupdf'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
            )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> DocumentRecord:
        keys = set(row.keys())
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            sha256=row["sha256"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            parser=row["parser"] if "parser" in keys else "pymupdf",
            file_path=row["file_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def add(
        self,
        *,
        document_id: str,
        filename: str,
        sha256: str,
        page_count: int,
        chunk_count: int,
        parser: str,
        file_path: Path,
    ) -> DocumentRecord:
        created_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, sha256, page_count, chunk_count, parser, file_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    sha256,
                    page_count,
                    chunk_count,
                    parser,
                    str(file_path),
                    created_at.isoformat(),
                ),
            )
        record = self.get(document_id)
        if record is None:  # pragma: no cover - defensive branch
            raise RuntimeError("文档元数据写入后无法读取")
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def find_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._to_record(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def delete(self, document_id: str) -> DocumentRecord | None:
        record = self.get(document_id)
        if record is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return record

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])
