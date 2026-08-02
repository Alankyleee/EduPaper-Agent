# AGENTS.md

## WHAT
EduPaper Agent is a traceable education-paper research assistant built with FastAPI, LangGraph, Chroma, PDF parsing and an OpenAI-compatible model API.

## WHY
- Answers must be grounded in returned evidence.
- Unsupported claims must be marked as insufficient evidence.
- The project must remain runnable without an API key.
- Uploaded documents and API keys must never be committed.

## HOW
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
python start.py
```

## TEST
After changing code, run:
```bash
ruff check .
pytest
```
External arXiv calls must be mocked in unit tests. Changes to retrieval or prompts should add or update an eval case.
