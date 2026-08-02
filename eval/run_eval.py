from __future__ import annotations

import json
from pathlib import Path
import time

from edupaper_agent.config import get_settings
from edupaper_agent.models import QueryRequest
from edupaper_agent.services import ServiceContainer


def main() -> None:
    cases = json.loads(Path("eval/cases.json").read_text(encoding="utf-8"))
    services = ServiceContainer(get_settings())
    rows = []

    for case in cases:
        started = time.perf_counter()
        response = services.query(QueryRequest.model_validate(case))
        latency = time.perf_counter() - started
        checks = {
            "has_citations": bool(response.citations),
            "no_empty_answer": bool(response.answer.strip()),
        }
        requested_checks = case.get("checks", [])
        passed = all(checks.get(name, False) for name in requested_checks)
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "latency_seconds": round(latency, 3),
                "citation_count": len(response.citations),
                "used_arxiv": response.used_arxiv,
                "warnings": response.warnings,
                "answer_preview": response.answer[:240],
                "checks": checks,
            }
        )

    output = Path("eval/results.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
