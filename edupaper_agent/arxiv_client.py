from __future__ import annotations

import arxiv

from .models import PaperResult


class ArxivClient:
    def search(self, query: str, *, max_results: int = 5) -> list[PaperResult]:
        client = arxiv.Client(page_size=max_results, delay_seconds=0.5, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[PaperResult] = []
        for result in client.results(search):
            papers.append(
                PaperResult(
                    title=" ".join(result.title.split()),
                    authors=[author.name for author in result.authors],
                    summary=" ".join(result.summary.split()),
                    published=result.published.isoformat() if result.published else None,
                    pdf_url=result.pdf_url,
                    entry_url=result.entry_id,
                    doi=result.doi,
                )
            )
        return papers
