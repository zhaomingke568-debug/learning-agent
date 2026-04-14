import arxiv
from typing import List

def search_arxiv(query: str, max_results: int = 3) -> List[dict]:
    """Search ArXiv for papers matching the query."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id,
            "authors": [author.name for author in result.authors],
            "published": result.published.strftime("%Y-%m-%d")
        })
    return results
