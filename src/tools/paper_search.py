"""
Academic paper search via Semantic Scholar API (free, no key required for basic use).
"""

import os
import json
import urllib.request
import urllib.parse
from typing import Any


S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,abstract,externalIds,url,citationCount,venue"


def paper_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Semantic Scholar for academic papers.
    Returns list of structured paper dicts.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": FIELDS,
    })
    url = f"{S2_BASE}/paper/search?{params}"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for p in data.get("data", [])[:max_results]:
            authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            results.append({
                "title": p.get("title", "Untitled"),
                "authors": authors,
                "year": p.get("year", "n.d."),
                "abstract": (p.get("abstract") or "")[:600],
                "url": p.get("url", ""),
                "venue": p.get("venue", ""),
                "citations": p.get("citationCount", 0),
                "paper_id": p.get("paperId", ""),
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
                "source_type": "academic",
            })
        return results
    except Exception as e:
        return _mock_paper_search(query, max_results, error=str(e))


def format_paper_results(papers: list[dict]) -> str:
    """Format paper results into a readable string for the LLM."""
    if not papers:
        return "No academic papers found."
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[P{i}] {p['title']} ({p['year']})")
        lines.append(f"     Authors: {p['authors']}")
        if p.get("venue"):
            lines.append(f"     Venue: {p['venue']}")
        lines.append(f"     Citations: {p.get('citations', 0)}")
        if p.get("abstract"):
            lines.append(f"     Abstract: {p['abstract'][:300]}...")
        lines.append(f"     URL: {p['url']}")
        lines.append("")
    return "\n".join(lines)


def _mock_paper_search(query: str, max_results: int, error: str = "") -> list[dict]:
    """Fallback mock for testing without network."""
    note = f" (API error: {error})" if error else ""
    return [
        {
            "title": f"Mock Paper: {query}{note}",
            "authors": "Smith, J., Doe, A.",
            "year": 2024,
            "abstract": (
                f"This mock paper discusses '{query}' in the context of HCI research. "
                "In production this would contain a real abstract from Semantic Scholar."
            ),
            "url": "https://api.semanticscholar.org",
            "venue": "Mock Conference",
            "citations": 0,
            "paper_id": "mock-id",
            "doi": "",
            "source_type": "academic_mock",
        }
    ]
