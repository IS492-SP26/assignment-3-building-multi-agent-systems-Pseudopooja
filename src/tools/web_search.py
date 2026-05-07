"""
Web search tool using Tavily API.
Falls back to a mock if TAVILY_API_KEY is not set (for testing).
"""

import os
import json
import re
import urllib.request
import urllib.parse
from typing import Any


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily. Returns list of structured result dicts.
    Each result has: title, url, content, score.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        return _mock_search(query, max_results)

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "source_type": "web",
            })
        return results
    except Exception as e:
        return [{"title": "Search error", "url": "", "content": str(e), "score": 0, "source_type": "web"}]


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['content'][:400]}")
        lines.append("")
    return "\n".join(lines)


def _mock_search(query: str, max_results: int) -> list[dict]:
    """Fallback mock when no API key is available."""
    return [
        {
            "title": f"Mock result for: {query}",
            "url": "https://example.com/mock",
            "content": (
                f"This is a mock search result for the query '{query}'. "
                "In production, this would contain real web content. "
                "Set TAVILY_API_KEY in your .env to enable live search."
            ),
            "score": 0.5,
            "source_type": "web_mock",
        }
    ]
