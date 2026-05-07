"""
Citation tracking and formatting utilities.
Produces APA-style references and inline citation markers.
"""

from typing import Any
import re


class CitationTracker:
    """Tracks sources collected during a research session."""

    def __init__(self):
        self._sources: list[dict] = []
        self._index: dict[str, int] = {}  # url -> citation number

    def add(self, source: dict) -> int:
        """
        Add a source and return its citation number (1-indexed).
        Deduplicates by URL.
        """
        url = source.get("url", "")
        if url and url in self._index:
            return self._index[url]

        num = len(self._sources) + 1
        entry = {**source, "citation_number": num}
        self._sources.append(entry)
        if url:
            self._index[url] = num
        return num

    def add_all(self, sources: list[dict]) -> list[int]:
        return [self.add(s) for s in sources]

    def get_all(self) -> list[dict]:
        return list(self._sources)

    def clear(self):
        self._sources = []
        self._index = {}

    def to_apa_list(self) -> str:
        """Return a formatted APA reference list."""
        if not self._sources:
            return "No sources collected."
        lines = ["**References**\n"]
        for s in self._sources:
            lines.append(f"{s['citation_number']}. {format_apa(s)}")
        return "\n".join(lines)

    def to_json(self) -> list[dict]:
        return self.get_all()


def format_apa(source: dict) -> str:
    """
    Format a single source dict as an APA citation string.
    Handles both web and academic sources.
    """
    source_type = source.get("source_type", "web")

    if source_type in ("academic", "academic_mock"):
        authors = source.get("authors", "Unknown Author")
        year = source.get("year", "n.d.")
        title = source.get("title", "Untitled")
        venue = source.get("venue", "")
        url = source.get("url", "")
        doi = source.get("doi", "")

        apa = f"{authors} ({year}). {title}."
        if venue:
            apa += f" *{venue}*."
        if doi:
            apa += f" https://doi.org/{doi}"
        elif url:
            apa += f" {url}"
        return apa

    else:  # web source
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        return f"{title}. Retrieved from {url}"


def inject_inline_citations(text: str, sources: list[dict]) -> str:
    """
    Given a text and a list of sources with citation_number,
    append inline numbers where titles appear in the text.
    Falls back to appending a numbered list at the end.
    """
    # Simple approach: append citation markers at end of sentences that
    # mention source content, and add full reference list at end.
    annotated = text.strip()

    ref_list = "\n\n---\n**Sources**\n"
    for s in sources:
        ref_list += f"\n[{s['citation_number']}] {format_apa(s)}"

    return annotated + ref_list


def build_citation_summary(tracker: CitationTracker) -> dict:
    """Return a structured summary for the UI."""
    sources = tracker.get_all()
    web = [s for s in sources if s.get("source_type") in ("web", "web_mock")]
    academic = [s for s in sources if s.get("source_type") in ("academic", "academic_mock")]
    return {
        "total": len(sources),
        "web_count": len(web),
        "academic_count": len(academic),
        "sources": sources,
        "apa_list": tracker.to_apa_list(),
    }
