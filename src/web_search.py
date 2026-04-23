"""
web_search.py — Web search tool using DuckDuckGo (no API key required).

Returns results in the same {text, source, score} format as the FAISS
retriever so that generator.py works without any changes.
"""

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo and return results as retriever-compatible chunks.

    Args:
        query:       The search query (same as the user's question).
        max_results: How many web results to fetch.

    Returns:
        List of {text, source, score} dicts.
        score is 0.0 — web results have no vector distance.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            text   = r.get("body", "").strip()
            source = r.get("href") or r.get("url") or "web"
            if text:
                results.append({"text": text, "source": source, "score": 0.0})
    return results
