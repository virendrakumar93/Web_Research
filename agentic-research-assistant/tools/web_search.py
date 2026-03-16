"""
Web Search Tool
===============

Purpose:
    Performs internet searches using DuckDuckGo and returns structured
    results with titles, URLs, and snippets.

Inputs:
    - query: The search query string.
    - max_results: Maximum number of results to return.
    - region: DuckDuckGo region code.
    - safesearch: Safe search level.

Outputs:
    - A list of dictionaries, each containing 'title', 'url', and 'snippet'.

Role in Architecture:
    Core tool used by the Search Agent to retrieve web results without
    any paid API. Wraps the duckduckgo-search library.
"""

from typing import Optional

from duckduckgo_search import DDGS

from utils.logger import get_logger

logger = get_logger(__name__)


def search_web(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
) -> list[dict]:
    """Search the web using DuckDuckGo and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        region: DuckDuckGo region code (e.g., 'wt-wt' for worldwide).
        safesearch: Safe search level ('on', 'moderate', 'off').

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    logger.info("Searching: '%s' (max_results=%d)", query, max_results)
    results = []

    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )

            for r in raw_results:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )

        logger.info("Found %d results for '%s'", len(results), query)
    except Exception as e:
        logger.error("Search failed for '%s': %s", query, str(e))

    return results


def search_multiple_queries(
    queries: list[str],
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
) -> dict[str, list[dict]]:
    """Run multiple search queries and return results keyed by query.

    Args:
        queries: List of search query strings.
        max_results: Maximum results per query.
        region: DuckDuckGo region code.
        safesearch: Safe search setting.

    Returns:
        Dictionary mapping each query to its list of result dicts.
    """
    all_results = {}
    for query in queries:
        all_results[query] = search_web(query, max_results, region, safesearch)
    return all_results
