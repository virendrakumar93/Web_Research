"""
Search Agent
============

Purpose:
    Executes web searches using DuckDuckGo for each planned query and
    collects the result URLs, titles, and snippets.

Inputs:
    - state: LangGraph state dict with 'search_queries'.
    - config: Search configuration (max_results, region, safesearch).

Outputs:
    - Updated state with 'search_results' — a flat list of result dicts.

Role in Architecture:
    Third agent in the pipeline. Bridges the research plan to actual
    internet data by performing the searches and collecting URLs for
    the Scraper Agent to process.
"""

from tools.web_search import search_web
from utils.logger import get_logger

logger = get_logger(__name__)


def search_agent(state: dict, search_config: dict) -> dict:
    """Execute web searches for all planned queries.

    Args:
        state: Workflow state with search_queries list.
        search_config: Dict with max_results, region, safesearch.

    Returns:
        Updated state with search_results list.
    """
    queries = state.get("search_queries", [])
    max_results = search_config.get("max_results", 5)
    region = search_config.get("region", "wt-wt")
    safesearch = search_config.get("safesearch", "moderate")

    logger.info("Search Agent executing %d queries", len(queries))

    all_results = []
    seen_urls = set()

    for query in queries:
        results = search_web(
            query=query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
        )
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                result["query"] = query
                all_results.append(result)

    state["search_results"] = all_results
    logger.info("Search Agent collected %d unique results", len(all_results))
    return state
