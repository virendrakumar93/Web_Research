"""
Web Scraper Agent
=================

Purpose:
    Downloads and extracts clean text content from the URLs found by
    the Search Agent. Uses trafilatura with BeautifulSoup fallback.

Inputs:
    - state: LangGraph state dict with 'search_results'.
    - scraper_config: Configuration with timeout, max_content_length,
      and user_agent.
    - max_urls_per_query: Limit on how many URLs to scrape per query.

Outputs:
    - Updated state with 'scraped_content' — a list of dicts containing
      url, title, content, and success flag.

Role in Architecture:
    Fourth agent in the pipeline. Converts raw URLs into clean text
    that the Information Extraction Agent can process with the LLM.
"""

from tools.scraper import scrape_url
from utils.logger import get_logger

logger = get_logger(__name__)


def scraper_agent(
    state: dict, scraper_config: dict, max_urls_per_query: int = 3
) -> dict:
    """Scrape content from search result URLs.

    Args:
        state: Workflow state with search_results.
        scraper_config: Dict with timeout, max_content_length, user_agent.
        max_urls_per_query: Max URLs to scrape per original query.

    Returns:
        Updated state with scraped_content list.
    """
    search_results = state.get("search_results", [])
    timeout = scraper_config.get("timeout", 15)
    max_content_length = scraper_config.get("max_content_length", 10000)
    user_agent = scraper_config.get("user_agent")

    logger.info("Scraper Agent processing %d URLs", len(search_results))

    # Limit URLs per query to avoid excessive scraping
    query_counts: dict[str, int] = {}
    urls_to_scrape = []
    for result in search_results:
        query = result.get("query", "default")
        query_counts[query] = query_counts.get(query, 0) + 1
        if query_counts[query] <= max_urls_per_query:
            urls_to_scrape.append(result)

    scraped = []
    for result in urls_to_scrape:
        url = result.get("url", "")
        if not url:
            continue

        data = scrape_url(
            url=url,
            timeout=timeout,
            max_content_length=max_content_length,
            user_agent=user_agent,
        )
        # Carry over the title from search if scrape didn't find one
        if not data.get("title") and result.get("title"):
            data["title"] = result["title"]

        scraped.append(data)

    state["scraped_content"] = scraped
    successful = sum(1 for s in scraped if s.get("success"))
    logger.info("Scraper Agent: %d/%d pages scraped successfully", successful, len(scraped))
    return state
