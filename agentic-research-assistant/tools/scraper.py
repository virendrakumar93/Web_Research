"""
Web Scraper Tool
================

Purpose:
    Downloads web pages and extracts clean, readable text content
    using trafilatura as the primary extractor with BeautifulSoup
    as a fallback.

Inputs:
    - url: The URL to scrape.
    - timeout: HTTP request timeout in seconds.
    - max_content_length: Maximum characters to return.

Outputs:
    - A dictionary with 'url', 'title', 'content', and 'success' fields.

Role in Architecture:
    Core tool used by the Web Scraper Agent to convert raw web pages
    into clean text suitable for LLM processing.
"""

from typing import Optional

import requests
import trafilatura
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def scrape_url(
    url: str,
    timeout: int = 15,
    max_content_length: int = 10000,
    user_agent: Optional[str] = None,
) -> dict:
    """Scrape a URL and extract clean text content.

    Uses trafilatura for primary extraction with BeautifulSoup as fallback.

    Args:
        url: The URL to scrape.
        timeout: HTTP request timeout in seconds.
        max_content_length: Maximum characters of content to return.
        user_agent: Custom User-Agent header string.

    Returns:
        Dict with keys: url, title, content, success.
    """
    result = {"url": url, "title": "", "content": "", "success": False}
    headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        html = response.text

        # Extract title with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""

        # Primary extraction with trafilatura
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        # Fallback to BeautifulSoup if trafilatura fails
        if not content:
            logger.info("Trafilatura returned empty for %s, using BS4 fallback", url)
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)

        if content:
            result["content"] = content[:max_content_length]
            result["success"] = True
            logger.info(
                "Scraped %s: %d chars extracted", url, len(result["content"])
            )
        else:
            logger.warning("No content extracted from %s", url)

    except requests.exceptions.Timeout:
        logger.error("Timeout scraping %s", url)
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for %s: %s", url, str(e))
    except Exception as e:
        logger.error("Unexpected error scraping %s: %s", url, str(e))

    return result


def scrape_multiple_urls(
    urls: list[str],
    timeout: int = 15,
    max_content_length: int = 10000,
    user_agent: Optional[str] = None,
) -> list[dict]:
    """Scrape multiple URLs and return results.

    Args:
        urls: List of URLs to scrape.
        timeout: HTTP request timeout in seconds.
        max_content_length: Maximum characters per page.
        user_agent: Custom User-Agent header string.

    Returns:
        List of result dicts from scrape_url.
    """
    results = []
    for url in urls:
        result = scrape_url(url, timeout, max_content_length, user_agent)
        results.append(result)
    return results
