"""
Information Extraction Agent
============================

Purpose:
    Uses the LLM to extract structured research insights from each
    scraped webpage. Produces topic/insight/evidence/source/url records.

Inputs:
    - state: LangGraph state dict with 'scraped_content', 'research_goal',
      and 'focus_areas'.

Outputs:
    - Updated state with 'extracted_insights' — a flat list of insight dicts.

Role in Architecture:
    Fifth agent in the pipeline. Transforms raw text into structured
    data that the Synthesis Agent can merge and deduplicate.
"""

import json

from llm.prompts import EXTRACTION_PROMPT
from tools.text_cleaner import clean_text, chunk_text
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_list(text: str) -> list:
    """Extract a JSON list from LLM output.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed list, or empty list on failure.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return []


def extraction_agent(state: dict, llm, text_config: dict) -> dict:
    """Extract structured insights from scraped content using the LLM.

    Args:
        state: Workflow state with scraped_content, research_goal, focus_areas.
        llm: ModelLoader instance for text generation.
        text_config: Dict with chunk_size and chunk_overlap.

    Returns:
        Updated state with extracted_insights list.
    """
    scraped = state.get("scraped_content", [])
    research_goal = state.get("research_goal", "")
    focus_areas = state.get("focus_areas", [])

    chunk_size = text_config.get("chunk_size", 2000)
    chunk_overlap = text_config.get("chunk_overlap", 200)

    logger.info("Extraction Agent processing %d scraped pages", len(scraped))

    all_insights = []

    for page in scraped:
        if not page.get("success") or not page.get("content"):
            continue

        url = page.get("url", "")
        title = page.get("title", "")
        content = clean_text(page["content"])

        # Chunk long content so each piece fits the LLM context
        chunks = chunk_text(content, chunk_size, chunk_overlap)

        for chunk in chunks[:2]:  # Limit to first 2 chunks per page
            prompt = EXTRACTION_PROMPT.format(
                research_goal=research_goal,
                focus_areas=json.dumps(focus_areas),
                url=url,
                title=title,
                content=chunk,
            )

            response = llm.generate(prompt)
            insights = parse_json_list(response)

            # Validate and sanitize each insight
            for insight in insights:
                if isinstance(insight, dict) and insight.get("insight"):
                    insight.setdefault("topic", "General")
                    insight.setdefault("evidence", "")
                    insight.setdefault("source", title)
                    insight["url"] = url
                    all_insights.append(insight)

    state["extracted_insights"] = all_insights
    logger.info("Extraction Agent produced %d insights", len(all_insights))
    return state
