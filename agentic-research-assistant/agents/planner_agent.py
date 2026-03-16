"""
Research Planner Agent
======================

Purpose:
    Takes the structured research goal and key topics from the Query
    Understanding Agent and creates a detailed research plan including
    refined search queries, expected output fields, and focus areas.

Inputs:
    - state: LangGraph state dict with 'research_goal' and 'key_topics'.

Outputs:
    - Updated state with refined 'search_queries', 'expected_fields',
      and 'focus_areas'.

Role in Architecture:
    Second agent in the pipeline. Refines and enriches the research
    parameters before the Search Agent begins retrieving results.
"""

import json

from agents.query_agent import parse_json_response
from llm.prompts import RESEARCH_PLANNER_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def planner_agent(state: dict, llm) -> dict:
    """Create a research plan with queries, fields, and focus areas.

    Args:
        state: Workflow state with research_goal and key_topics.
        llm: ModelLoader instance for text generation.

    Returns:
        Updated state with search_queries, expected_fields, and focus_areas.
    """
    research_goal = state.get("research_goal", "")
    key_topics = state.get("key_topics", [])

    logger.info("Planner Agent creating research plan for: '%s'", research_goal[:100])

    prompt = RESEARCH_PLANNER_PROMPT.format(
        research_goal=research_goal,
        key_topics=json.dumps(key_topics),
    )
    response = llm.generate(prompt)
    parsed = parse_json_response(response)

    # Use planner queries if available, otherwise keep existing ones
    if parsed.get("queries"):
        state["search_queries"] = parsed["queries"]

    state["expected_fields"] = parsed.get(
        "expected_fields",
        ["Topic", "Key Insight", "Evidence", "Source", "Link"],
    )
    state["focus_areas"] = parsed.get("focus_areas", key_topics)

    logger.info(
        "Plan ready — %d queries, fields: %s",
        len(state["search_queries"]),
        state["expected_fields"],
    )
    return state
