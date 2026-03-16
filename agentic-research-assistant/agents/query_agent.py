"""
Query Understanding Agent
=========================

Purpose:
    Converts a user's natural language research prompt into structured
    research goals, key topics, and search queries using the LLM.

Inputs:
    - state: LangGraph state dict containing the user's 'question'.

Outputs:
    - Updated state with 'research_goal', 'key_topics', and 'search_queries'.

Role in Architecture:
    First agent in the pipeline. Transforms free-form user input into
    structured data that downstream agents (Planner, Search) can act on.
"""

import json

from llm.prompts import QUERY_UNDERSTANDING_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_response(text: str) -> dict:
    """Attempt to extract a JSON object from LLM output.

    Handles cases where the model wraps JSON in markdown code fences
    or includes preamble text.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dictionary, or a fallback dict on failure.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {}


def query_understanding_agent(state: dict, llm) -> dict:
    """Analyze the user's question and produce structured research parameters.

    Args:
        state: Workflow state containing 'question' key.
        llm: ModelLoader instance for text generation.

    Returns:
        Updated state with research_goal, key_topics, and search_queries.
    """
    question = state.get("question", "")
    logger.info("Query Understanding Agent processing: '%s'", question[:100])

    prompt = QUERY_UNDERSTANDING_PROMPT.format(question=question)
    response = llm.generate(prompt)
    parsed = parse_json_response(response)

    state["research_goal"] = parsed.get("research_goal", question)
    state["key_topics"] = parsed.get("key_topics", [question])
    state["search_queries"] = parsed.get("search_queries", [question])

    logger.info(
        "Research goal: %s | Queries: %s",
        state["research_goal"],
        state["search_queries"],
    )
    return state
