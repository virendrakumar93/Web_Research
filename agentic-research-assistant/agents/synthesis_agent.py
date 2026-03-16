"""
Research Synthesizer Agent
==========================

Purpose:
    Combines insights from multiple sources, removes duplicates, and
    organizes the final set of findings. Optionally uses the LLM for
    intelligent deduplication when there are many insights.

Inputs:
    - state: LangGraph state dict with 'extracted_insights' and
      'research_goal'.
    - max_sources: Maximum number of final entries.

Outputs:
    - Updated state with 'synthesized_insights' — a deduplicated,
      organized list of insight dicts, and 'results_table' — a
      formatted markdown table string.

Role in Architecture:
    Sixth (final processing) agent. Produces the finished research
    output that the Table Generator / UI will display.
"""

import json

import pandas as pd

from agents.extraction_agent import parse_json_list
from llm.prompts import SYNTHESIS_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def _basic_dedup(insights: list[dict]) -> list[dict]:
    """Simple deduplication based on insight text similarity.

    Args:
        insights: List of insight dicts.

    Returns:
        Deduplicated list.
    """
    seen = set()
    unique = []
    for item in insights:
        key = item.get("insight", "").strip().lower()[:100]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def synthesis_agent(state: dict, llm, max_sources: int = 10) -> dict:
    """Synthesize and deduplicate extracted insights.

    For small sets (<= max_sources), uses basic deduplication.
    For larger sets, uses the LLM for intelligent merging.

    Args:
        state: Workflow state with extracted_insights and research_goal.
        llm: ModelLoader instance for text generation.
        max_sources: Maximum entries in the final output.

    Returns:
        Updated state with synthesized_insights and results_table.
    """
    insights = state.get("extracted_insights", [])
    research_goal = state.get("research_goal", "")

    logger.info("Synthesis Agent processing %d insights", len(insights))

    if not insights:
        state["synthesized_insights"] = []
        state["results_table"] = "No results found."
        return state

    # Basic dedup first
    unique = _basic_dedup(insights)

    if len(unique) <= max_sources:
        synthesized = unique
    else:
        # Use LLM for intelligent synthesis when there are many insights
        prompt = SYNTHESIS_PROMPT.format(
            research_goal=research_goal,
            insights=json.dumps(unique, indent=2),
            max_sources=max_sources,
        )
        response = llm.generate(prompt)
        synthesized = parse_json_list(response)

        if not synthesized:
            # Fallback: just take the first max_sources entries
            synthesized = unique[:max_sources]

    state["synthesized_insights"] = synthesized

    # Generate the results table
    state["results_table"] = generate_table(synthesized)
    logger.info("Synthesis complete: %d final insights", len(synthesized))
    return state


def generate_table(insights: list[dict]) -> str:
    """Format insights into a markdown table.

    Args:
        insights: List of insight dicts with topic, insight, evidence,
                  source, url keys.

    Returns:
        Markdown-formatted table string.
    """
    if not insights:
        return "No results found."

    rows = []
    for item in insights:
        rows.append(
            {
                "Topic": item.get("topic", ""),
                "Key Insight": item.get("insight", ""),
                "Evidence": item.get("evidence", ""),
                "Source": item.get("source", ""),
                "Link": item.get("url", ""),
            }
        )

    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def format_as_bullets(insights: list[dict], research_goal: str) -> str:
    """Format insights as a bullet-point summary.

    Args:
        insights: List of insight dicts.
        research_goal: The original research goal.

    Returns:
        Markdown bullet-point string.
    """
    if not insights:
        return "No results found."

    lines = [f"## Research: {research_goal}\n"]
    current_topic = ""

    for item in insights:
        topic = item.get("topic", "General")
        if topic != current_topic:
            lines.append(f"\n### {topic}")
            current_topic = topic
        insight = item.get("insight", "")
        evidence = item.get("evidence", "")
        source = item.get("source", "")
        url = item.get("url", "")
        lines.append(f"- **{insight}**")
        if evidence:
            lines.append(f"  - Evidence: {evidence}")
        if source:
            lines.append(f"  - Source: [{source}]({url})")

    return "\n".join(lines)
