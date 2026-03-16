"""
Research Workflow Graph
=======================

Purpose:
    Orchestrates the multi-agent research pipeline using LangGraph.
    Defines the state schema, wires up all agents as graph nodes,
    and provides the main `run_research` entry point.

Inputs:
    - user_question: The research prompt from the user.
    - config: Full application config dictionary.
    - llm: Loaded ModelLoader instance.
    - conversation_memory: Optional ConversationMemory for follow-ups.

Outputs:
    - Final state dictionary containing synthesized_insights and
      results_table ready for display.

Role in Architecture:
    Central orchestration layer. This is the "brain" that sequences
    all agents in the correct order: Query Understanding → Planning →
    Search → Scraping → Extraction → Synthesis.
"""

from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from agents.query_agent import query_understanding_agent
from agents.planner_agent import planner_agent
from agents.search_agent import search_agent
from agents.scraper_agent import scraper_agent
from agents.extraction_agent import extraction_agent
from agents.synthesis_agent import synthesis_agent
from utils.logger import get_logger

logger = get_logger(__name__)


class ResearchState(TypedDict, total=False):
    """State schema for the research workflow graph.

    All agents read from and write to this shared state dictionary.
    """
    question: str
    research_goal: str
    key_topics: list[str]
    search_queries: list[str]
    expected_fields: list[str]
    focus_areas: list[str]
    search_results: list[dict]
    scraped_content: list[dict]
    extracted_insights: list[dict]
    synthesized_insights: list[dict]
    results_table: str
    error: str


def build_research_graph(config: dict, llm) -> StateGraph:
    """Build and compile the LangGraph research workflow.

    Creates a linear pipeline of agent nodes connected in sequence:
    query_understanding → planning → searching → scraping → extraction → synthesis

    Args:
        config: Full application configuration dictionary.
        llm: Loaded ModelLoader instance.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    search_config = config.get("search", {})
    scraper_config = config.get("scraper", {})
    text_config = config.get("text", {})
    research_config = config.get("research", {})
    max_sources = research_config.get("max_sources", 10)
    max_urls_per_query = research_config.get("max_urls_per_query", 3)

    # Define node functions that close over config and llm

    def node_query_understanding(state: ResearchState) -> ResearchState:
        try:
            return query_understanding_agent(state, llm)
        except Exception as e:
            logger.error("Query Understanding failed: %s", e)
            state["error"] = f"Query understanding failed: {e}"
            # Fallback: use the question directly
            state["research_goal"] = state.get("question", "")
            state["key_topics"] = [state.get("question", "")]
            state["search_queries"] = [state.get("question", "")]
            return state

    def node_planning(state: ResearchState) -> ResearchState:
        try:
            return planner_agent(state, llm)
        except Exception as e:
            logger.error("Planning failed: %s", e)
            # Fallback: use defaults
            state["expected_fields"] = [
                "Topic", "Key Insight", "Evidence", "Source", "Link"
            ]
            state["focus_areas"] = state.get("key_topics", [])
            return state

    def node_searching(state: ResearchState) -> ResearchState:
        try:
            return search_agent(state, search_config)
        except Exception as e:
            logger.error("Search failed: %s", e)
            state["search_results"] = []
            state["error"] = f"Search failed: {e}"
            return state

    def node_scraping(state: ResearchState) -> ResearchState:
        try:
            return scraper_agent(state, scraper_config, max_urls_per_query)
        except Exception as e:
            logger.error("Scraping failed: %s", e)
            state["scraped_content"] = []
            return state

    def node_extraction(state: ResearchState) -> ResearchState:
        try:
            return extraction_agent(state, llm, text_config)
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            state["extracted_insights"] = []
            return state

    def node_synthesis(state: ResearchState) -> ResearchState:
        try:
            return synthesis_agent(state, llm, max_sources)
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            state["synthesized_insights"] = []
            state["results_table"] = "Research completed but synthesis failed."
            return state

    # Build the graph
    graph = StateGraph(ResearchState)

    graph.add_node("query_understanding", node_query_understanding)
    graph.add_node("planning", node_planning)
    graph.add_node("searching", node_searching)
    graph.add_node("scraping", node_scraping)
    graph.add_node("extraction", node_extraction)
    graph.add_node("synthesis", node_synthesis)

    # Wire the linear pipeline
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "planning")
    graph.add_edge("planning", "searching")
    graph.add_edge("searching", "scraping")
    graph.add_edge("scraping", "extraction")
    graph.add_edge("extraction", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_research(question: str, config: dict, llm) -> dict:
    """Execute the full research pipeline for a given question.

    Args:
        question: The user's research question.
        config: Full application configuration dictionary.
        llm: Loaded ModelLoader instance.

    Returns:
        Final state dictionary with results_table and synthesized_insights.
    """
    logger.info("Starting research pipeline for: '%s'", question[:100])

    graph = build_research_graph(config, llm)
    initial_state: ResearchState = {"question": question}
    final_state = graph.invoke(initial_state)

    logger.info("Research pipeline complete.")
    return final_state
