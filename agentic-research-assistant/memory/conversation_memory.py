"""
Conversation Memory Module
==========================

Purpose:
    Maintains conversation history across research sessions, enabling
    follow-up questions that build on previous research results.

Inputs:
    - Research questions, goals, and synthesized insights from each
      research run.
    - Follow-up questions from the user.

Outputs:
    - Context summaries for follow-up processing.
    - Augmented queries that incorporate prior findings.

Role in Architecture:
    Stateful memory layer that sits alongside the LangGraph workflow,
    allowing the system to maintain context between user interactions.
"""

import json
from dataclasses import dataclass, field

from llm.prompts import FOLLOWUP_PROMPT
from agents.query_agent import parse_json_response
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchTurn:
    """A single turn of research conversation."""
    question: str
    research_goal: str
    insights: list[dict]
    results_table: str


class ConversationMemory:
    """Manages conversation history for multi-turn research sessions.

    Stores previous research turns and provides context for follow-up
    questions so the user can refine or extend earlier research.
    """

    def __init__(self, max_turns: int = 5):
        """Initialize conversation memory.

        Args:
            max_turns: Maximum number of previous turns to retain.
        """
        self.turns: list[ResearchTurn] = []
        self.max_turns = max_turns

    def add_turn(
        self,
        question: str,
        research_goal: str,
        insights: list[dict],
        results_table: str,
    ) -> None:
        """Record a completed research turn.

        Args:
            question: The user's original question.
            research_goal: The derived research goal.
            insights: List of synthesized insight dicts.
            results_table: The formatted results table string.
        """
        turn = ResearchTurn(
            question=question,
            research_goal=research_goal,
            insights=insights,
            results_table=results_table,
        )
        self.turns.append(turn)

        # Trim old turns if over limit
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        logger.info("Memory updated: %d turns stored", len(self.turns))

    def has_context(self) -> bool:
        """Check whether there is prior conversation context."""
        return len(self.turns) > 0

    def get_previous_context(self) -> dict:
        """Get a summary of the most recent research turn.

        Returns:
            Dict with previous_goal and previous_findings keys,
            or empty dict if no history exists.
        """
        if not self.turns:
            return {}

        last = self.turns[-1]
        # Summarize insights to avoid exceeding LLM context
        findings_summary = []
        for insight in last.insights[:10]:
            findings_summary.append(
                f"- {insight.get('topic', '')}: {insight.get('insight', '')}"
            )

        return {
            "previous_goal": last.research_goal,
            "previous_findings": "\n".join(findings_summary),
        }

    def process_followup(self, followup_question: str, llm) -> dict:
        """Process a follow-up question using prior context and the LLM.

        Args:
            followup_question: The user's follow-up question.
            llm: ModelLoader instance for text generation.

        Returns:
            Dict with updated_goal, additional_queries, and reuse_previous.
        """
        context = self.get_previous_context()
        if not context:
            return {
                "updated_goal": followup_question,
                "additional_queries": [followup_question],
                "reuse_previous": False,
            }

        prompt = FOLLOWUP_PROMPT.format(
            previous_goal=context["previous_goal"],
            previous_findings=context["previous_findings"],
            followup_question=followup_question,
        )

        response = llm.generate(prompt)
        parsed = parse_json_response(response)

        result = {
            "updated_goal": parsed.get("updated_goal", followup_question),
            "additional_queries": parsed.get(
                "additional_queries", [followup_question]
            ),
            "reuse_previous": parsed.get("reuse_previous", False),
        }

        logger.info("Follow-up processed: goal='%s'", result["updated_goal"][:100])
        return result

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns = []
        logger.info("Conversation memory cleared.")
