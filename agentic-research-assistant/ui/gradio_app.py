"""
Gradio User Interface
=====================

Purpose:
    Provides a web-based GUI for the Multi-Agent Research Assistant
    using Gradio. Includes a main research interface with table output
    and a follow-up question interface for refining results.

Inputs:
    - config: Application configuration dictionary.
    - llm: Loaded ModelLoader instance.

Outputs:
    - A Gradio Blocks app object ready to launch.

Role in Architecture:
    Presentation layer. Connects the user's browser to the research
    workflow and conversation memory, displaying results in tables
    and supporting multi-turn interaction.
"""

import json

import gradio as gr
import pandas as pd

from workflows.research_graph import run_research
from memory.conversation_memory import ConversationMemory
from agents.synthesis_agent import format_as_bullets, generate_table
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app(config: dict, llm) -> gr.Blocks:
    """Create and configure the Gradio application.

    Args:
        config: Full application configuration dictionary.
        llm: Loaded ModelLoader instance.

    Returns:
        A Gradio Blocks app ready to launch.
    """
    memory = ConversationMemory()

    def run_research_handler(question: str, output_format: str):
        """Handle the main research button click.

        Args:
            question: User's research prompt.
            output_format: Desired output format (table, bullets, report).

        Returns:
            Tuple of (status_message, results_output, raw_data_json).
        """
        if not question.strip():
            return "Please enter a research question.", "", ""

        try:
            yield "Researching... This may take a few minutes.", "", ""

            state = run_research(question, config, llm)

            insights = state.get("synthesized_insights", [])
            research_goal = state.get("research_goal", question)

            # Store in memory for follow-ups
            memory.add_turn(
                question=question,
                research_goal=research_goal,
                insights=insights,
                results_table=state.get("results_table", ""),
            )

            # Format output based on user selection
            if output_format == "Bullet Points":
                output = format_as_bullets(insights, research_goal)
            elif output_format == "Table":
                output = state.get("results_table", "No results found.")
            else:
                output = state.get("results_table", "No results found.")

            raw_json = json.dumps(insights, indent=2) if insights else "[]"

            status = f"Research complete — {len(insights)} insights found."
            yield status, output, raw_json

        except Exception as e:
            logger.error("Research failed: %s", e)
            yield f"Error: {str(e)}", "", ""

    def followup_handler(followup_question: str, output_format: str):
        """Handle follow-up question submission.

        Args:
            followup_question: The follow-up question.
            output_format: Desired output format.

        Returns:
            Tuple of (status_message, results_output, raw_data_json).
        """
        if not followup_question.strip():
            return "Please enter a follow-up question.", "", ""

        if not memory.has_context():
            return "No previous research to follow up on. Run a research query first.", "", ""

        try:
            yield "Processing follow-up...", "", ""

            # Process the follow-up to get refined queries
            followup_info = memory.process_followup(followup_question, llm)

            # Run new research with the updated goal
            state = run_research(followup_info["updated_goal"], config, llm)

            insights = state.get("synthesized_insights", [])
            research_goal = state.get("research_goal", followup_question)

            # If reusing previous findings, merge them
            if followup_info.get("reuse_previous") and memory.has_context():
                prev = memory.get_previous_context()
                # Previous insights are already in memory; new ones extend them
                pass

            memory.add_turn(
                question=followup_question,
                research_goal=research_goal,
                insights=insights,
                results_table=state.get("results_table", ""),
            )

            if output_format == "Bullet Points":
                output = format_as_bullets(insights, research_goal)
            else:
                output = state.get("results_table", "No results found.")

            raw_json = json.dumps(insights, indent=2) if insights else "[]"

            status = f"Follow-up complete — {len(insights)} insights found."
            yield status, output, raw_json

        except Exception as e:
            logger.error("Follow-up failed: %s", e)
            yield f"Error: {str(e)}", "", ""

    def clear_handler():
        """Clear conversation memory and outputs."""
        memory.clear()
        return "", "", "", ""

    # Build the Gradio UI
    with gr.Blocks(
        title="Multi-Agent Research Assistant",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Multi-Agent Research Assistant
            *Autonomous internet research powered by open-source LLMs and multi-agent orchestration.*

            Enter a research topic below and the system will:
            1. Understand your question and plan the research
            2. Search the internet using DuckDuckGo
            3. Scrape and extract information from relevant pages
            4. Synthesize findings into a structured table with citations
            """
        )

        with gr.Tab("Research"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Research Question",
                        placeholder="e.g., Compare the top 5 open-source LLMs released in 2024 by performance and parameter count",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    output_format = gr.Radio(
                        choices=["Table", "Bullet Points"],
                        value="Table",
                        label="Output Format",
                    )

            with gr.Row():
                research_btn = gr.Button("Run Research", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", scale=1)

            status_output = gr.Textbox(label="Status", interactive=False)
            results_output = gr.Markdown(label="Results")

            with gr.Accordion("Raw Data (JSON)", open=False):
                raw_output = gr.Code(language="json", label="Raw Insights")

        with gr.Tab("Follow-up"):
            gr.Markdown(
                """
                ### Ask a Follow-up Question
                Refine or extend your previous research. The system remembers
                your prior results and uses them as context.
                """
            )
            followup_input = gr.Textbox(
                label="Follow-up Question",
                placeholder="e.g., Which of those models supports the longest context window?",
                lines=2,
            )
            followup_format = gr.Radio(
                choices=["Table", "Bullet Points"],
                value="Table",
                label="Output Format",
            )
            followup_btn = gr.Button("Ask Follow-up", variant="primary")
            followup_status = gr.Textbox(label="Status", interactive=False)
            followup_results = gr.Markdown(label="Follow-up Results")

            with gr.Accordion("Raw Data (JSON)", open=False):
                followup_raw = gr.Code(language="json", label="Raw Insights")

        # Wire up event handlers
        research_btn.click(
            fn=run_research_handler,
            inputs=[question_input, output_format],
            outputs=[status_output, results_output, raw_output],
        )

        followup_btn.click(
            fn=followup_handler,
            inputs=[followup_input, followup_format],
            outputs=[followup_status, followup_results, followup_raw],
        )

        clear_btn.click(
            fn=clear_handler,
            inputs=[],
            outputs=[question_input, status_output, results_output, raw_output],
        )

    return app
