"""
Prompt Templates
================

Purpose:
    Stores all prompt templates used by the specialized agents.
    Centralizing prompts here keeps agent logic clean and makes
    prompt engineering easier.

Inputs:
    - Template variables filled in at call time by each agent.

Outputs:
    - Formatted prompt strings ready for LLM consumption.

Role in Architecture:
    Shared resource consumed by every agent module. Changing a prompt
    here automatically updates the corresponding agent behavior.
"""

QUERY_UNDERSTANDING_PROMPT = """You are a research query analyst. Given a user's research question, produce a JSON object with:
- "research_goal": a one-sentence summary of what needs to be researched
- "key_topics": a list of 2-4 core topics or entities to investigate
- "search_queries": a list of 2-3 effective web search queries to find relevant information

User question: {question}

Respond ONLY with valid JSON, no extra text."""


RESEARCH_PLANNER_PROMPT = """You are a research planner. Given the research goal and key topics below, create a structured research plan as a JSON object with:
- "queries": a list of 2-3 specific search queries to execute
- "expected_fields": a list of column names for the final results table (e.g., ["Topic", "Key Insight", "Evidence", "Source", "Link"])
- "focus_areas": a list of specific aspects to focus on during extraction

Research goal: {research_goal}
Key topics: {key_topics}

Respond ONLY with valid JSON, no extra text."""


EXTRACTION_PROMPT = """You are an information extraction specialist. Given the following text scraped from a webpage, extract structured research insights relevant to the research goal.

Research goal: {research_goal}
Focus areas: {focus_areas}

Source URL: {url}
Source title: {title}

Scraped content:
{content}

Extract insights as a JSON list of objects. Each object must have:
- "topic": the specific topic or entity discussed
- "insight": the key finding or claim
- "evidence": supporting data, quote, or fact
- "source": the name or title of the source
- "url": "{url}"

Return ONLY a JSON list. If no relevant information is found, return an empty list []."""


SYNTHESIS_PROMPT = """You are a research synthesizer. Given the following extracted insights from multiple sources, produce a deduplicated and organized JSON list of the most important findings.

Research goal: {research_goal}

Extracted insights:
{insights}

Rules:
1. Remove duplicate or near-duplicate insights.
2. Keep the most informative version when merging.
3. Ensure every entry has all fields: topic, insight, evidence, source, url.
4. Order by relevance to the research goal.
5. Return at most {max_sources} entries.

Return ONLY a valid JSON list."""


FOLLOWUP_PROMPT = """You are a research assistant continuing a conversation. The user has a follow-up question about previous research.

Previous research goal: {previous_goal}
Previous findings summary:
{previous_findings}

Follow-up question: {followup_question}

Determine what additional research is needed. Produce a JSON object with:
- "updated_goal": the refined research goal incorporating the follow-up
- "additional_queries": a list of 1-2 new search queries to answer the follow-up
- "reuse_previous": true if previous findings are still relevant, false otherwise

Respond ONLY with valid JSON, no extra text."""


FORMAT_BULLET_PROMPT = """Convert the following research data into a clear bullet-point summary grouped by topic. Include source citations.

Research goal: {research_goal}
Data:
{data}

Format as markdown bullet points."""


FORMAT_REPORT_PROMPT = """Convert the following research data into a structured markdown report with sections, summaries, and citations.

Research goal: {research_goal}
Data:
{data}

Include an Executive Summary, Detailed Findings by topic, and a Sources section."""
