"""Researcher — independent worker. Produces key facts on a topic."""

from agents.base import BaseAgent
from core.config import MODEL


class ResearcherAgent(BaseAgent):
    name = "researcher"
    model = "gemini/gemini-2.5-flash"
    max_tokens = 1024       # <-- this agent's output-token cap
    effort = "low"
    budget = 2000           # <-- soft total-token budget
    depends_on = []

    system = (
        "You are a research agent. Return exactly 5 terse, factual bullet points. "
        "No preamble, no conclusion."
    )

    def build_user(self, topic: str, context: dict[str, str]) -> str:
        return f"Give 5 key facts about: {topic}"
