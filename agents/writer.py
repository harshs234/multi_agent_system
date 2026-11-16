"""Writer — depends on the Researcher. Turns its facts into a summary."""

from agents.base import BaseAgent
from core.config import MODEL


class WriterAgent(BaseAgent):
    name = "writer"
    model = "gemini/gemini-2.5-flash"
    max_tokens = 1024       # <-- this agent's output-token cap
    effort = "low"
    budget = 2000
    depends_on = ["researcher"]   # <-- dependency: needs researcher's output

    system = (
        "You are a writing agent. Write a single tight ~90-word paragraph. "
        "No headings, no lists."
    )

    def build_user(self, topic: str, context: dict[str, str]) -> str:
        facts = context.get("researcher", "(no facts provided)")
        return f"Using ONLY these facts, write a summary about {topic}:\n{facts}"
