"""Classifier — independent easy task. Sentiment-tags a batch of reviews."""

from agents.base import BaseAgent
from core.config import MODEL


class ClassifierAgent(BaseAgent):
    name = "classifier"
    model = "gemini/gemini-2.5-flash"           # e.g. "openai/gpt-4o-mini" or "gemini/gemini-1.5-flash" to cut cost
    max_tokens = 1024       # reasoning models (e.g. gemini-2.5-flash) spend output
                            # tokens on thinking first — 256 truncated the answer
    effort = None           # trivial task: no reasoning needed, saves tokens
    budget = 2000
    depends_on = []

    system = (
        "You classify sentiment. Reply with one line per input: 'positive' or 'negative'."
    )

    REVIEWS = [
        "Shipping was fast but the product broke in a week.",
        "Absolutely love it, best purchase this year!",
        "Meh. Does the job, nothing special.",
        "Terrible support, still waiting for a refund.",
        "Works exactly as described, very happy.",
    ]

    def build_user(self, topic: str, context: dict[str, str]) -> str:
        numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.REVIEWS))
        return "Classify each review:\n" + numbered
