"""BaseAgent: shared run logic. Each concrete agent subclasses this and sets
its own token economy (model / max_tokens / effort / budget) and prompt.

The LLM call goes through core.llm (LiteLLM) so any provider works."""

import time

from core.config import MODEL
from core.registry import Registry
from core.logger import RunLogger
from core.llm import complete


class BaseAgent:
    # --- identity + token economy (override per agent) ---------------------- #
    name: str = "base"
    model: str = "gemini/gemini-2.5-flash"             # LiteLLM model string: "provider/model"
    max_tokens: int = 1024          # HARD cap on this agent's output tokens
    effort: str | None = "low"      # reasoning effort; dropped if provider lacks it
    budget: int | None = None       # soft total-token budget (flagged if exceeded)
    depends_on: list[str] = []      # names of agents whose output this one needs

    # --- prompt ------------------------------------------------------------- #
    system: str = ""

    def __init__(self, reg: Registry, logger: RunLogger) -> None:
        self.reg = reg
        self.logger = logger

    def build_user(self, topic: str, context: dict[str, str]) -> str:
        """Return the user message. `context` holds outputs of depends_on agents."""
        raise NotImplementedError

    def run(self, topic: str, context: dict[str, str]) -> str:
        self.reg.set_status(self.name, "running", started=time.time())
        try:
            result = complete(
                model=self.model,
                system=self.system,
                user=self.build_user(topic, context),
                max_tokens=self.max_tokens,   # <-- hard output-token cap per agent
                effort=self.effort,
            )
            self.reg.record_usage(self.name, result)
            self.reg.set_status(self.name, "done", ended=time.time(), output=result.text)
            return result.text
        except Exception as e:  # noqa: BLE001  (surface any failure in dashboard + log)
            self.reg.set_status(self.name, "error", ended=time.time(), error=str(e))
            return ""
        finally:
            self.logger.log_event(self.reg.stats[self.name])       # diagnostics -> logs/
            self.logger.save_output(self.reg.stats[self.name])     # actual output -> outputs/
