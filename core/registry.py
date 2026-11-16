"""Thread-safe shared state + token/cost ledger.

Every agent writes its usage here; the head agent reads it to monitor the fleet.
"""

import time
import threading
from dataclasses import dataclass

from core.llm import LLMResult


@dataclass
class AgentStat:
    name: str
    model: str
    status: str = "pending"          # pending | running | done | error
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    budget: int | None = None        # soft total-token budget for this agent
    started: float | None = None
    ended: float | None = None
    output: str = ""
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write

    @property
    def over_budget(self) -> bool:
        return self.budget is not None and self.total_tokens > self.budget

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.time()) - self.started


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stats: dict[str, AgentStat] = {}

    def register(self, name: str, model: str, budget: int | None = None) -> None:
        with self._lock:
            self.stats[name] = AgentStat(name=name, model=model, budget=budget)

    def set_status(self, name: str, status: str, **kw) -> None:
        with self._lock:
            s = self.stats[name]
            s.status = status
            for k, v in kw.items():
                setattr(s, k, v)

    def record_usage(self, name: str, result: LLMResult) -> None:
        with self._lock:
            s = self.stats[name]
            s.input_tokens += result.input_tokens
            s.output_tokens += result.output_tokens
            s.cache_read += result.cache_read
            s.cache_write += result.cache_write
            s.cost += result.cost

    def snapshot(self) -> list[AgentStat]:
        with self._lock:
            return [AgentStat(**vars(s)) for s in self.stats.values()]
