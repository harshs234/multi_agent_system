"""Head agent — always-on supervisor.

Two jobs:
  1. monitor()  : live dashboard of every agent's status + token economy
                  (runs in its own thread the whole time work is happening)
  2. assess()   : an LLM pass over the collected stats (the 'monitoring' report)

The head starts even when you run a single worker.
"""

import time
import threading

from agents.base import BaseAgent
from core.config import MODEL
from core.registry import Registry


class HeadAgent(BaseAgent):
    name = "head"
    model = "gemini/gemini-2.5-flash"
    max_tokens = 512
    effort = "low"
    budget = 2000
    depends_on = []

    system = (
        "You are a supervisor agent overseeing a fleet of worker agents. "
        "You are given each worker's status and token economy. In 4-6 sentences, "
        "assess efficiency, call out any agent that went over budget or was the "
        "costliest, and suggest one concrete optimization (model choice, effort "
        "level, token cap, or prompt-caching). Be specific and brief."
    )

    # ------------------------------------------------------------------ #
    # 1. Live monitor
    # ------------------------------------------------------------------ #
    def monitor(self, done_event: threading.Event) -> None:
        icon = {"pending": "·", "running": "▶", "done": "✓", "error": "✗"}
        while True:
            snap = self.reg.snapshot()
            out = ["\033[H\033[J" + "═" * 78,
                   " HEAD AGENT — live monitor",
                   "═" * 78,
                   f" {'agent':<12}{'status':<10}{'tok/budget':>14}{'cost $':>11}{'t(s)':>7}",
                   " " + "─" * 76]
            total = 0.0
            for s in snap:
                total += s.cost
                budget = s.budget if s.budget is not None else "∞"
                tok = f"{s.total_tokens}/{budget}"
                flag = " !" if s.over_budget else ""
                out.append(
                    f" {s.name:<12}{icon.get(s.status, '?') + ' ' + s.status:<10}"
                    f"{tok:>14}{s.cost:>11.5f}{s.elapsed:>7.1f}{flag}"
                )
            out += [" " + "─" * 76, f" {'TOTAL':<12}{'':<10}{'':>14}{total:>11.5f}"]
            print("\n".join(out), flush=True)
            if done_event.is_set():
                break
            time.sleep(0.4)

    # ------------------------------------------------------------------ #
    # 2. LLM assessment — reuses BaseAgent.run via build_user below
    # ------------------------------------------------------------------ #
    def build_user(self, topic: str, context: dict[str, str]) -> str:
        snap = [s for s in self.reg.snapshot() if s.name != self.name]
        report = "\n".join(
            f"- {s.name} ({s.model}): {s.status}, total_tokens={s.total_tokens} "
            f"budget={s.budget} over_budget={s.over_budget} "
            f"cost=${s.cost:.5f} time={s.elapsed:.1f}s"
            for s in snap
        )
        total = sum(s.cost for s in snap)
        return f"Run report (total cost ${total:.5f}):\n{report}"

    def assess(self) -> str:
        return self.run(topic="", context={})
