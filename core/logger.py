"""Per-run logging. One record per agent call."""

import os
import json
import threading
from datetime import datetime, timezone
from dataclasses import asdict

from core.registry import Registry, AgentStat


class RunLogger:
    """
    logs/run-YYYYMMDD-HHMMSS.jsonl : one JSON object per agent call (machine-readable)
    logs/run-YYYYMMDD-HHMMSS.txt   : human-readable summary written at the end
    """

    def __init__(self, log_dir: str, output_dir: str | None = None) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.jsonl_path = os.path.join(log_dir, f"run-{self.run_id}.jsonl")
        self.txt_path = os.path.join(log_dir, f"run-{self.run_id}.txt")
        self._lock = threading.Lock()

        # outputs/<agent>/run-<id>.txt  — one folder per agent, runs inside it
        self.output_root = output_dir

    def save_output(self, stat: AgentStat) -> None:
        """Write one agent's FULL output to outputs/<agent>/run-<id>.txt."""
        if not self.output_root or stat.status != "done":
            return
        agent_dir = os.path.join(self.output_root, stat.name)
        os.makedirs(agent_dir, exist_ok=True)
        with self._lock:
            with open(os.path.join(agent_dir, f"run-{self.run_id}.txt"), "w", encoding="utf-8") as f:
                f.write(stat.output or "")

    def write_combined_output(self, reg: Registry) -> None:
        """Write all agents' outputs for this run into outputs/_combined/run-<id>.md."""
        if not self.output_root:
            return
        combined_dir = os.path.join(self.output_root, "_combined")
        os.makedirs(combined_dir, exist_ok=True)
        parts = [f"# Run {self.run_id}\n"]
        for s in reg.snapshot():
            parts.append(f"## {s.name}  ({s.model}) — {s.status}\n")
            parts.append((s.output or s.error or "(no output)") + "\n")
        with open(os.path.join(combined_dir, f"run-{self.run_id}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    def log_event(self, stat: AgentStat) -> None:
        record = asdict(stat)
        record["run_id"] = self.run_id
        record["ts"] = datetime.now(timezone.utc).isoformat()
        record["total_tokens"] = stat.total_tokens
        record["over_budget"] = stat.over_budget
        record["elapsed"] = round(stat.elapsed, 2)
        record["output"] = (stat.output or "")[:4000]
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self, reg: Registry) -> None:
        snap = reg.snapshot()
        total = sum(s.cost for s in snap)
        lines = [f"RUN {self.run_id}", "=" * 64]
        for s in snap:
            flag = "  [OVER BUDGET]" if s.over_budget else ""
            lines.append(
                f"{s.name:<12} {s.status:<7} model={s.model} "
                f"in={s.input_tokens} out={s.output_tokens} total={s.total_tokens} "
                f"budget={s.budget} cost=${s.cost:.5f} t={s.elapsed:.1f}s{flag}"
            )
        lines += ["-" * 64, f"TOTAL COST: ${total:.5f}"]
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
