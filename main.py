"""
Runner for the multi-agent system.

Run all workers (head monitors automatically):
    python main.py "renewable energy storage"

Run a SINGLE worker (head still monitors it):
    python main.py "renewable energy storage" --agent classifier

Run a worker that has a dependency — the dependency runs first automatically:
    python main.py "renewable energy storage" --agent writer   # also runs researcher

List available agents:
    python main.py --list
"""

import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

from core.config import LOG_DIR, OUTPUT_DIR
from core.registry import Registry
from core.logger import RunLogger
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.classifier import ClassifierAgent
from agents.head import HeadAgent

# Registry of worker agents (head is handled separately — it always runs).
WORKER_CLASSES = {
    ResearcherAgent.name: ResearcherAgent,
    WriterAgent.name: WriterAgent,
    ClassifierAgent.name: ClassifierAgent,
}


def resolve_dependencies(selected: list[str]) -> set[str]:
    """Expand the selection to include every agent it transitively depends on."""
    needed: set[str] = set()

    def add(name: str) -> None:
        if name in needed:
            return
        if name not in WORKER_CLASSES:
            sys.exit(f"Unknown agent: {name}. Available: {', '.join(WORKER_CLASSES)}")
        needed.add(name)
        for dep in WORKER_CLASSES[name].depends_on:
            add(dep)

    for s in selected:
        add(s)
    return needed


def run_workers(instances: dict, needed: set[str], topic: str) -> dict[str, str]:
    """Run agents side-by-side, level by level, respecting dependencies."""
    context: dict[str, str] = {}
    remaining = set(needed)
    while remaining:
        ready = [n for n in remaining
                 if all(dep in context for dep in WORKER_CLASSES[n].depends_on)]
        with ThreadPoolExecutor(max_workers=len(ready)) as pool:
            futures = {pool.submit(instances[n].run, topic, context): n for n in ready}
            for fut in futures:
                name = futures[fut]
                context[name] = fut.result()
        remaining -= set(ready)
    return context


def main() -> None:
    # Windows consoles default to cp1252, which can't encode the dashboard's
    # box-drawing/▶✓ characters — force UTF-8 so printing never crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description="Multi-agent system with head monitoring.")
    parser.add_argument("topic", nargs="?", default="renewable energy storage",
                        help="Topic for the researcher/writer.")
    parser.add_argument("--agent", action="append", metavar="NAME",
                        help="Run only this agent (repeatable). Default: all workers.")
    parser.add_argument("--model", metavar="LITELLM_MODEL",
                        help="Override the model for ALL agents, e.g. openai/gpt-4o-mini.")
    parser.add_argument("--list", action="store_true", help="List available agents and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available worker agents:")
        for name, cls in WORKER_CLASSES.items():
            dep = f" (depends on: {', '.join(cls.depends_on)})" if cls.depends_on else ""
            print(f"  - {name}{dep}")
        print("  - head  (always runs: monitors + assesses)")
        return

    selected = args.agent if args.agent else list(WORKER_CLASSES)
    needed = resolve_dependencies(selected)

    reg = Registry()
    logger = RunLogger(LOG_DIR, OUTPUT_DIR)

    # Register selected workers...
    instances = {}
    for name in needed:
        cls = WORKER_CLASSES[name]
        agent = cls(reg, logger)
        if args.model:
            agent.model = args.model     # CLI override wins over the per-agent file
        instances[name] = agent
        reg.register(name, agent.model, budget=cls.budget)

    # ...and ALWAYS register + start the head agent for monitoring.
    head = HeadAgent(reg, logger)
    if args.model:
        head.model = args.model
    reg.register(head.name, head.model, budget=head.budget)

    done = threading.Event()
    monitor = threading.Thread(target=head.monitor, args=(done,), daemon=True)
    monitor.start()

    try:
        context = run_workers(instances, needed, args.topic)
        head.assess()   # head reviews the token economy of whatever ran
    finally:
        done.set()
        monitor.join()

    # Detailed printout below the final dashboard frame.
    print("\n" + "═" * 78 + "\n WORKER OUTPUTS\n" + "═" * 78)
    for name in needed:
        s = reg.stats[name]
        print(f"\n[{name}] ({s.status})\n{s.output or s.error}")

    print("\n" + "═" * 78 + "\n HEAD AGENT ASSESSMENT\n" + "═" * 78)
    print(reg.stats["head"].output or reg.stats["head"].error)

    total = sum(s.cost for s in reg.snapshot())
    print(f"\nTotal run cost: ${total:.5f}")

    logger.write_summary(reg)
    logger.write_combined_output(reg)
    print(f"\nLogs written:\n  {logger.jsonl_path}\n  {logger.txt_path}")
    print(f"Agent outputs saved under:\n  {logger.output_root}  (one folder per agent)")


if __name__ == "__main__":
    main()
