"""Shared configuration: default model + paths.

Model strings are LiteLLM-style ("provider/model"). Switch provider by changing
MODEL here, per-agent in each agent file, or with `--model` on the command line.

Examples:
    anthropic/claude-opus-5-5
    openai/gpt-4o-mini
    gemini/gemini-1.5-pro
    mistral/mistral-large-latest
    ollama/llama3            (local; no API key needed)

Cost is computed by LiteLLM per provider — no manual pricing table needed.
"""

import os

MODEL = "anthropic/claude-opus-5-5"  # default model for any agent that doesn't override it

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# logs/    = run diagnostics (tokens, cost, status) — machine + summary
# outputs/ = the actual text each agent produced (the agentic AI output)
LOG_DIR = os.path.join(_ROOT, "logs")
OUTPUT_DIR = os.path.join(_ROOT, "outputs")
