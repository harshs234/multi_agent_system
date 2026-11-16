# Multi-Agent System (provider-neutral via LiteLLM)

Three worker agents run **side-by-side**; a **head agent** always runs alongside to
monitor their status and **token economy** (tokens → USD), live.

```
researcher ─► writer        researcher + classifier run in parallel;
classifier (independent)    writer starts when researcher finishes;
head        (always on)     head monitors everything + gives a final assessment.
```

## Structure

```
multi_agent_system/
├── main.py              # runner / CLI
├── requirements.txt
├── core/
│   ├── config.py        # default model + paths
│   ├── llm.py           # LiteLLM wrapper (any provider) + cost
│   ├── registry.py      # shared thread-safe state + token ledger
│   └── logger.py        # per-run logs
├── agents/
│   ├── base.py          # shared run logic
│   ├── researcher.py    # independent        — owns its token config
│   ├── writer.py        # depends on researcher
│   ├── classifier.py    # independent, easy task
│   └── head.py          # monitor + supervisor
└── logs/                # created on first run
```

Each agent file sets its **own token economy** at the top of the class:
`model`, `max_tokens` (hard output cap), `effort`, `budget` (soft, flagged if exceeded).

## 1. Install

```bash
cd multi_agent_system
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Set a provider key

LiteLLM reads the key for whichever provider your model uses:

| Provider   | Example model string              | Env var             |
|------------|-----------------------------------|---------------------|
| Anthropic  | `anthropic/claude-opus-5-5`       | `ANTHROPIC_API_KEY` |
| OpenAI     | `openai/gpt-4o-mini`              | `OPENAI_API_KEY`    |
| Google     | `gemini/gemini-1.5-flash`         | `GEMINI_API_KEY`    |
| Groq       | `groq/llama-3.3-70b-versatile`    | `GROQ_API_KEY`      |
| Mistral    | `mistral/mistral-large-latest`    | `MISTRAL_API_KEY`   |
| Cohere     | `cohere/command-r-plus`           | `COHERE_API_KEY`    |
| OpenRouter | `openrouter/meta-llama/llama-3.1-8b-instruct` | `OPENROUTER_API_KEY` |
| Ollama     | `ollama/llama3` (local)           | *(none)*            |

LiteLLM supports 100+ providers — any of them works the same way (model string + its
key). Full list: https://docs.litellm.ai/docs/providers

```powershell
# PowerShell (current session)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# or
$env:OPENAI_API_KEY = "sk-..."
```

## 3. Run

```bash
# All workers, head monitors automatically
python main.py "renewable energy storage"

# A SINGLE worker — head still runs and monitors it
python main.py "renewable energy storage" --agent classifier

# A dependent worker — its dependency runs first automatically
python main.py "renewable energy storage" --agent writer

# Use a different provider/model for ALL agents
python main.py "quantum computing" --model openai/gpt-4o-mini

# List agents
python main.py --list
```

## Output

- **Live dashboard** (head agent) — status, `tokens/budget`, cost, elapsed, per agent + total.
- **Worker outputs** + **head assessment** printed at the end.
- **Logs** in `logs/`:
  - `run-<timestamp>.jsonl` — one record per agent call (machine-readable)
  - `run-<timestamp>.txt` — human-readable summary

## Mixing providers per agent

Give each agent a different provider by editing its file — e.g. a cheap model for the
easy task and a strong one for writing:

```python
# agents/classifier.py
model = "openai/gpt-4o-mini"
# agents/writer.py
model = "anthropic/claude-opus-5-5"
```

The head's dashboard will show the cost difference between them.
