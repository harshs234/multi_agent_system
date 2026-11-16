"""Provider-neutral LLM layer (LiteLLM).

One function, any provider (LiteLLM supports 100+). Model strings are LiteLLM-style:
    anthropic/claude-opus-5-5
    openai/gpt-4o-mini               (or just: gpt-4o-mini)
    gemini/gemini-1.5-flash
    groq/llama-3.3-70b-versatile
    mistral/mistral-large-latest
    ollama/llama3                    (local, no API key)

Each provider reads its own key from the environment:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, ...
Full provider list: https://docs.litellm.ai/docs/providers
"""

from dataclasses import dataclass

import litellm

# Drop params a given model doesn't support (e.g. reasoning_effort on non-reasoning
# models) instead of erroring — keeps the same code working across every provider.
litellm.drop_params = True
litellm.suppress_debug_info = True


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    cost: float


def complete(model: str, system: str, user: str, max_tokens: int,
             effort: str | None = None) -> LLMResult:
    kwargs = {}
    if effort:
        kwargs["reasoning_effort"] = effort  # mapped per-provider; dropped if unsupported

    resp = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )

    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    try:
        cost = litellm.completion_cost(completion_response=resp) or 0.0
    except Exception:  # noqa: BLE001  (pricing not known for some local/custom models)
        cost = 0.0

    return LLMResult(text, inp, out, cache_read, cache_write, cost)
