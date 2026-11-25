"""Which chat model the graph's cluster/synthesize/critique nodes use.

Picked by LLM_PROVIDER (see .env.example):
  - "anthropic" (default): Claude. The API key is always passed in
    explicitly. On the live demo each visitor pastes their own key into
    the UI; running locally, it can come from a .env file.
  - "ollama": a local model served by Ollama (https://ollama.com), so no
    Anthropic key is needed at all.
"""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class MissingApiKey(ValueError):
    pass


def model_name() -> str:
    return OLLAMA_MODEL if PROVIDER == "ollama" else ANTHROPIC_MODEL


def local_anthropic_key() -> str | None:
    """An ANTHROPIC_API_KEY from a local .env file, if there is one.

    Read straight from the file rather than os.environ on purpose: hosting
    platforms (Render included) inject their configured secrets as
    environment variables, and the live demo must never silently fall back
    to the owner's key. Visitors bring their own. .env is gitignored, so it
    only exists on a machine someone is running Scout on themselves."""
    return dotenv_values(Path(__file__).parent / ".env").get("ANTHROPIC_API_KEY") or None


def needs_user_key() -> bool:
    return PROVIDER != "ollama" and not local_anthropic_key()


def make_llm(api_key: str | None = None) -> BaseChatModel:
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx raised from Ollama's small default: the cluster step
        # feeds the model ~25 source-tagged findings in one prompt.
        return ChatOllama(
            model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0, num_ctx=16384
        )

    from langchain_anthropic import ChatAnthropic

    api_key = api_key or local_anthropic_key()
    if not api_key:
        raise MissingApiKey(
            "An Anthropic API key is required (or set LLM_PROVIDER=ollama to run locally)."
        )
    return ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=16000, api_key=api_key)


def check_anthropic_key(api_key: str) -> None:
    """Fail fast on a bad key with one cheap, free call (listing models),
    instead of letting a research job run its web searches first and then
    die minutes later at the first LLM step. Raises anthropic.AuthenticationError
    for a rejected key."""
    from anthropic import Anthropic

    Anthropic(api_key=api_key, max_retries=1, timeout=20).models.list(limit=1)
