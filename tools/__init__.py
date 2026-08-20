"""Shared plumbing for Scout's tool nodes: one Tavily client, one retry
policy, one way to turn a URL into a source domain.
"""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

_client: TavilyClient | None = None


def tavily_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def domain_of(url: str) -> str:
    host = urlparse(url).netloc
    return host[4:] if host.startswith("www.") else host


# 3 attempts, exponential backoff starting at 1s capped at 8s. A tool call
# that's still failing after that gets reraised — the graph node decides
# what to do with a dead tool, not this layer.
with_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
