"""Tavily web + news search tool node."""

from schema import Finding
from tools import domain_of, tavily_client, with_retry


@with_retry
def _search(query: str, max_results: int, topic: str, search_depth: str) -> dict:
    return tavily_client().search(
        query=query,
        max_results=max_results,
        topic=topic,
        search_depth=search_depth,
    )


def _to_findings(response: dict, sub_question: str) -> list[Finding]:
    findings = []
    for result in response.get("results", []):
        url = result.get("url")
        if not url:
            continue
        findings.append(
            Finding(
                claim=result.get("content") or result.get("title", ""),
                source_url=url,
                source_domain=domain_of(url),
                snippet=result.get("content"),
                sub_question=sub_question,
            )
        )
    return findings


def web_search(query: str, max_results: int = 5) -> list[Finding]:
    """General web search — basic depth, 1 Tavily credit."""
    response = _search(query, max_results, topic="general", search_depth="basic")
    return _to_findings(response, query)


def news_search(query: str, max_results: int = 5) -> list[Finding]:
    """News-scoped search — advanced depth, 2 Tavily credits."""
    response = _search(query, max_results, topic="news", search_depth="advanced")
    return _to_findings(response, query)
