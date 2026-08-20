"""Pydantic models for Scout's structured output."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl


class Finding(BaseModel):
    """A single raw, source-tagged claim pulled from a tool call."""

    claim: str
    source_url: HttpUrl
    source_domain: str
    snippet: str | None = None
    sub_question: str | None = None


class ClaimCluster(BaseModel):
    """One atomic claim plus every source URL that backs it.

    Raw Findings hold whatever a search snippet actually said, so two
    sources agreeing on the same fact almost never share exact wording.
    An LLM pass clusters raw findings into these before verify() can do
    its (deterministic) domain-counting.
    """

    claim: str
    source_urls: list[HttpUrl]


class ClaimClusters(BaseModel):
    clusters: list[ClaimCluster]


class ReportNarrative(BaseModel):
    summary: str
    red_flags: list[str]


class Fact(BaseModel):
    claim: str
    sources: list[HttpUrl]
    confidence: Literal["verified", "flagged"]


class DueDiligenceReport(BaseModel):
    company: str
    summary: str
    key_facts: list[Fact]
    red_flags: list[str]
    sources: list[HttpUrl]
    generated_at: datetime
