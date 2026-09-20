"""Pydantic models for the GDELT DOC 2.0 API (mode=timelinevolraw) and the
provenance-wrapped cache format we store it in.

Two distinct exception types, because they call for different handling:
- GDELTRateLimitError: transient (429/5xx) -> retry with backoff.
- GDELTResponseError: the body isn't valid JSON, or doesn't match the schema
  we expect (this is how GDELT's plain-text query errors show up, sometimes
  under HTTP 200) -> fail loudly, immediately, no retry.
"""
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field


class GDELTRateLimitError(Exception):
    """HTTP 429 or 5xx from the GDELT API. Caller should retry with backoff."""


class GDELTResponseError(Exception):
    """Response body is not valid JSON, or fails schema validation. Do not retry."""


class GDELTTimelinePoint(BaseModel):
    date: str
    value: float
    norm: float


class GDELTTimelineSeries(BaseModel):
    series: str
    data: List[GDELTTimelinePoint] = Field(default_factory=list)


class GDELTTimelineResponse(BaseModel):
    timeline: List[GDELTTimelineSeries] = Field(default_factory=list)


class FetchMeta(BaseModel):
    query: str
    timespan: str
    url: str
    fetched_at: datetime

    @staticmethod
    def now(query: str, timespan: str, url: str) -> "FetchMeta":
        return FetchMeta(
            query=query,
            timespan=timespan,
            url=url,
            fetched_at=datetime.now(timezone.utc),
        )


class CachedResponse(BaseModel):
    meta: FetchMeta
    response: GDELTTimelineResponse
