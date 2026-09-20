"""Low-level GDELT DOC 2.0 API client: rate limiting, retry/backoff, and the
raw-JSON cache. This is the only module that makes network calls.

Two lessons from hitting the live API directly (not just the docs):
  - It enforces "one request every 5 seconds" -- sometimes returned as plain
    text with HTTP 429, sometimes with HTTP 200. So we can't trust status
    codes alone; every response body is parsed and schema-validated
    regardless of status code.
  - A malformed query (e.g. parentheses around a single, non-OR'd term)
    comes back as a plain-text error, not JSON. That's not a transient
    problem, so it must not be retried -- it should fail loudly immediately.
"""
import hashlib
import json
import random
import time
from pathlib import Path

import requests

from src.config import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_JITTER_SECONDS,
    BACKOFF_MAX_SECONDS,
    GDELT_DOC_URL,
    GDELT_FORMAT,
    GDELT_MODE,
    MAX_RETRIES,
    MIN_REQUEST_INTERVAL_SECONDS,
    RAW_DIR,
    TIMESPAN,
)

_RATE_LIMIT_MARKER = "please limit requests"
from src.gdelt_models import (
    CachedResponse,
    FetchMeta,
    GDELTResponseError,
    GDELTTimelineResponse,
    GDELTRateLimitError,
)


class GDELTClient:
    def __init__(self, cache_dir: Path = RAW_DIR):
        self.cache_dir = cache_dir
        self._last_request_time = 0.0

    def fetch(self, trend_id: str, tier: str, query: str, timespan: str = TIMESPAN) -> CachedResponse:
        """Return a validated CachedResponse for (trend_id, tier, query, timespan),
        serving from the on-disk cache when available and never hitting the
        network on a cache hit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_path(trend_id, tier, query, timespan)
        if cache_path.exists():
            return self._load_cache(cache_path)
        cached = self._fetch_with_retry(query, timespan)
        self._write_cache(cache_path, cached)
        return cached

    def is_cached(self, trend_id: str, tier: str, query: str, timespan: str = TIMESPAN) -> bool:
        return self.cache_path(trend_id, tier, query, timespan).exists()

    def cache_path(self, trend_id: str, tier: str, query: str, timespan: str = TIMESPAN) -> Path:
        digest = hashlib.sha256(f"{query}|{timespan}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{trend_id}__{tier}__{digest}.json"

    def _load_cache(self, path: Path) -> CachedResponse:
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise GDELTResponseError(f"Corrupted cache file {path}: {exc}") from exc
        try:
            return CachedResponse.model_validate(raw)
        except Exception as exc:
            raise GDELTResponseError(f"Cache file {path} failed schema validation: {exc}") from exc

    def _write_cache(self, path: Path, cached: CachedResponse) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(cached.model_dump_json(indent=2))
        tmp.replace(path)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _fetch_with_retry(self, query: str, timespan: str) -> CachedResponse:
        params = {"query": query, "mode": GDELT_MODE, "format": GDELT_FORMAT, "timespan": timespan}
        last_exc: Exception = GDELTRateLimitError("no attempts made")

        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = requests.get(GDELT_DOC_URL, params=params, timeout=30)
            except requests.exceptions.RequestException as exc:
                last_exc = GDELTRateLimitError(f"network error on attempt {attempt + 1}: {exc}")
                self._sleep_backoff(attempt)
                continue

            is_rate_limit_text = _RATE_LIMIT_MARKER in resp.text.lower()
            if resp.status_code == 429 or resp.status_code >= 500 or is_rate_limit_text:
                # Confirmed live: GDELT's rate-limit message ("please limit
                # requests to one every 5 seconds") can come back under HTTP
                # 200, not just 429/5xx -- content, not just status, decides
                # whether this is retryable.
                last_exc = GDELTRateLimitError(
                    f"HTTP {resp.status_code} on attempt {attempt + 1} for query={query!r}: "
                    f"{resp.text[:200]!r}"
                )
                self._sleep_backoff(attempt)
                continue

            # Non-retryable status/content. GDELT can still return a
            # plain-text query error (e.g. an illegal-character message)
            # under HTTP 200, so always attempt to parse + validate rather
            # than trusting the status code.
            try:
                payload = resp.json()
            except ValueError as exc:
                raise GDELTResponseError(
                    f"Non-JSON response (HTTP {resp.status_code}) for query={query!r}: "
                    f"{resp.text[:300]!r}"
                ) from exc
            try:
                parsed = GDELTTimelineResponse.model_validate(payload)
            except Exception as exc:
                raise GDELTResponseError(
                    f"Schema validation failed for query={query!r}: {exc}"
                ) from exc

            meta = FetchMeta.now(query=query, timespan=timespan, url=GDELT_DOC_URL)
            return CachedResponse(meta=meta, response=parsed)

        raise last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= MAX_RETRIES:
            return
        backoff = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS) + random.uniform(
            0, BACKOFF_JITTER_SECONDS
        )
        time.sleep(backoff)
