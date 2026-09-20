"""Entry point: fetch GDELT DOC 2.0 timeline volume for every (trend, tier)
pair plus a brands-only baseline per tier, sequentially, through the cached
+ rate-limited client. Run as:

    python -m src.ingest_gdelt

Resumable by construction: the raw cache is durable, so a rerun skips any
(trend, tier) already fetched and only attempts what's missing. If GDELT
starts throttling hard, this fails fast per-request (src.config.MAX_RETRIES)
and stops the whole run after MAX_CONSECUTIVE_TASK_FAILURES in a row, rather
than grinding through the remaining tasks for nothing -- rerun later to pick
up where it left off.
"""
import logging
import sys
from typing import List, Optional, Tuple

from src.config import BASELINE_TREND_ID, MAX_CONSECUTIVE_TASK_FAILURES, TIERS, TIMESPAN
from src.config_models import BrandsConfig, ConfigError, TrendsConfig
from src.gdelt_client import GDELTClient
from src.gdelt_models import GDELTResponseError, GDELTRateLimitError
from src.query_builder import build_baseline_query, build_trend_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_gdelt")


def build_tasks(brands: BrandsConfig, trends: TrendsConfig) -> List[Tuple[str, str, str]]:
    """Return a list of (trend_id, tier, query) tasks: baselines first (3),
    then every trend x tier combination (in trends.yaml / TIERS order)."""
    tasks: List[Tuple[str, str, str]] = []
    for tier in TIERS:
        tasks.append((BASELINE_TREND_ID, tier, build_baseline_query(brands.tiers[tier])))
    for trend in trends.trends:
        for tier in TIERS:
            query = build_trend_query(trend, brands.tiers[tier])
            tasks.append((trend.id, tier, query))
    return tasks


def run(limit: Optional[int] = None) -> None:
    try:
        brands = BrandsConfig.from_yaml()
        trends = TrendsConfig.from_yaml()
    except ConfigError as exc:
        log.error("Config validation failed: %s", exc)
        raise SystemExit(1) from exc

    tasks = build_tasks(brands, trends)
    if limit is not None:
        tasks = tasks[:limit]

    client = GDELTClient()
    already_cached = sum(1 for (t, tier, q) in tasks if client.is_cached(t, tier, q))
    log.info(
        "Running %d ingest tasks (timespan=%s): %d already cached, %d to fetch",
        len(tasks),
        TIMESPAN,
        already_cached,
        len(tasks) - already_cached,
    )

    fetched, skipped, failed = [], [], []
    consecutive_failures = 0
    stopped_early = False

    for i, (trend_id, tier, query) in enumerate(tasks, start=1):
        was_cached = client.is_cached(trend_id, tier, query)
        try:
            cached = client.fetch(trend_id, tier, query)
        except GDELTResponseError:
            # Not transient -- a real query/schema bug. Fail loud immediately.
            log.error("Non-retryable failure on trend=%s tier=%s query=%r", trend_id, tier, query)
            raise
        except GDELTRateLimitError as exc:
            consecutive_failures += 1
            failed.append((trend_id, tier))
            log.error(
                "[%d/%d] FAILED trend=%s tier=%s (consecutive failures: %d/%d): %s",
                i, len(tasks), trend_id, tier, consecutive_failures, MAX_CONSECUTIVE_TASK_FAILURES, exc,
            )
            if consecutive_failures >= MAX_CONSECUTIVE_TASK_FAILURES:
                log.error(
                    "Stopping after %d consecutive failures -- rerun later to resume "
                    "(already-cached tasks are skipped automatically).",
                    consecutive_failures,
                )
                stopped_early = True
                break
            continue

        consecutive_failures = 0
        n_points = sum(len(s.data) for s in cached.response.timeline)
        if was_cached:
            skipped.append((trend_id, tier))
            log.info("[%d/%d] cached trend=%s tier=%s (%d points)", i, len(tasks), trend_id, tier, n_points)
        else:
            fetched.append((trend_id, tier))
            log.info("[%d/%d] fetched trend=%s tier=%s (%d points)", i, len(tasks), trend_id, tier, n_points)

    log.info(
        "Ingest %s: %d already cached, %d newly fetched, %d failed (of %d total).",
        "stopped early" if stopped_early else "complete",
        len(skipped),
        len(fetched),
        len(failed),
        len(tasks),
    )
    if failed:
        log.info("Missing (trend_id, tier): %s", failed)


if __name__ == "__main__":
    limit_arg = None
    if len(sys.argv) > 1:
        limit_arg = int(sys.argv[1])
    run(limit=limit_arg)
