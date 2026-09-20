"""raw (data/raw/*.json) -> staging (typed parquet) -> mart (weekly aggregates).

Every run fully recomputes staging + both marts from whatever is currently in
data/raw/, and writes atomically (temp file + rename). Nothing is ever
appended to an existing output, so reruns can't create duplicates -- that
guarantee holds by construction, not by a dedup step. This also means every
column, including `ingested_at`, must come from the raw layer's own
provenance metadata rather than wall-clock time at transform time, or two
runs over identical inputs would produce different output and break the
idempotency guarantee.
"""
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_TREND_ID,
    RAW_DIR,
    STAGING_BASELINE_PATH,
    STAGING_MENTIONS_PATH,
    WEEKLY_BASELINE_PATH,
    WEEKLY_MENTIONS_PATH,
)
from src.gdelt_models import CachedResponse
from src.gdelt_models import GDELTResponseError

log = logging.getLogger("transform")


def _load_raw_files(raw_dir: Path = RAW_DIR) -> List[Path]:
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*.json"))


def _parse_cache_filename(path: Path) -> tuple:
    """{trend_id}__{tier}__{hash}.json -> (trend_id, tier)"""
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 3:
        raise GDELTResponseError(f"Unexpected raw cache filename shape: {path.name}")
    trend_id, tier, _hash = parts
    return trend_id, tier


def raw_to_rows(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every raw cache file into one long DataFrame:
    trend_id, tier, date (datetime64), value, norm, ingested_at (datetime64)."""
    records = []
    for path in _load_raw_files(raw_dir):
        trend_id, tier = _parse_cache_filename(path)
        try:
            cached = CachedResponse.model_validate_json(path.read_text())
        except Exception as exc:
            raise GDELTResponseError(f"Raw cache file {path} failed schema validation: {exc}") from exc

        for series in cached.response.timeline:
            for point in series.data:
                records.append(
                    {
                        "trend_id": trend_id,
                        "tier": tier,
                        "date": pd.to_datetime(point.date, format="%Y%m%dT%H%M%SZ"),
                        "value": point.value,
                        "norm": point.norm,
                        "ingested_at": cached.meta.fetched_at,
                    }
                )

    columns = ["trend_id", "tier", "date", "value", "norm", "ingested_at"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns)


def split_staging(rows: pd.DataFrame) -> tuple:
    """Split the combined raw rows into (mentions, baseline) staging frames."""
    is_baseline = rows["trend_id"] == BASELINE_TREND_ID
    baseline = rows.loc[is_baseline, ["tier", "date", "value", "norm", "ingested_at"]].reset_index(drop=True)
    mentions = rows.loc[~is_baseline].reset_index(drop=True)
    return mentions, baseline


def _week_start(dates: pd.Series) -> pd.Series:
    # Monday-start ISO week.
    return dates.dt.to_period("W-SUN").apply(lambda p: p.start_time.normalize())


def weekly_baseline(baseline_staging: pd.DataFrame) -> pd.DataFrame:
    if baseline_staging.empty:
        return pd.DataFrame(columns=["tier", "week_start", "baseline_mention_count"])
    df = baseline_staging.copy()
    df["week_start"] = _week_start(df["date"])
    out = (
        df.groupby(["tier", "week_start"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "baseline_mention_count"})
    )
    return out.sort_values(["tier", "week_start"]).reset_index(drop=True)


def weekly_mentions(mentions_staging: pd.DataFrame, weekly_baseline_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_staging.empty:
        return pd.DataFrame(
            columns=["trend_id", "tier", "week_start", "mention_count", "baseline_mention_count", "trend_share"]
        )
    df = mentions_staging.copy()
    df["week_start"] = _week_start(df["date"])
    weekly = (
        df.groupby(["trend_id", "tier", "week_start"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "mention_count"})
    )
    merged = weekly.merge(weekly_baseline_df, on=["tier", "week_start"], how="left")
    # A missing or zero baseline for a week yields an undefined (NaN) share,
    # never inf/crash from a divide-by-zero. np.nan (not pd.NA) keeps this a
    # plain float64 column so downstream rolling/median math works normally.
    safe_baseline = merged["baseline_mention_count"].astype(float).replace(0.0, np.nan)
    merged["trend_share"] = merged["mention_count"] / safe_baseline
    return merged.sort_values(["trend_id", "tier", "week_start"]).reset_index(drop=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run(
    raw_dir: Path = RAW_DIR,
    staging_mentions_path: Path = STAGING_MENTIONS_PATH,
    staging_baseline_path: Path = STAGING_BASELINE_PATH,
    weekly_baseline_path: Path = WEEKLY_BASELINE_PATH,
    weekly_mentions_path: Path = WEEKLY_MENTIONS_PATH,
) -> None:
    rows = raw_to_rows(raw_dir)
    mentions_staging, baseline_staging = split_staging(rows)

    _atomic_write_parquet(mentions_staging, staging_mentions_path)
    _atomic_write_parquet(baseline_staging, staging_baseline_path)

    weekly_baseline_df = weekly_baseline(baseline_staging)
    weekly_mentions_df = weekly_mentions(mentions_staging, weekly_baseline_df)

    _atomic_write_parquet(weekly_baseline_df, weekly_baseline_path)
    _atomic_write_parquet(weekly_mentions_df, weekly_mentions_path)

    log.info(
        "transform complete: %d staging mention rows, %d staging baseline rows, "
        "%d weekly_baseline rows, %d weekly_mentions rows",
        len(mentions_staging),
        len(baseline_staging),
        len(weekly_baseline_df),
        len(weekly_mentions_df),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
