import math
from pathlib import Path

import pandas as pd
import pytest

from src import transform
from src.gdelt_models import (
    CachedResponse,
    FetchMeta,
    GDELTTimelinePoint,
    GDELTTimelineResponse,
    GDELTTimelineSeries,
)

# 2026-01-05 is a Monday; each *_MON constant is the Monday of its ISO week.
WEEK1_MON = "20260105"
WEEK2_MON = "20260112"
WEEK3_MON = "20260119"


def _write_raw(raw_dir: Path, trend_id: str, tier: str, points, fetched_at="2026-01-01T00:00:00+00:00", query="q"):
    series = GDELTTimelineSeries(
        series=query, data=[GDELTTimelinePoint(date=d, value=v, norm=n) for d, v, n in points]
    )
    meta = FetchMeta(query=query, timespan="12m", url="https://api.gdeltproject.org/api/v2/doc/doc", fetched_at=fetched_at)
    cached = CachedResponse(meta=meta, response=GDELTTimelineResponse(timeline=[series]))
    path = raw_dir / f"{trend_id}__{tier}__{abs(hash((trend_id, tier, fetched_at))) % 10**8}.json"
    path.write_text(cached.model_dump_json())
    return path


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_raw_to_rows_types_and_provenance(raw_dir):
    _write_raw(
        raw_dir,
        "_baseline",
        "luxury",
        [(f"{WEEK1_MON}T000000Z", 100.0, 5000.0)],
        fetched_at="2026-02-01T12:00:00+00:00",
    )
    rows = transform.raw_to_rows(raw_dir)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert pd.api.types.is_datetime64_any_dtype(rows["date"])
    assert row["date"] == pd.Timestamp("2026-01-05")
    assert row["value"] == 100.0
    assert row["norm"] == 5000.0
    assert row["ingested_at"] == pd.Timestamp("2026-02-01T12:00:00+00:00")


def test_weekly_aggregation_and_share_math(tmp_path, raw_dir):
    _write_raw(raw_dir, "_baseline", "luxury", [(f"{WEEK1_MON}T000000Z", 100.0, 1.0), (f"{WEEK2_MON}T000000Z", 200.0, 1.0)])
    _write_raw(
        raw_dir,
        "suede",
        "luxury",
        [(f"{WEEK1_MON}T000000Z", 10.0, 1.0), (f"{WEEK1_MON}T120000Z", 5.0, 1.0), (f"{WEEK2_MON}T000000Z", 20.0, 1.0)],
    )

    out_paths = {
        "staging_mentions_path": tmp_path / "sm.parquet",
        "staging_baseline_path": tmp_path / "sb.parquet",
        "weekly_baseline_path": tmp_path / "wb.parquet",
        "weekly_mentions_path": tmp_path / "wm.parquet",
    }
    transform.run(raw_dir=raw_dir, **out_paths)

    wm = pd.read_parquet(out_paths["weekly_mentions_path"])
    wm = wm.sort_values("week_start").reset_index(drop=True)
    assert len(wm) == 2
    assert wm.loc[0, "mention_count"] == 15.0  # two same-week points summed
    assert wm.loc[0, "baseline_mention_count"] == 100.0
    assert wm.loc[0, "trend_share"] == pytest.approx(0.15)
    assert wm.loc[1, "mention_count"] == 20.0
    assert wm.loc[1, "trend_share"] == pytest.approx(0.1)


def test_zero_baseline_week_produces_nan_share_not_inf(tmp_path, raw_dir):
    _write_raw(raw_dir, "_baseline", "luxury", [(f"{WEEK1_MON}T000000Z", 0.0, 1.0)])
    _write_raw(raw_dir, "suede", "luxury", [(f"{WEEK1_MON}T000000Z", 5.0, 1.0)])

    out_paths = {
        "staging_mentions_path": tmp_path / "sm.parquet",
        "staging_baseline_path": tmp_path / "sb.parquet",
        "weekly_baseline_path": tmp_path / "wb.parquet",
        "weekly_mentions_path": tmp_path / "wm.parquet",
    }
    transform.run(raw_dir=raw_dir, **out_paths)

    wm = pd.read_parquet(out_paths["weekly_mentions_path"])
    assert len(wm) == 1
    assert math.isnan(wm.loc[0, "trend_share"])
    assert not math.isinf(wm.loc[0, "trend_share"])


def test_gap_weeks_stay_absent_not_zero_filled(tmp_path, raw_dir):
    _write_raw(raw_dir, "_baseline", "luxury", [(f"{WEEK1_MON}T000000Z", 100.0, 1.0), (f"{WEEK3_MON}T000000Z", 100.0, 1.0)])
    _write_raw(
        raw_dir,
        "suede",
        "luxury",
        [(f"{WEEK1_MON}T000000Z", 10.0, 1.0), (f"{WEEK3_MON}T000000Z", 10.0, 1.0)],
    )  # no week2 data at all -- a true gap, not a zero

    out_paths = {
        "staging_mentions_path": tmp_path / "sm.parquet",
        "staging_baseline_path": tmp_path / "sb.parquet",
        "weekly_baseline_path": tmp_path / "wb.parquet",
        "weekly_mentions_path": tmp_path / "wm.parquet",
    }
    transform.run(raw_dir=raw_dir, **out_paths)

    wm = pd.read_parquet(out_paths["weekly_mentions_path"])
    assert len(wm) == 2  # not 3 -- week2 never appears as an explicit zero row
    assert set(wm["week_start"].dt.strftime("%Y%m%d")) == {WEEK1_MON, WEEK3_MON}


def test_transform_is_idempotent(tmp_path, raw_dir):
    _write_raw(raw_dir, "_baseline", "luxury", [(f"{WEEK1_MON}T000000Z", 100.0, 1.0)])
    _write_raw(raw_dir, "suede", "luxury", [(f"{WEEK1_MON}T000000Z", 10.0, 1.0)])

    out_paths = {
        "staging_mentions_path": tmp_path / "sm.parquet",
        "staging_baseline_path": tmp_path / "sb.parquet",
        "weekly_baseline_path": tmp_path / "wb.parquet",
        "weekly_mentions_path": tmp_path / "wm.parquet",
    }
    transform.run(raw_dir=raw_dir, **out_paths)
    first = pd.read_parquet(out_paths["weekly_mentions_path"])

    transform.run(raw_dir=raw_dir, **out_paths)
    second = pd.read_parquet(out_paths["weekly_mentions_path"])

    pd.testing.assert_frame_equal(first, second)
