"""weekly_mentions.parquet + weekly_baseline.parquet -> trend_diffusion.csv.

Per (trend, tier): detect a "sustained" breakout in trend_share (mentions
normalized by that tier's own baseline press volume, so tiers with
structurally more press don't look like first movers by construction), find
the peak week, compute lags between tiers, and classify the trend by how far
it has diffused.

Sustained-signal definition (replaces a fixed volume threshold):
  - reindex the trend's weekly share onto a dense weekly calendar spanning
    the tier's full observed window, filling *true gaps* (weeks the trend
    had zero matches) with share=0. Weeks where the tier's own baseline was
    zero stay NaN (undefined share) rather than being coerced to 0 -- they
    are a different case from "no coverage that week".
  - a week is "elevated" if its SIGNAL_ROLLING_WINDOW_WEEKS rolling mean
    share exceeds the trailing TRAILING_BASELINE_WINDOW_WEEKS-week median
    share by SIGNAL_FACTOR.
  - first_signal_week is the earliest week starting a run of SUSTAINED_WEEKS
    consecutive elevated weeks -- one stray elevated week alone can never
    set it.
  - peak_week is the max-share week among all elevated weeks (ties broken by
    the earliest such week).
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    SIGNAL_FACTOR,
    SIGNAL_ROLLING_WINDOW_WEEKS,
    SUSTAINED_WEEKS,
    TIERS,
    TRAILING_BASELINE_WINDOW_WEEKS,
    TREND_DIFFUSION_CSV,
    WEEKLY_BASELINE_PATH,
    WEEKLY_MENTIONS_PATH,
)
from src.config_models import TrendsConfig

log = logging.getLogger("analyze")


@dataclass
class TierSignal:
    tier: str
    first_signal_week: Optional[pd.Timestamp]
    peak_week: Optional[pd.Timestamp]


def tier_calendar(weekly_baseline_df: pd.DataFrame, tier: str) -> pd.DatetimeIndex:
    weeks = weekly_baseline_df.loc[weekly_baseline_df["tier"] == tier, "week_start"]
    if weeks.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(weeks.min(), weeks.max(), freq="7D")


def share_series(
    weekly_mentions_df: pd.DataFrame, trend_id: str, tier: str, calendar: pd.DatetimeIndex
) -> pd.Series:
    """Weekly trend_share for (trend_id, tier), reindexed onto `calendar`.
    Weeks absent from the mart (true gaps -- zero trend mentions that week)
    are filled with 0.0. Weeks present in the mart with a NaN share (the
    tier's baseline itself was 0 that week) are left as NaN."""
    sub = weekly_mentions_df[
        (weekly_mentions_df["trend_id"] == trend_id) & (weekly_mentions_df["tier"] == tier)
    ]
    indexed = sub.set_index("week_start")["trend_share"]
    present = calendar.isin(indexed.index)
    reindexed = indexed.reindex(calendar)
    reindexed.loc[~present] = reindexed.loc[~present].fillna(0.0)
    return reindexed


def elevated_weeks(s: pd.Series) -> pd.Series:
    rolling_mean = s.rolling(window=SIGNAL_ROLLING_WINDOW_WEEKS, min_periods=SIGNAL_ROLLING_WINDOW_WEEKS).mean()
    trailing_median = (
        s.rolling(window=TRAILING_BASELINE_WINDOW_WEEKS, min_periods=TRAILING_BASELINE_WINDOW_WEEKS)
        .median()
        .shift(1)
    )
    # Comparisons against NaN evaluate to False in pandas, so weeks without
    # enough rolling/trailing history are correctly never "elevated".
    return rolling_mean > (trailing_median * SIGNAL_FACTOR)


def first_sustained_week(elevated: pd.Series) -> Optional[pd.Timestamp]:
    n = SUSTAINED_WEEKS
    if len(elevated) < n:
        return None
    for i in range(len(elevated) - n + 1):
        if elevated.iloc[i : i + n].all():
            return elevated.index[i]
    return None


def peak_week_among_elevated(s: pd.Series, elevated: pd.Series) -> Optional[pd.Timestamp]:
    candidates = s[elevated.fillna(False)].dropna()
    if candidates.empty:
        return None
    max_val = candidates.max()
    tied = candidates[candidates == max_val]
    return tied.index.min()


def _week_delta(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[int]:
    if start is None or end is None:
        return None
    return int((end - start).days // 7)


def classify(tier_signals: dict) -> str:
    if tier_signals["affordable"].first_signal_week is not None:
        return "mass"
    if tier_signals["semi_luxury"].first_signal_week is not None:
        return "spreading"
    if tier_signals["luxury"].first_signal_week is not None:
        return "emerging"
    return "no_signal"


def analyze_trend(trend_id: str, weekly_mentions_df: pd.DataFrame, weekly_baseline_df: pd.DataFrame) -> dict:
    tier_signals = {}
    for tier in TIERS:
        calendar = tier_calendar(weekly_baseline_df, tier)
        s = share_series(weekly_mentions_df, trend_id, tier, calendar)
        elevated = elevated_weeks(s)
        first = first_sustained_week(elevated)
        peak = peak_week_among_elevated(s, elevated)
        tier_signals[tier] = TierSignal(tier=tier, first_signal_week=first, peak_week=peak)

    return {
        "trend_id": trend_id,
        "luxury_first_signal_week": tier_signals["luxury"].first_signal_week,
        "luxury_peak_week": tier_signals["luxury"].peak_week,
        "semi_luxury_first_signal_week": tier_signals["semi_luxury"].first_signal_week,
        "semi_luxury_peak_week": tier_signals["semi_luxury"].peak_week,
        "affordable_first_signal_week": tier_signals["affordable"].first_signal_week,
        "affordable_peak_week": tier_signals["affordable"].peak_week,
        "lag_luxury_to_semi_weeks": _week_delta(
            tier_signals["luxury"].first_signal_week, tier_signals["semi_luxury"].first_signal_week
        ),
        "lag_semi_to_affordable_weeks": _week_delta(
            tier_signals["semi_luxury"].first_signal_week, tier_signals["affordable"].first_signal_week
        ),
        "classification": classify(tier_signals),
    }


def run(
    weekly_mentions_path: Path = WEEKLY_MENTIONS_PATH,
    weekly_baseline_path: Path = WEEKLY_BASELINE_PATH,
    output_path: Path = TREND_DIFFUSION_CSV,
) -> pd.DataFrame:
    weekly_mentions_df = pd.read_parquet(weekly_mentions_path)
    weekly_baseline_df = pd.read_parquet(weekly_baseline_path)
    trends = TrendsConfig.from_yaml()

    rows = [analyze_trend(t.id, weekly_mentions_df, weekly_baseline_df) for t in trends.trends]
    out = pd.DataFrame(rows)
    labels = {t.id: t.label for t in trends.trends}
    out.insert(1, "label", out["trend_id"].map(labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(output_path)
    log.info("analyze complete: wrote %d trend rows to %s", len(out), output_path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
