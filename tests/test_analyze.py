import math

import pandas as pd
import pytest

from src import analyze
from src.config import SIGNAL_FACTOR, SUSTAINED_WEEKS, TRAILING_BASELINE_WINDOW_WEEKS

QUIET_N = TRAILING_BASELINE_WINDOW_WEEKS + 2  # enough history for a trailing median to exist


# --- unit tests on the primitives (exercise the config constants directly) ---


def test_elevated_weeks_respects_trailing_window_and_signal_factor():
    values = [0.01] * QUIET_N + [0.05] * 5  # jump clears SIGNAL_FACTOR=2.0 over a 0.01 baseline
    idx = pd.date_range("2026-01-05", periods=len(values), freq="7D")
    s = pd.Series(values, index=idx)
    elevated = analyze.elevated_weeks(s)
    assert not elevated.iloc[:TRAILING_BASELINE_WINDOW_WEEKS].any()  # no signal before enough trailing history
    # the jump itself, and the week right after, read as elevated
    assert elevated.iloc[QUIET_N]
    assert elevated.iloc[QUIET_N + 1]


def test_first_sustained_week_requires_n_consecutive_weeks():
    n = SUSTAINED_WEEKS
    values = [False, True, False, False] + [True] * n + [False]
    idx = pd.date_range("2026-01-05", periods=len(values), freq="7D")
    elevated = pd.Series(values, index=idx)
    assert analyze.first_sustained_week(elevated) == idx[4]


def test_single_stray_elevated_week_never_sets_signal():
    idx = pd.date_range("2026-01-05", periods=4, freq="7D")
    elevated = pd.Series([False, True, False, False], index=idx)
    assert analyze.first_sustained_week(elevated) is None


def test_peak_week_tie_break_picks_earliest():
    idx = pd.date_range("2026-01-05", periods=5, freq="7D")
    s = pd.Series([0.10, 0.15, 0.12, 0.15, 0.09], index=idx)
    elevated = pd.Series([True] * 5, index=idx)
    assert analyze.peak_week_among_elevated(s, elevated) == idx[1]


def test_share_series_fills_true_gaps_but_preserves_existing_nan():
    calendar = pd.date_range("2026-01-05", periods=3, freq="7D")
    weekly_mentions_df = pd.DataFrame(
        {
            "trend_id": ["suede", "suede"],
            "tier": ["luxury", "luxury"],
            "week_start": [calendar[0], calendar[2]],
            "trend_share": [0.1, float("nan")],
        }
    )
    # calendar[1] has no row at all for this trend/tier -- a true gap.
    s = analyze.share_series(weekly_mentions_df, "suede", "luxury", calendar)
    assert s.iloc[0] == 0.1
    assert s.iloc[1] == 0.0
    assert math.isnan(s.iloc[2])  # present-but-undefined (zero baseline) stays NaN, not coerced to 0


# --- end-to-end analyze_trend tests ---


def _tier_frames(tier, breakout_week, n_weeks=30, quiet_share=0.01, elevated_share=0.10, baseline=100.0):
    idx = pd.date_range("2026-01-05", periods=n_weeks, freq="7D")
    shares = [quiet_share if i < breakout_week else elevated_share for i in range(n_weeks)]
    mentions = pd.DataFrame(
        {
            "trend_id": "suede",
            "tier": tier,
            "week_start": idx,
            "mention_count": [s * baseline for s in shares],
            "baseline_mention_count": baseline,
            "trend_share": shares,
        }
    )
    baseline_df = pd.DataFrame({"tier": tier, "week_start": idx, "baseline_mention_count": baseline})
    return mentions, baseline_df


def test_analyze_trend_normal_case_lag_and_peak():
    lux_m, lux_b = _tier_frames("luxury", breakout_week=10)
    semi_m, semi_b = _tier_frames("semi_luxury", breakout_week=13)
    aff_m, aff_b = _tier_frames("affordable", breakout_week=17)

    weekly_mentions_df = pd.concat([lux_m, semi_m, aff_m], ignore_index=True)
    weekly_baseline_df = pd.concat([lux_b, semi_b, aff_b], ignore_index=True)

    result = analyze.analyze_trend("suede", weekly_mentions_df, weekly_baseline_df)

    idx = pd.date_range("2026-01-05", periods=30, freq="7D")
    assert result["luxury_first_signal_week"] == idx[10]
    assert result["semi_luxury_first_signal_week"] == idx[13]
    assert result["affordable_first_signal_week"] == idx[17]
    assert result["lag_luxury_to_semi_weeks"] == 3
    assert result["lag_semi_to_affordable_weeks"] == 4
    assert result["classification"] == "mass"


def test_analyze_trend_no_semi_luxury_signal():
    lux_m, lux_b = _tier_frames("luxury", breakout_week=10)
    # semi_luxury never breaks out: elevated_share == quiet_share, i.e. flat.
    semi_m, semi_b = _tier_frames("semi_luxury", breakout_week=999, elevated_share=0.01)
    aff_m, aff_b = _tier_frames("affordable", breakout_week=17)

    weekly_mentions_df = pd.concat([lux_m, semi_m, aff_m], ignore_index=True)
    weekly_baseline_df = pd.concat([lux_b, semi_b, aff_b], ignore_index=True)

    result = analyze.analyze_trend("suede", weekly_mentions_df, weekly_baseline_df)

    assert result["semi_luxury_first_signal_week"] is None
    assert result["lag_luxury_to_semi_weeks"] is None
    assert result["lag_semi_to_affordable_weeks"] is None
    # affordable still reached signal despite semi_luxury having none --
    # classification is by furthest tier reached, not a strict tier chain.
    assert result["classification"] == "mass"


def test_analyze_trend_no_signal_anywhere_classifies_no_signal():
    lux_m, lux_b = _tier_frames("luxury", breakout_week=999, elevated_share=0.01)
    semi_m, semi_b = _tier_frames("semi_luxury", breakout_week=999, elevated_share=0.01)
    aff_m, aff_b = _tier_frames("affordable", breakout_week=999, elevated_share=0.01)

    weekly_mentions_df = pd.concat([lux_m, semi_m, aff_m], ignore_index=True)
    weekly_baseline_df = pd.concat([lux_b, semi_b, aff_b], ignore_index=True)

    result = analyze.analyze_trend("suede", weekly_mentions_df, weekly_baseline_df)
    assert result["classification"] == "no_signal"
    assert result["lag_luxury_to_semi_weeks"] is None
