"""weekly_mentions.parquet -> one PNG per trend, charts/{trend_id}.png.

Plots trend_share (mentions as a % of that tier's own baseline press volume
that week), not raw counts -- raw counts aren't comparable across tiers.
Color slots follow the validated dataviz palette's first three categorical
hues in fixed order (never cycled/reassigned): luxury=blue, semi_luxury=
orange, affordable=aqua. Those three slots are the ones documented to clear
CVD/contrast checks against each other; aqua sits below 3:1 contrast on a
light surface on its own, so every line also gets a direct end-of-line label
(not just a legend) per the palette's relief rule.
"""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.config import CHARTS_DIR, TIERS, WEEKLY_MENTIONS_PATH
from src.config_models import TrendsConfig

log = logging.getLogger("charts")

TIER_COLORS = {"luxury": "#2a78d6", "semi_luxury": "#eb6834", "affordable": "#1baf7a"}
TIER_LABELS = {"luxury": "Luxury", "semi_luxury": "Semi-luxury", "affordable": "Affordable"}

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# FW26 runway shows (the trend-setting event this whole project hinges on).
FW26_START = pd.Timestamp("2026-02-01")
FW26_END = pd.Timestamp("2026-03-31")


def plot_trend(trend_id: str, label: str, weekly_mentions_df: pd.DataFrame, out_dir: Path = CHARTS_DIR) -> Path:
    sub_all = weekly_mentions_df[weekly_mentions_df["trend_id"] == trend_id]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    all_weeks = []
    label_points = []  # (tier, true_y) -- placed after all lines are drawn, once ylim is known
    for tier in TIERS:
        sub = sub_all[sub_all["tier"] == tier].sort_values("week_start")
        if sub.empty:
            continue
        x = sub["week_start"]
        y = sub["trend_share"] * 100
        ax.plot(x, y, color=TIER_COLORS[tier], linewidth=2, solid_capstyle="round", label=TIER_LABELS[tier])
        all_weeks.append(x)

        valid = sub.dropna(subset=["trend_share"])
        if not valid.empty:
            label_points.append((tier, float(valid.iloc[-1]["trend_share"]) * 100))

    if all_weeks:
        x_min = min(w.min() for w in all_weeks)
        x_max = max(w.max() for w in all_weeks)
        ax.set_xlim(x_min - pd.Timedelta(days=3), x_max + pd.Timedelta(days=28))

        # Direct end-of-line labels, all anchored at the same right-edge x so
        # they read as a clean stack -- but series that end near the same
        # share value would otherwise collide into unreadable overlapping
        # text, so nudge them apart vertically, preserving relative order.
        if label_points:
            ax.relim()
            ax.autoscale_view()
            y_lo, y_hi = ax.get_ylim()
            min_gap = 0.045 * (y_hi - y_lo)
            label_points.sort(key=lambda p: p[1])
            placed = []
            prev_y = None
            for tier, true_y in label_points:
                y_pos = true_y if prev_y is None else max(true_y, prev_y + min_gap)
                placed.append((tier, y_pos))
                prev_y = y_pos
            label_x = x_max + pd.Timedelta(days=6)
            for tier, y_pos in placed:
                ax.text(label_x, y_pos, TIER_LABELS[tier], fontsize=9, color=INK_SECONDARY, va="center")
        if FW26_START <= x_max and FW26_END >= x_min:
            ax.axvspan(FW26_START, FW26_END, color=GRID, alpha=0.8, zorder=0)
            ax.text(
                FW26_START,
                1.0,
                " FW26 shows",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                color=INK_MUTED,
                va="bottom",
            )

    ax.set_title(f"{label} — mention share by tier", fontsize=13, color=INK_PRIMARY, loc="left", pad=12)
    ax.set_ylabel("Trend mentions as % of tier's press volume", fontsize=9, color=INK_SECONDARY)

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.text(
        0.01,
        0.01,
        "Share = trend mentions / tier's total brand-press mentions that week (GDELT DOC 2.0, news volume).",
        fontsize=7,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trend_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(weekly_mentions_path: Path = WEEKLY_MENTIONS_PATH, out_dir: Path = CHARTS_DIR) -> None:
    weekly_mentions_df = pd.read_parquet(weekly_mentions_path)
    trends = TrendsConfig.from_yaml()
    for trend in trends.trends:
        path = plot_trend(trend.id, trend.label, weekly_mentions_df, out_dir)
        log.info("wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
