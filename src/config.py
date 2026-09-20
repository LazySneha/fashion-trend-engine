"""Shared paths and tunable constants for the Trickle-Down Trend Tracker pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
BRANDS_YAML = CONFIG_DIR / "brands.yaml"
TRENDS_YAML = CONFIG_DIR / "trends.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
MARTS_DIR = DATA_DIR / "marts"
OUTPUT_DIR = DATA_DIR / "output"
CHARTS_DIR = ROOT / "charts"

STAGING_MENTIONS_PATH = STAGING_DIR / "staging_mentions.parquet"
STAGING_BASELINE_PATH = STAGING_DIR / "staging_baseline.parquet"
WEEKLY_BASELINE_PATH = MARTS_DIR / "weekly_baseline.parquet"
WEEKLY_MENTIONS_PATH = MARTS_DIR / "weekly_mentions.parquet"
TREND_DIFFUSION_CSV = OUTPUT_DIR / "trend_diffusion.csv"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MODE = "timelinevolraw"
GDELT_FORMAT = "json"
TIMESPAN = "12m"

# --- GDELT client / rate limiting ---
# The API itself states: "Please limit requests to one every 5 seconds."
# Fail fast rather than hammer a stuck request: the raw cache is durable, so
# a failed task just gets retried cheaply on the next `make ingest` rather
# than needing to be rescued by an enormous in-process retry budget.
MIN_REQUEST_INTERVAL_SECONDS = 5.0
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 10.0
BACKOFF_JITTER_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0

# ingest_gdelt.py orchestration: give up on the whole run after this many
# consecutive task failures (each already exhausted its own MAX_RETRIES),
# rather than grinding through the remaining tasks one-by-one for nothing.
MAX_CONSECUTIVE_TASK_FAILURES = 3

# Sentinel trend_id used for the brands-only baseline query per tier.
BASELINE_TREND_ID = "_baseline"

TIERS = ("luxury", "semi_luxury", "affordable")

# --- Sustained-signal detection (analyze.py) ---
# A week is "elevated" if its 3-week rolling mean share exceeds the trailing
# median share (over the preceding TRAILING_BASELINE_WINDOW_WEEKS weeks) by
# SIGNAL_FACTOR. first_signal_week requires SUSTAINED_WEEKS consecutive
# elevated weeks, so a single stray article can never set a first-signal week.
SIGNAL_ROLLING_WINDOW_WEEKS = 3
TRAILING_BASELINE_WINDOW_WEEKS = 8
SIGNAL_FACTOR = 2.0
SUSTAINED_WEEKS = 2
