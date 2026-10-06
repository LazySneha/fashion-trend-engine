"""DuckDB access layer for the query agent: read-only views over the parquet
marts, a parsed SELECT-only guardrail, data-coverage facts, and query logging.

This module makes no network calls of any kind -- not to GDELT, not to the
Claude API. It is the only place the agent touches data, so the "the agent
never re-fetches from GDELT" guarantee is structural: there is no code path
here that could.

The guardrail parses the SQL rather than pattern-matching it. DuckDB's own
`json_serialize_sql` is the parser, so what we validate is exactly what DuckDB
would execute -- a regex can be fooled by comments, string literals, or
stacked statements, and a parser cannot.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from src.config import (
    OUTPUT_DIR,
    TREND_DIFFUSION_CSV,
    WEEKLY_BASELINE_PATH,
    WEEKLY_MENTIONS_PATH,
)
from src.config_models import TrendsConfig

log = logging.getLogger("agent_db")

AGENT_QUERY_LOG = OUTPUT_DIR / "agent_queries.jsonl"

# Hard cap on rows handed back to the model. A runaway SELECT would otherwise
# dump the whole mart into the context window and cost real money.
MAX_RESULT_ROWS = 200

# Defense in depth behind the parser. The parse already guarantees a single
# SELECT, but a SELECT can still reach the filesystem through DuckDB's table
# functions (read_csv, read_parquet, ...). The agent only ever needs the three
# views below, so any file/extension-reaching identifier is rejected outright.
_BLOCKED_IDENTIFIERS = (
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "read_ndjson", "glob", "parquet_scan", "csv_scan",
    "install", "load", "attach", "detach", "copy", "export", "import",
    "pragma_table_info", "pragma_database_list", "getenv", "shell",
)


class SqlGuardError(Exception):
    """Raised when candidate SQL is not a single, plain SELECT over the marts."""


# --- views ---

VIEW_SOURCES = {
    "weekly_mentions": WEEKLY_MENTIONS_PATH,
    "weekly_baseline": WEEKLY_BASELINE_PATH,
    "trend_diffusion": TREND_DIFFUSION_CSV,
}

SCHEMA_DOC = """\
weekly_mentions  -- one row per (trend, tier, week)
  trend_id                 TEXT     e.g. 'oxblood'
  tier                     TEXT     'luxury' | 'semi_luxury' | 'affordable'
  week_start               DATE     Monday of the week
  mention_count            DOUBLE   raw article count for the trend in that tier
  baseline_mention_count   DOUBLE   that tier's total brand-press articles that week
  trend_share              DOUBLE   mention_count / baseline_mention_count (NULL if baseline was 0)

weekly_baseline  -- one row per (tier, week): the tier's own press volume
  tier                     TEXT
  week_start               DATE
  baseline_mention_count   DOUBLE

trend_diffusion  -- one row per trend: the computed diffusion result
  trend_id                        TEXT
  label                           TEXT     human-readable trend name
  luxury_first_signal_week        DATE     NULL if no sustained breakout detected
  luxury_peak_week                DATE
  semi_luxury_first_signal_week   DATE
  semi_luxury_peak_week           DATE
  affordable_first_signal_week    DATE
  affordable_peak_week            DATE
  lag_luxury_to_semi_weeks        BIGINT   weeks from luxury's first signal to semi-luxury's
  lag_semi_to_affordable_weeks    BIGINT   weeks from semi-luxury's first signal to affordable's
  classification                  TEXT     'mass' | 'spreading' | 'emerging' | 'no_signal'

Always compare tiers using trend_share, never mention_count: luxury brands get
structurally more press than affordable brands regardless of any trend, so raw
counts are not comparable across tiers."""


def connect(read_only_views: bool = True) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per mart. Raises if a mart is missing --
    the agent must never silently answer from a partial warehouse."""
    missing = [str(p) for p in VIEW_SOURCES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Mart files missing, run `make transform analyze` first: " + ", ".join(missing)
        )
    con = duckdb.connect()
    for view, path in VIEW_SOURCES.items():
        reader = "read_csv_auto" if path.suffix == ".csv" else "read_parquet"
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM {reader}('{path}')")
    return con


# --- guardrail ---


def validate_select(sql: str) -> None:
    """Raise SqlGuardError unless `sql` is exactly one plain SELECT statement.

    Uses DuckDB's parser (json_serialize_sql), not a regex: stacked statements,
    DDL/DML, and anything that isn't a SELECT_NODE fail to parse or fail the
    node-type check. A second pass rejects filesystem-reaching identifiers.
    """
    if not sql or not sql.strip():
        raise SqlGuardError("Empty SQL.")

    probe = duckdb.connect()
    try:
        raw = probe.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        parsed = json.loads(raw)
    except Exception as exc:
        raise SqlGuardError(f"SQL could not be parsed: {exc}") from exc
    finally:
        probe.close()

    if parsed.get("error"):
        # DuckDB's serializer only accepts SELECT; DDL/DML and stacked
        # statements land here.
        detail = parsed.get("error_message") or parsed.get("error_type") or "not a SELECT statement"
        raise SqlGuardError(f"Only a single SELECT is allowed. Parser said: {detail}")

    statements = parsed.get("statements") or []
    if len(statements) != 1:
        raise SqlGuardError(f"Expected exactly 1 statement, got {len(statements)}.")

    node_type = (statements[0].get("node") or {}).get("type")
    if node_type not in ("SELECT_NODE", "SET_OPERATION_NODE"):
        raise SqlGuardError(f"Statement is {node_type}, not a SELECT.")

    # String literals are blanked first so a denylisted word inside a quoted
    # value (a trend label, say) can't trip the check.
    lowered = re.sub(r"'[^']*'", "''", sql.lower())
    for ident in _BLOCKED_IDENTIFIERS:
        if re.search(r"\b" + re.escape(ident) + r"\s*\(", lowered):
            raise SqlGuardError(
                f"Identifier '{ident}' is not allowed; query the views "
                f"({', '.join(VIEW_SOURCES)}) only."
            )


# --- execution + logging ---


def _log_query(record: Dict[str, Any]) -> None:
    AGENT_QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_QUERY_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def run_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    question: Optional[str] = None,
    max_rows: int = MAX_RESULT_ROWS,
) -> Dict[str, Any]:
    """Validate, execute, log. Returns a dict with columns/rows/truncated, or
    raises SqlGuardError. Execution errors are returned (not raised) so the
    agent can see its own mistake and correct the SQL on the next turn."""
    validate_select(sql)

    started = time.time()
    error = None
    columns: List[str] = []
    rows: List[List[Any]] = []
    truncated = False
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description]
        fetched = cur.fetchmany(max_rows + 1)
        if len(fetched) > max_rows:
            truncated = True
            fetched = fetched[:max_rows]
        rows = [list(r) for r in fetched]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = int((time.time() - started) * 1000)
    _log_query(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "sql": " ".join(sql.split()),
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
    )
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "error": error,
        "elapsed_ms": elapsed_ms,
    }


# --- coverage ---


def coverage(con: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, Any]:
    """Which trends actually have data, per tier.

    Ingest is incremental and GDELT throttles, so the marts are routinely
    partial. The agent is told this up front rather than being left to infer
    it from empty result sets -- an empty set is indistinguishable from a real
    zero unless you know the tier was never fetched.
    """
    owns = con is None
    if owns:
        con = connect()
    try:
        present = con.execute(
            "SELECT trend_id, tier FROM weekly_mentions GROUP BY 1, 2"
        ).fetchall()
    finally:
        if owns:
            con.close()

    by_trend: Dict[str, List[str]] = {}
    for trend_id, tier in present:
        by_trend.setdefault(trend_id, []).append(tier)

    all_tiers = {"luxury", "semi_luxury", "affordable"}
    trends = TrendsConfig.from_yaml().trends
    labels = {t.id: t.label for t in trends}

    complete, partial, absent = [], {}, []
    for t in trends:
        tiers = sorted(by_trend.get(t.id, []))
        if not tiers:
            absent.append(t.id)
        elif set(tiers) == all_tiers:
            complete.append(t.id)
        else:
            partial[t.id] = sorted(all_tiers - set(tiers))

    return {
        "labels": labels,
        "complete": complete,
        "partial": partial,  # trend_id -> list of MISSING tiers
        "absent": absent,  # no data at all for any tier
        "total": len(trends),
    }


def coverage_prompt_block(cov: Dict[str, Any]) -> str:
    """Render coverage as the system-prompt section the refusal rules key off."""
    lines = [
        "DATA COVERAGE (authoritative -- derived from the marts at startup):",
        "A trend's diffusion result is only trustworthy when all three tiers were "
        "ingested. GDELT throttling left parts of the sweep incomplete.",
        "",
        "Trends with COMPLETE data (all three tiers) -- safe to answer in full:",
    ]
    lines.append("  " + (", ".join(cov["complete"]) if cov["complete"] else "(none)"))
    lines.append("")
    lines.append("Trends with PARTIAL data -- you MUST NOT present a lag, a ranking "
                 "position, or a diffusion verdict for these:")
    if cov["partial"]:
        for trend_id, missing in sorted(cov["partial"].items()):
            lines.append(f"  {trend_id}: MISSING tier(s) {', '.join(missing)}")
    else:
        lines.append("  (none)")
    if cov["absent"]:
        lines.append("")
        lines.append("Trends with NO data at all -- refuse outright:")
        lines.append("  " + ", ".join(cov["absent"]))
    return "\n".join(lines)
