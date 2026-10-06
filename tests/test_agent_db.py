"""Tests for the agent's data layer: the SELECT-only guardrail, result
capping, query logging, and coverage classification.

No network and no Claude API key needed -- everything here runs against
synthetic in-memory tables. The guardrail is the security boundary between a
language model and a database, so it is tested against the attacks it exists
to stop, not just the happy path.
"""
import json

import duckdb
import pytest

from src.agent_db import (
    SqlGuardError,
    coverage,
    coverage_prompt_block,
    run_query,
    validate_select,
)


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(
        "CREATE VIEW weekly_mentions AS SELECT * FROM (VALUES "
        "('oxblood','luxury',DATE '2026-01-05',10.0,100.0,0.10), "
        "('oxblood','semi_luxury',DATE '2026-01-05',5.0,50.0,0.10), "
        "('oxblood','affordable',DATE '2026-01-05',2.0,40.0,0.05), "
        "('velvet','semi_luxury',DATE '2026-01-05',1.0,50.0,0.02) "
        ") AS t(trend_id, tier, week_start, mention_count, baseline_mention_count, trend_share)"
    )
    yield c
    c.close()


# --- guardrail: what it must reject ---


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE weekly_mentions",
        "DELETE FROM weekly_mentions",
        "INSERT INTO weekly_mentions VALUES ('x','luxury',DATE '2026-01-05',1,1,1)",
        "UPDATE weekly_mentions SET tier = 'luxury'",
        "CREATE TABLE evil (a INT)",
        "ATTACH 'other.db'",
    ],
)
def test_guardrail_rejects_non_select(sql):
    with pytest.raises(SqlGuardError):
        validate_select(sql)


def test_guardrail_rejects_stacked_statements():
    # The classic injection shape: a valid SELECT followed by a destructive
    # statement. A regex on the leading keyword would wave this through.
    with pytest.raises(SqlGuardError):
        validate_select("SELECT 1; DROP TABLE weekly_mentions")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('/tmp/anything.parquet')",
        "SELECT * FROM glob('/**')",
    ],
)
def test_guardrail_rejects_filesystem_access(sql):
    # These parse as SELECTs, so the parser alone would allow them -- the
    # identifier denylist is the layer that stops them.
    with pytest.raises(SqlGuardError):
        validate_select(sql)


def test_guardrail_rejects_empty():
    with pytest.raises(SqlGuardError):
        validate_select("   ")


# --- guardrail: what it must allow ---


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM weekly_mentions",
        "SELECT trend_id, MAX(trend_share) FROM weekly_mentions GROUP BY trend_id",
        "WITH t AS (SELECT * FROM weekly_mentions) SELECT COUNT(*) FROM t",
        "SELECT trend_id FROM weekly_mentions UNION SELECT trend_id FROM weekly_mentions",
        "SELECT * FROM weekly_mentions ORDER BY week_start DESC LIMIT 10",
    ],
)
def test_guardrail_allows_plain_selects(sql):
    validate_select(sql)  # must not raise


def test_guardrail_allows_denylisted_word_inside_a_string_literal():
    # 'copy' is denylisted as a function, but must be fine as data.
    validate_select("SELECT * FROM weekly_mentions WHERE trend_id = 'copy(x)'")


# --- execution ---


def test_run_query_returns_columns_and_rows(con, tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", tmp_path / "q.jsonl")
    out = run_query(con, "SELECT trend_id, tier FROM weekly_mentions ORDER BY trend_id, tier")
    assert out["error"] is None
    assert out["columns"] == ["trend_id", "tier"]
    assert out["row_count"] == 4
    assert out["truncated"] is False


def test_run_query_caps_rows_and_flags_truncation(con, tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", tmp_path / "q.jsonl")
    out = run_query(con, "SELECT * FROM weekly_mentions", max_rows=2)
    assert out["row_count"] == 2
    assert out["truncated"] is True


def test_run_query_returns_sql_errors_instead_of_raising(con, tmp_path, monkeypatch):
    # The agent needs to see its own mistake to correct it on the next turn,
    # so a bad query is data, not an exception.
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", tmp_path / "q.jsonl")
    out = run_query(con, "SELECT nonexistent_column FROM weekly_mentions")
    assert out["error"] is not None
    assert out["rows"] == []


def test_run_query_still_raises_on_guardrail_violation(con, tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", tmp_path / "q.jsonl")
    with pytest.raises(SqlGuardError):
        run_query(con, "DROP TABLE weekly_mentions")


def test_every_query_is_logged(con, tmp_path, monkeypatch):
    log_path = tmp_path / "q.jsonl"
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", log_path)
    run_query(con, "SELECT 1 AS a", question="how many?")
    run_query(con, "SELECT 2 AS b", question="and again?")
    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["question"] == "how many?"
    assert lines[0]["sql"] == "SELECT 1 AS a"
    assert "ts" in lines[0] and "elapsed_ms" in lines[0]


# --- coverage ---


def test_coverage_separates_complete_from_partial(con):
    cov = coverage(con)
    assert "oxblood" in cov["complete"]
    assert cov["partial"]["velvet"] == ["affordable", "luxury"]
    # Trends in the config with no rows at all are 'absent', not 'partial'.
    assert "suede" in cov["absent"]


def test_coverage_prompt_block_names_missing_tiers(con):
    block = coverage_prompt_block(coverage(con))
    assert "velvet" in block
    assert "affordable" in block and "luxury" in block
    assert "PARTIAL" in block


# --- metric-honesty check (regression: it used to fail honest caveats) ---


# These four are verbatim excerpts from real eval runs. Two of them were
# scored as violations by earlier versions of this check -- once for saying
# "not sales", once for saying "nothing on sales". Denying having sales data is
# the behaviour the agent is supposed to have, so the fixtures are kept
# verbatim rather than trimmed to suit the rule.
@pytest.mark.parametrize(
    "answer",
    [
        "I can't answer that. This dataset has no sales data. It only counts news articles "
        "from GDELT, so it measures how much press coverage a trend gets. It has nothing on "
        "sales, sell-through, inventory or what shoppers buy at Zara or anywhere else.",
        "Sheer layering has the most press momentum for fall. I measured this with trend_share, "
        "the trend's share of each tier's brand-press articles. This is press coverage only, "
        "not sales or what shoppers are buying.",
        "I can't tell you what's selling best at Zara. This dataset counts news articles, so it "
        "only measures how much press a trend gets. It has no sales or sell-through data.",
        "It only counts news articles from GDELT, so it measures press coverage. It has no "
        "inventory, stock levels, product catalogues, or retailer data.",
    ],
)
def test_honest_sales_caveats_are_not_violations(answer):
    from src.agent_eval import sales_language_failures

    assert sales_language_failures(answer) == []


def test_sales_vocabulary_without_naming_the_metric_is_a_violation():
    # The rule that replaced negation-phrase matching: sales words are fine in
    # a disclaimer, but an answer that never says what it IS measuring has no
    # business using them.
    from src.agent_eval import sales_language_failures

    assert sales_language_failures("Suede had the highest sell-through in Q3.") != []
    assert sales_language_failures(
        "Suede led on sell-through. Measured as press coverage share."
    ) == []


@pytest.mark.parametrize(
    "answer",
    [
        "Oxblood is the best selling trend this fall.",
        "Velvet was top-selling in the affordable tier.",
        "Purple drove the strongest sales growth this season.",
        "Leopard is the most popular trend with shoppers.",
        # A claim phrase is still caught even when the metric IS named, because
        # naming the metric doesn't license misdescribing it.
        "By press coverage share, oxblood is the best selling trend this fall.",
    ],
)
def test_sales_claims_are_violations(answer):
    from src.agent_eval import sales_language_failures

    assert sales_language_failures(answer) != []


def test_guardrail_rejections_are_logged_then_raised(con, tmp_path, monkeypatch):
    # The rejected query is the one you most want to read later.
    log_path = tmp_path / "q.jsonl"
    monkeypatch.setattr("src.agent_db.AGENT_QUERY_LOG", log_path)
    with pytest.raises(SqlGuardError):
        run_query(con, "SELECT 1; DROP TABLE weekly_mentions", question="bad one")
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["question"] == "bad one"
    assert "DROP TABLE" in entry["sql"]
    assert entry["error"].startswith("REJECTED:")
