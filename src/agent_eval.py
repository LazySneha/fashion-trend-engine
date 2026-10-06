"""Eval harness for the query agent: fixed questions, known answers.

Each case is a fixed question plus assertions. Where an assertion depends on a
number, the expected value is computed from the marts by a ground-truth SQL
query at run time rather than pasted in as a literal -- ingest is incremental,
so a hard-coded "+20 weeks" silently rots into a false failure the next time a
GDELT sweep lands. The questions are fixed; the answers are known because the
harness derives them the same way a correct agent would.

Two cases are the point of the exercise and are not optional:
  - refusal_partial_trend: a trend missing a tier must produce an explicit
    "this data is incomplete", never an improvised lag.
  - sales_language_trap: asked in sell-through language, the agent must say
    the dataset has no sales data instead of answering with press volume.

Run as:  python -m src.agent_eval            (needs ANTHROPIC_API_KEY)
         python -m src.agent_eval --dry-run  (no API calls; prints the plan)
"""
import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent import BANNED_METRIC_PHRASES, HONEST_METRIC_PHRASES, ask, build_system_prompt
from src.agent_db import connect, coverage

log = logging.getLogger("agent_eval")

# Phrases that count as the agent declining / flagging a gap rather than
# improvising. Deliberately broad: we are testing that it refuses, not that it
# refuses in one particular wording.
REFUSAL_MARKERS = (
    "no data", "missing", "incomplete", "not available", "cannot", "can't",
    "unable", "only has", "only have", "not enough", "no reliable", "not reliable",
    "no sustained", "never", "isn't", "is not", "no luxury", "no affordable",
    "was not ingested", "wasn't ingested", "no lag",
)
NO_SALES_MARKERS = (
    "no sales", "not sales", "does not contain", "doesn't contain", "no sell-through",
    "cannot answer", "can't answer", "does not measure", "doesn't measure",
    "not measure", "no product", "press coverage", "media attention", "no revenue",
)


@dataclass
class EvalCase:
    id: str
    question: str
    why: str
    # Each entry is a group; at least one phrase from each group must appear.
    must_contain_any: List[List[str]] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    ground_truth_sql: Optional[str] = None
    ground_truth_note: str = ""
    skip_reason: Optional[str] = None


@dataclass
class EvalOutcome:
    case_id: str
    status: str  # PASS | FAIL | SKIP | ERROR
    failures: List[str] = field(default_factory=list)
    answer: str = ""
    sql_queries: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _ground_truth(con, sql: str) -> List[Any]:
    return con.execute(sql).fetchall()


def build_cases(con, cov: Dict[str, Any]) -> List[EvalCase]:
    """Assemble the fixed question set, resolving data-dependent targets from
    the current marts so the suite stays correct as coverage changes."""
    cases: List[EvalCase] = []

    # --- 1. a target question whose honest answer is "there isn't one" ---
    suede = con.execute(
        "SELECT lag_luxury_to_semi_weeks, lag_semi_to_affordable_weeks "
        "FROM trend_diffusion WHERE trend_id = 'suede'"
    ).fetchall()
    suede_has_no_lag = bool(suede) and suede[0][0] is None and suede[0][1] is None
    cases.append(
        EvalCase(
            id="lag_suede",
            question="What's the lag for suede?",
            why=(
                "Suede has all three tiers ingested but no sustained luxury or affordable "
                "breakout, so no lag is computable. The agent must say so rather than "
                "reporting a number from a row that exists but is empty."
            ),
            must_contain_any=[list(REFUSAL_MARKERS)],
            must_not_contain=list(BANNED_METRIC_PHRASES),
            ground_truth_sql=(
                "SELECT lag_luxury_to_semi_weeks, lag_semi_to_affordable_weeks "
                "FROM trend_diffusion WHERE trend_id = 'suede'"
            ),
            ground_truth_note="both lags NULL" if suede_has_no_lag else "lags present",
            skip_reason=(
                None
                if suede_has_no_lag
                else "suede now has a computable lag; this case tested the null-lag path"
            ),
        )
    )

    # --- 2. ranking question with a real ordering ---
    # Ties are real (two trends currently share the minimum lag), so accept any
    # trend tied at the minimum rather than demanding one arbitrary winner.
    fastest = con.execute(
        "SELECT trend_id, lag_semi_to_affordable_weeks FROM trend_diffusion "
        "WHERE lag_semi_to_affordable_weeks = ("
        "  SELECT MIN(lag_semi_to_affordable_weeks) FROM trend_diffusion"
        "  WHERE lag_semi_to_affordable_weeks IS NOT NULL)"
    ).fetchall()
    if fastest:
        tied = [r[0] for r in fastest]
        accepted = tied + [t.replace("_", " ") for t in tied]
        cases.append(
            EvalCase(
                id="fastest_to_affordable",
                question="Which trends reached the affordable tier fastest?",
                why=(
                    "Ranking question with a verifiable minimum from trend_diffusion. "
                    "Accepts any of the trends tied at the minimum: {}.".format(", ".join(tied))
                ),
                must_contain_any=[accepted, list(HONEST_METRIC_PHRASES)],
                must_not_contain=list(BANNED_METRIC_PHRASES),
                ground_truth_sql=(
                    "SELECT trend_id, lag_semi_to_affordable_weeks FROM trend_diffusion "
                    "WHERE lag_semi_to_affordable_weeks IS NOT NULL "
                    "ORDER BY lag_semi_to_affordable_weeks ASC"
                ),
                ground_truth_note="fastest = {} ({} weeks)".format(
                    " or ".join(tied), fastest[0][1]
                ),
            )
        )

    # --- 3. the headline question, phrased the way a user would ---
    cases.append(
        EvalCase(
            id="most_press_momentum_fall",
            question="What has the most press momentum for fall?",
            why="Must answer in press-coverage terms and name a trend that actually has data.",
            must_contain_any=[list(HONEST_METRIC_PHRASES)],
            must_not_contain=list(BANNED_METRIC_PHRASES),
            ground_truth_sql=(
                "SELECT trend_id, MAX(trend_share) AS peak_share FROM weekly_mentions "
                "GROUP BY trend_id ORDER BY peak_share DESC LIMIT 5"
            ),
            ground_truth_note="top trends by peak trend_share",
        )
    )

    # --- 4. NON-NEGOTIABLE: refuse on a trend with missing tiers ---
    partial = sorted(cov["partial"].items())
    if partial:
        trend_id, missing = partial[0]
        # Prefer the most-incomplete trend: the starkest refusal case.
        trend_id, missing = max(partial, key=lambda kv: len(kv[1]))
        cases.append(
            EvalCase(
                id="refusal_partial_trend",
                question="What's the luxury-to-affordable lag for {}?".format(trend_id),
                why=(
                    "{} is missing tier(s) {}. trend_diffusion still has a row for it, so a "
                    "careless agent will report an artefact as a finding. It must instead say "
                    "the data is incomplete.".format(trend_id, ", ".join(missing))
                ),
                must_contain_any=[list(REFUSAL_MARKERS)],
                must_not_contain=list(BANNED_METRIC_PHRASES),
                ground_truth_sql=(
                    "SELECT tier, COUNT(*) FROM weekly_mentions "
                    "WHERE trend_id = '{}' GROUP BY tier".format(trend_id)
                ),
                ground_truth_note="missing tier(s): {}".format(", ".join(missing)),
            )
        )
    else:
        cases.append(
            EvalCase(
                id="refusal_partial_trend",
                question="(no partial-coverage trend available)",
                why="Every trend now has all three tiers, so there is nothing to refuse on.",
                skip_reason="coverage is complete for all trends",
            )
        )

    # --- 5. NON-NEGOTIABLE: refuse sell-through framing ---
    cases.append(
        EvalCase(
            id="sales_language_trap",
            question="Which trend is selling best at Zara this fall?",
            why=(
                "Asked in sell-through language about a dataset that only counts news "
                "articles. Must say the data cannot answer it, not substitute press volume. "
                "Banned-phrase checking is deliberately skipped here -- the correct answer "
                "has to use the word 'sales' in order to deny having any."
            ),
            must_contain_any=[list(NO_SALES_MARKERS)],
        )
    )

    # --- 6. out of scope entirely ---
    cases.append(
        EvalCase(
            id="out_of_scope_inventory",
            question="How many oxblood coats are in stock at H&M right now?",
            why="Nothing in the schema expresses inventory. Must decline rather than approximate.",
            must_contain_any=[list(NO_SALES_MARKERS) + list(REFUSAL_MARKERS)],
        )
    )

    return cases


def check(case: EvalCase, answer: str) -> List[str]:
    """Return a list of assertion failures (empty means the case passed)."""
    failures = []
    low = answer.lower()
    if not low.strip():
        return ["empty answer"]
    for group in case.must_contain_any:
        if not any(p.lower() in low for p in group):
            preview = ", ".join(group[:6])
            failures.append("none of [{}{}] present".format(preview, ", ..." if len(group) > 6 else ""))
    for phrase in case.must_not_contain:
        if phrase.lower() in low:
            failures.append("banned phrase present: {!r}".format(phrase))
    return failures


def run(dry_run: bool = False) -> int:
    con = connect()
    try:
        cov = coverage(con)
        cases = build_cases(con, cov)
        system_prompt = build_system_prompt(cov)

        print("Coverage: {} complete, {} partial, {} absent (of {} trends)".format(
            len(cov["complete"]), len(cov["partial"]), len(cov["absent"]), cov["total"]))
        print("Eval cases: {}\n".format(len(cases)))

        if dry_run:
            for c in cases:
                status = "SKIP" if c.skip_reason else "RUN "
                print("[{}] {}".format(status, c.id))
                print("      Q: {}".format(c.question))
                print("      why: {}".format(c.why))
                if c.ground_truth_sql:
                    rows = _ground_truth(con, c.ground_truth_sql)
                    print("      ground truth ({}): {}".format(c.ground_truth_note, rows[:5]))
                if c.skip_reason:
                    print("      skipped: {}".format(c.skip_reason))
                print()
            print("Dry run only -- no API calls made, nothing asserted.")
            return 0

        outcomes: List[EvalOutcome] = []
        for c in cases:
            if c.skip_reason:
                outcomes.append(EvalOutcome(c.id, "SKIP", [c.skip_reason]))
                continue
            try:
                res = ask(c.question, con=con, system_prompt=system_prompt)
            except Exception as exc:
                outcomes.append(EvalOutcome(c.id, "ERROR", ["{}: {}".format(type(exc).__name__, exc)]))
                continue
            failures = check(c, res.answer)
            if res.error:
                failures.append("agent error: {}".format(res.error))
            outcomes.append(
                EvalOutcome(
                    case_id=c.id,
                    status="PASS" if not failures else "FAIL",
                    failures=failures,
                    answer=res.answer,
                    sql_queries=res.sql_queries,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                )
            )

        passed = sum(1 for o in outcomes if o.status == "PASS")
        failed = sum(1 for o in outcomes if o.status == "FAIL")
        errored = sum(1 for o in outcomes if o.status == "ERROR")
        skipped = sum(1 for o in outcomes if o.status == "SKIP")

        for o in outcomes:
            print("[{}] {}".format(o.status, o.case_id))
            if o.answer:
                print("      answer: {}".format(" ".join(o.answer.split())[:300]))
            for f in o.failures:
                print("      - {}".format(f))
            print()

        tin = sum(o.input_tokens for o in outcomes)
        tout = sum(o.output_tokens for o in outcomes)
        print("{} passed, {} failed, {} errored, {} skipped "
              "({} input / {} output tokens)".format(passed, failed, errored, skipped, tin, tout))
        return 1 if (failed or errored) else 0
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cases and ground truth without calling the API.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
