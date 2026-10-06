"""Natural-language query agent over the trend marts.

One question in, one answer out, with Claude writing DuckDB SQL against the
existing parquet marts through a single tool. It never calls GDELT -- the only
data path is src.agent_db, which reads the marts and nothing else, so a
question can never cost an API fetch.

Two behaviours matter more than fluency here, and both are enforced in the
system prompt and covered by src.agent_eval:

1. It refuses to answer past its data. Ingest is incremental and GDELT
   throttles, so most trends are missing at least one tier. Coverage is
   computed from the marts at startup and injected into the prompt, because an
   empty result set is indistinguishable from a real zero unless you already
   know the tier was never fetched.

2. It describes the metric honestly. This pipeline measures press coverage
   volume, not sell-through. "Most press momentum" is accurate; "best selling"
   is a claim the data cannot support at all.

Run as:  python -m src.agent "which trends reached the affordable tier fastest?"
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent_db import (
    SCHEMA_DOC,
    SqlGuardError,
    connect,
    coverage,
    coverage_prompt_block,
    run_query,
)

log = logging.getLogger("agent")

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
EFFORT = "medium"
# A question needs 2 API calls in the normal case (one to write the SQL, one to
# read results and answer). The cap exists so a confused model retrying bad SQL
# can't spend without bound.
MAX_ITERATIONS = 6

# Vocabulary that would misrepresent what this pipeline measures. Asserted
# against in the eval harness, not just described in prose.
#
# Split deliberately. A claim phrase is wrong however it is used -- there is no
# sentence about this dataset in which "best selling" is correct. A sales noun
# is different: the *right* answer often has to use the word in order to deny
# having any ("it has no sales data"), so those are only a failure when used
# affirmatively. src.agent_eval checks them against surrounding negation.
BANNED_CLAIM_PHRASES = (
    "best selling", "best-selling", "bestselling", "top selling", "top-selling",
    "selling best", "most popular", "shoppers bought", "consumers bought",
    "units sold",
)
SALES_NOUNS = (
    "sales", "sell-through", "sellthrough", "revenue", "sold", "purchased",
)
HONEST_METRIC_PHRASES = (
    "press", "media", "coverage", "mention", "attention", "momentum", "article",
)

SYSTEM_PROMPT_TEMPLATE = """\
You answer questions about fashion trend diffusion by querying a small DuckDB \
warehouse with the run_sql tool.

WHAT THE DATA ACTUALLY MEASURES -- this governs every word you write:
This pipeline counts NEWS ARTICLES from the GDELT API. It measures PRESS \
COVERAGE VOLUME and nothing else. It contains no sales data, no product \
catalogues, no inventory, no sell-through, no revenue, and no consumer \
behaviour of any kind.

Therefore:
- Say "most press momentum", "most media attention", "most press coverage".
- NEVER say "best selling", "top selling", "sell-through", "sales", "revenue", \
"most popular", or "what shoppers are buying". The data cannot support those \
claims and using that language would be a factual error.
- Do not say a trend is "trending" or "winning" without naming the metric. \
Say "trending in press coverage" or "leading in media attention" instead.
- A trend reaching the affordable tier means AFFORDABLE-BRAND PRESS began \
covering it -- NOT that any retailer shipped a product. If a question implies \
product adoption, say plainly that this dataset cannot answer that.

{coverage_block}

REFUSAL RULES -- follow these even when a query returns rows:
- If asked about a trend with PARTIAL data, say explicitly which tier(s) are \
missing and that no reliable diffusion answer exists for it. Do NOT report a \
lag, a rank, or a classification for that trend as if it were a finding. The \
trend_diffusion table still has a row for such trends, but its values are \
artefacts of missing data, not results.
- If asked about a trend with NO data, say so and stop. Do not substitute a \
similar trend.
- If asked about something the schema cannot express (sales, prices, \
inventory, social media, consumer sentiment), say the dataset does not contain \
it. Do not approximate it with press coverage and do not guess.
- Never invent a number. Every figure you state must come from a query result \
in this conversation.

SCHEMA:
{schema}

HOW TO WORK:
- Call run_sql to get the facts you need, then answer in plain prose.
- Alias every computed column with an explicit AS and a double-quoted name, \
e.g. COUNT(*) AS "weeks". Several natural alias names (weeks, day, month, \
year) are reserved interval keywords in DuckDB and fail to parse as bare \
aliases, costing you a wasted turn.
- Prefer trend_diffusion for lag/diffusion questions and weekly_mentions for \
time-series or "most momentum" questions.
- Rank by trend_share, never by mention_count, when comparing across tiers.
- When a question is about "fall", that means the FW26 season; the runway \
shows ran Feb-Mar 2026.
- Keep the answer to a few sentences. State the metric plainly, give the \
numbers you found, and name any data gap that affects the answer."""

RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": (
        "Run one read-only SELECT against the trend marts (DuckDB). "
        "Returns columns and rows. Only a single SELECT is permitted; DDL, DML, "
        "stacked statements, and file-reading functions are rejected by a parser "
        "before execution. Results are capped at 200 rows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single DuckDB SELECT over weekly_mentions, weekly_baseline, or trend_diffusion.",
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class AgentResult:
    question: str
    answer: str
    sql_queries: List[str] = field(default_factory=list)
    iterations: int = 0
    stop_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


def build_system_prompt(cov: Optional[Dict[str, Any]] = None) -> str:
    if cov is None:
        cov = coverage()
    return SYSTEM_PROMPT_TEMPLATE.format(
        coverage_block=coverage_prompt_block(cov), schema=SCHEMA_DOC
    )


def _client():
    """Import and construct the Anthropic client lazily, so that importing this
    module (for tests, or for the SQL layer alone) needs no API key."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is required to run the agent: pip install anthropic"
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. The agent needs Claude API credentials; "
            "the SQL layer (src.agent_db) and its tests do not."
        )
    return anthropic.Anthropic()


def ask(question: str, con=None, system_prompt: Optional[str] = None) -> AgentResult:
    """Answer one question. Opens its own DuckDB connection unless given one."""
    owns_con = con is None
    if owns_con:
        con = connect()
    if system_prompt is None:
        system_prompt = build_system_prompt()

    result = AgentResult(question=question, answer="")
    try:
        client = _client()
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]

        for i in range(MAX_ITERATIONS):
            result.iterations = i + 1
            # tool_choice is left at its default (auto): Opus 5.5 rejects a
            # forced tool_choice with a 400, so the prompt does the steering.
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[RUN_SQL_TOOL],
                messages=messages,
            )
            result.stop_reason = response.stop_reason
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens

            # Always check stop_reason before reading content: a policy decline
            # returns HTTP 200 with no usable answer.
            if response.stop_reason == "refusal":
                detail = getattr(response, "stop_details", None)
                result.error = "Model declined the request" + (
                    f" ({detail.category})" if detail is not None else ""
                )
                result.answer = result.error
                return result

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                result.answer = "\n".join(
                    b.text for b in response.content if b.type == "text"
                ).strip()
                return result

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                sql = block.input.get("sql", "") if isinstance(block.input, dict) else ""
                result.sql_queries.append(sql)
                try:
                    out = run_query(con, sql, question=question)
                    if out["error"]:
                        payload = f"Query failed: {out['error']}\nFix the SQL and try again."
                        is_error = True
                    else:
                        payload = json.dumps(
                            {
                                "columns": out["columns"],
                                "rows": out["rows"],
                                "row_count": out["row_count"],
                                "truncated": out["truncated"],
                            },
                            default=str,
                        )
                        is_error = False
                except SqlGuardError as exc:
                    payload = f"Rejected by SQL guardrail: {exc}"
                    is_error = True
                    log.warning("guardrail rejected SQL: %s", exc)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": payload,
                        "is_error": is_error,
                    }
                )

            if not tool_results:
                # stop_reason said tool_use but no tool_use block arrived. Bail
                # rather than post an empty user turn, which the API rejects.
                result.error = "stop_reason was tool_use but no tool_use block was present."
                result.answer = result.error
                return result

            # All results go back in ONE user message: splitting them trains the
            # model out of making parallel calls.
            messages.append({"role": "user", "content": tool_results})

        result.error = f"Hit MAX_ITERATIONS ({MAX_ITERATIONS}) without a final answer."
        result.answer = result.error
        return result
    finally:
        if owns_con:
            con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the trend marts a question.")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--show-sql", action="store_true", help="Print the SQL the agent ran.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = ask(" ".join(args.question))

    if args.show_sql:
        for q in res.sql_queries:
            print("SQL> " + " ".join(q.split()))
        print()
    print(res.answer)
    print(
        "\n[{} API call(s), {} in / {} out tokens]".format(
            res.iterations, res.input_tokens, res.output_tokens
        ),
        file=sys.stderr,
    )
    return 1 if res.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
