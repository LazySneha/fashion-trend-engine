# CLAUDE.md

Context for Claude Code sessions in this repo. Read `README.md` for the full write-up and
findings; this file is the working brief.

## What this is

**Trickle-Down Trend Tracker.** Measures how fashion trends diffuse across brand price tiers,
using public news coverage as a proxy for trend momentum.

**Hypothesis:** luxury houses set trends on the runway (Fall/Winter 2026 shows, Feb–Mar 2026),
semi-luxury adapts them next, affordable mass-produces them last — and that lag is measurable in
weeks.

Sole data source: the **GDELT DOC 2.0 API**, `mode=timelinevolraw` — daily article *counts* only.
No headlines, no URLs, no article text; the article-list endpoint is never called, so no
copyrighted content is fetched or stored.

## Architecture / data flow

```
config/*.yaml → ingest_gdelt → data/raw/*.json → transform → data/staging/*.parquet
                                                           → data/marts/*.parquet
                                                              ├→ analyze → data/output/trend_diffusion.csv
                                                              └→ charts  → charts/*.png
```

| Module | Role |
|---|---|
| `src/config.py` | All paths + tunable constants. Change thresholds here, not inline. |
| `src/config_models.py` | Pydantic models + the **only** readers of `config/*.yaml`. |
| `src/query_builder.py` | Pure GDELT query-string construction. No network, no I/O. |
| `src/gdelt_models.py` | Response schemas + the two error types. |
| `src/gdelt_client.py` | **Only** module that makes network calls. Rate limit, retry, cache. |
| `src/ingest_gdelt.py` | Orchestrates 45 tasks: 3 tier baselines + 14 trends × 3 tiers. |
| `src/transform.py` | raw → staging → weekly marts. Full recompute, atomic writes. |
| `src/analyze.py` | Sustained-signal detection, tier lags, classification. |
| `src/charts.py` | One PNG per trend, three tier lines each. |
| `src/agent_db.py` | DuckDB views over the marts, parsed SELECT-only guardrail, coverage, query log. |
| `src/agent.py` | NL query agent (Claude tool-calling loop). Reads marts only — never GDELT. |
| `src/agent_eval.py` | Eval suite: fixed questions, ground truth computed from the marts. |

Three layers stay structurally separate:
- **raw** (`data/raw/`) — exactly what GDELT returned, wrapped with fetch provenance. A cache
  file is never rewritten in place.
- **staging** (`data/staging/`) — typed, normalized rows.
- **mart** (`data/marts/`) — weekly aggregates joined into `trend_share`.

## Key design decisions

**Normalize by tier baseline, never raw counts.** Luxury houses get structurally more press than
affordable brands regardless of trend, so raw counts would make luxury the first mover for *every*
trend by construction. Each tier gets a brands-only baseline query; everything downstream runs on
`trend_share = trend_mentions / tier_baseline_mentions`.

**Full recompute for idempotency.** `transform.py` rebuilds staging and both marts from the entire
`data/raw/` directory every run, writing temp-file-then-rename. Nothing is ever appended, so reruns
cannot duplicate — a guarantee by construction, not by a dedup step. This is why `ingested_at` must
come from the raw layer's own `fetched_at` provenance and never from `datetime.now()`: wall-clock
stamping would make two runs over identical inputs differ and break the guarantee.

**Hashed cache keys.** A raw file is `{trend_id}__{tier}__{sha256(query|timespan)[:16]}.json`.
The hash covers the query string, so ingest is resumable by construction and a changed query
fetches fresh rather than silently reusing stale data. ⚠️ The flip side: an edit to
`brands.yaml`/`query_builder.py` **orphans** the old file, which `transform.py` still globs and
double-counts. See Known limitations.

**Fail-fast retries.** GDELT's *effective* throttling is stricter than its stated 5s floor, and
it returns rate-limit messages as plain text under HTTP 200 as often as 429 — so the client
parses and schema-validates **every** body regardless of status code. Two error types with
deliberately different handling:
- `GDELTRateLimitError` (429/5xx/rate-limit text/network) → retry with capped backoff.
- `GDELTResponseError` (non-JSON or schema mismatch, i.e. a real query bug) → **never retried**,
  raises immediately.

Per-request budget is `MAX_RETRIES=3`; the whole run stops after
`MAX_CONSECUTIVE_TASK_FAILURES=3` rather than grinding through the rest for nothing. The raw
cache is durable, so a failed task is just retried cheaply on the next `make ingest`.

**Elevated-signal detection** (not a fixed volume threshold — one viral article must never mark a
trend as emerged). Per `(trend, tier)`, on a dense weekly calendar:
- a week is **elevated** if its 3-week rolling mean share exceeds the trailing 8-week median
  share by `SIGNAL_FACTOR=2.0`;
- `first_signal_week` requires `SUSTAINED_WEEKS=2` consecutive elevated weeks;
- `peak_week` is the max-share week *among elevated weeks*, ties broken earliest;
- classification: `mass` (affordable fired) > `spreading` (semi) > `emerging` (luxury) >
  `no_signal`.

Gap weeks (trend had zero matches) fill to `share=0`; weeks where the tier's *baseline* was zero
stay `NaN` — a genuinely different case, and never an inf or a crash.

**Query agent reads marts only.** `src/agent_db.py` is the agent's sole data path and touches
local parquet/CSV only, so a question can never trigger a GDELT fetch — structural, not a rule.
SQL is validated by *parsing* it with DuckDB's own `json_serialize_sql` (one `SELECT`, nothing
else), never by regex; a second layer blocks filesystem table functions (`read_csv`, `glob`) that
would otherwise be valid SELECTs. Coverage is computed from the marts at startup and injected into
the system prompt, because an empty result set is indistinguishable from a real zero unless you
know that tier was never ingested. The agent must say "press momentum", never "best selling" —
this dataset has no sales data at all. Both behaviours are asserted in `src/agent_eval.py`.
Model is `claude-opus-5-5`; `tool_choice` stays `auto` because Opus 5.5 rejects a forced one.

## How to run

Everything goes through `.venv` — the Makefile calls `.venv/bin/python` directly, so `make`
targets work without activating.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make ingest      # 45 GDELT queries, sequential, ~5s/request floor. Resumable.
make transform   # raw -> staging -> weekly marts
make analyze     # marts -> data/output/trend_diffusion.csv
make charts      # marts -> charts/*.png
make all         # the four above, in order
make test        # pytest, synthetic fixtures only, no network
make clean       # rm staging/marts/output/charts (NOT data/raw — the cache is precious)

make agent Q="'...'"  # ask the marts a question (needs ANTHROPIC_API_KEY)
make eval             # agent eval suite (needs ANTHROPIC_API_KEY)
make eval-dry         # eval cases + ground truth, no API calls
```

`python -m src.ingest_gdelt 5` runs only the first 5 tasks — smoke test before committing to the
full rate-limited sweep.

## Current state

- **44 of 45 queries cached**, 1 missing: `le_smoking`/`affordable` (HTTP 429 after 4 attempts).
  Rerun `make ingest` to retry it; it skips what's cached.
- 13 of 14 trends have all three tiers. `le_smoking` is the only partial one.
- `transform` / `analyze` / `charts` all run clean. **47 tests pass.**
- The orphaned luxury-baseline cache file has been deleted and the marts/charts rebuilt; luxury
  `trend_share` is now correct (it was exactly half). Lag table was unaffected either way.
- ⚠️ **The README's Findings table is stale.** It was written against the earlier 7-trend subset.
  The 6 trends that filled in since do not reproduce its semi→affordable claim (across 11 trends
  with that lag: 5 negative, 6 positive). The luxury→semi negative result holds and strengthens
  (6 of 9 negative). A banner in the README says so; the prose was deliberately left as written
  rather than quietly restated. Regenerating that section is an open decision.

## Conventions

- **Contract-first.** Pydantic models at every boundary (config, API response, cache). Define the
  schema before the code that fills it.
- **Fail loudly.** Malformed config, non-JSON responses, schema mismatches raise immediately with
  the offending value in the message. Never silently coerce, never swallow, never return a
  partial result that looks complete.
- **Isolate the network.** `gdelt_client.py` is the only module that touches the wire.
  `query_builder.py` is deliberately pure so query construction is trivially testable.
- **Tests cover transform and lag logic** — the two places a silent wrong answer is plausible.
  Synthetic fixtures only, no network. Any change to signal detection or weekly aggregation needs
  a test alongside it.
- **Never commit `data/` or `charts/`** (both gitignored). Representative charts for the README
  are copied into `docs/`, which *is* tracked.
- Comments explain *why*, especially where the reason is a live-API quirk that the docs don't
  mention. Keep that habit — several non-obvious behaviors are only recorded there.

## Known limitations

- **GDELT throttling** is the main operational constraint. Effective limits are stricter than the
  stated one-request-per-5s, individual requests sometimes need minutes of retrying, and heavy
  testing once triggered what looked like an extended soft-block on one IP (a different network
  cleared it immediately). This is why 11 queries are still missing.
- **Luxury→semi lag inversion.** For 3 of 5 `mass` trends the semi-luxury signal fires *before*
  luxury's, and for 2 more luxury never fires. Read literally that contradicts the hypothesis. The
  likeliest explanation is a real limitation of a rolling same-series baseline: FW26 luxury
  coverage is already "priced in" to luxury's own trailing median, so no fresh breakout registers
  even when the trend is genuinely luxury-led. Partial-ingest gaps are a second candidate.
  Distinguishing these needs the full 45-query dataset and a second season. The semi→affordable
  lag (+8 to +36 weeks) is consistent and in the predicted direction — that's the solid finding.
- **Orphaned cache file double-counts the luxury baseline.** `data/raw/` contains two
  `_baseline__luxury__*.json` files, from before `query_builder` started quoting `Hermès`. Both
  are globbed by `transform.py`, so the luxury weekly baseline is **exactly 2× too high** and
  every luxury `trend_share` is **exactly half** its true value. Verified impact: charts only —
  luxury lines plot at half height, so cross-tier level comparison is misleading. Classifications
  and lags are unaffected, because the signal test is a ratio and scale-invariant in the
  denominator (re-running analyze without the orphan changes 0 of 14 rows). Fix by deleting
  `data/raw/_baseline__luxury__e6884784e3bb85da.json` (the older, unquoted-`Hermès` one) and
  rerunning `make transform charts`. **General hazard:** any future edit to `brands.yaml` or the
  query builder orphans cache files the same way.
- **Press attention ≠ product adoption.** A trend "reaching" affordable means affordable-brand
  press started mentioning it, not that anything shipped. The Shopify `/products.json` step in
  Next Steps is the real fix.
- **"The Row" is unresolved.** No disambiguating `search_term` exists that doesn't gut recall, so
  it stays a documented noise source in the luxury baseline.
- **Fashion-week weeks inflate luxury coverage** across the board. Baseline normalization
  mitigates but doesn't eliminate this; charts shade the FW26 window for this reason.
