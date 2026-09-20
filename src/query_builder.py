"""Pure GDELT query-string construction. No network, no I/O -- kept separate
so it's trivially unit-testable.

GDELT quirks learned by hitting the live API directly:
  - Parentheses are only valid around an actual OR'd group of 2+ terms. A
    single term must be bare.
  - A bare term with a "special" character (confirmed: "&", e.g. "H&M") is
    rejected as an "illegal character" -- GDELT's own error message says to
    quote it instead (its example is a dash: "f-16"), so any term that isn't
    plain alphanumeric gets quoted, not just multi-word phrases.
"""
import re
from typing import List, Optional

from src.config_models import Brand, Trend

_PLAIN_ALNUM = re.compile(r"^[A-Za-z0-9]+$")


def format_term(term: str) -> str:
    return term if _PLAIN_ALNUM.match(term) else f'"{term}"'


def or_block(terms: List[str]) -> str:
    if not terms:
        return ""
    formatted = [format_term(t) for t in terms]
    if len(formatted) == 1:
        return formatted[0]
    return "(" + " OR ".join(formatted) + ")"


def build_trend_query(trend: Trend, tier_brands: List[Brand]) -> str:
    blocks: List[Optional[str]] = [
        or_block(trend.query_terms),
        or_block(trend.context) if trend.context else None,
        or_block([b.search_term for b in tier_brands]),
    ]
    return " ".join(b for b in blocks if b)


def build_baseline_query(tier_brands: List[Brand]) -> str:
    return or_block([b.search_term for b in tier_brands])
