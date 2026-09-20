from src.config_models import Brand, Trend, TrendEvidence
from src.query_builder import build_baseline_query, build_trend_query, format_term, or_block


def test_format_term_bare_when_single_word():
    assert format_term("suede") == "suede"


def test_format_term_quoted_when_multi_word():
    assert format_term("faux fur") == '"faux fur"'


def test_format_term_quoted_when_special_character():
    # Live API: a bare "H&M" was rejected with "illegal character"; GDELT's
    # own error says to quote it (its example: a dash, "f-16").
    assert format_term("H&M") == '"H&M"'
    assert format_term("Abercrombie & Fitch") == '"Abercrombie & Fitch"'


def test_or_block_single_term_has_no_parens():
    # GDELT: "Parentheses may only be used around OR'd statements" -- a lone
    # term must never be wrapped, confirmed against the live API.
    assert or_block(["suede"]) == "suede"


def test_or_block_multi_term_is_parenthesized_and_quoted():
    assert or_block(["purple", "violet", "plum"]) == "(purple OR violet OR plum)"
    assert or_block(["Chanel", "Ralph Lauren"]) == '(Chanel OR "Ralph Lauren")'


def test_or_block_empty_is_empty_string():
    assert or_block([]) == ""


def _trend(query_terms, context=None):
    return Trend(
        id="t",
        label="T",
        query_terms=query_terms,
        context=context,
        evidence=TrendEvidence(brands=["Chanel"], source="https://example.com"),
    )


def test_build_trend_query_no_context_single_term():
    tier_brands = [Brand(name="Chanel", search_term="Chanel"), Brand(name="Prada", search_term="Prada")]
    query = build_trend_query(_trend(["suede"]), tier_brands)
    assert query == "suede (Chanel OR Prada)"


def test_build_trend_query_with_context_and_multi_terms():
    tier_brands = [Brand(name="Chanel", search_term="Chanel"), Brand(name="Prada", search_term="Prada")]
    trend = _trend(["purple", "violet", "plum"], context=["dress", "coat"])
    query = build_trend_query(trend, tier_brands)
    assert query == "(purple OR violet OR plum) (dress OR coat) (Chanel OR Prada)"


def test_build_trend_query_uses_search_term_not_name():
    # This is the whole point of the brand-name fix: ambiguous brand names
    # (Theory, COS, Mango, Reformation) query on a disambiguated search_term.
    tier_brands = [Brand(name="Theory", search_term="Theory clothing")]
    query = build_trend_query(_trend(["velvet"]), tier_brands)
    assert query == 'velvet "Theory clothing"'


def test_build_baseline_query_is_brands_only():
    tier_brands = [Brand(name="Chanel", search_term="Chanel"), Brand(name="Prada", search_term="Prada")]
    assert build_baseline_query(tier_brands) == "(Chanel OR Prada)"
