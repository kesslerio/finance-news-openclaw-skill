"""Tests for summarize helpers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from datetime import datetime

import summarize
from summarize import (
    MoverContext,
    SectorCluster,
    WatchpointsData,
    build_briefing_summary,
    build_portfolio_message,
    build_watchpoints_data,
    classify_move_type,
    detect_sector_clusters,
    format_symbol_display,
    format_watchpoints,
    get_index_change,
    match_headline_to_symbol,
    validate_briefing_structure,
)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 15, 0)


class _FakeUrlopenResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_translate_via_minimax_api_parses_markdown_json(monkeypatch):
    payload = {
        "content": [
            {"type": "text", "text": "```json\n[\"Titel A\", \"Titel B\"]\n```"}
        ]
    }
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    monkeypatch.setattr(
        summarize.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeUrlopenResponse(payload),
    )

    translated, success = summarize.translate_via_minimax_api(["Title A", "Title B"], deadline=None)
    assert success is True
    assert translated == ["Titel A", "Titel B"]


def test_translate_headlines_uses_minimax_api_first(monkeypatch):
    monkeypatch.setattr(
        summarize,
        "translate_via_minimax_api",
        lambda titles, deadline: (["Titel"], True),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_agent_prompt should not be called when MiniMax API succeeds")

    monkeypatch.setattr(summarize, "run_agent_prompt", fail_if_called)

    translated, success = summarize.translate_headlines(["Title"], deadline=None)
    assert success is True
    assert translated == ["Titel"]


def test_translate_headlines_falls_back_to_openclaw(monkeypatch):
    monkeypatch.setattr(
        summarize,
        "translate_via_minimax_api",
        lambda titles, deadline: (titles, False),
    )
    monkeypatch.setattr(summarize, "run_agent_prompt", lambda *_a, **_k: "[\"Titel\"]")

    translated, success = summarize.translate_headlines(["Title"], deadline=None)
    assert success is True
    assert translated == ["Titel"]


def test_translate_via_minimax_api_missing_key(monkeypatch):
    """Returns original titles when MINIMAX_CODING_PLAN_API_KEY is not set."""
    monkeypatch.delenv("MINIMAX_CODING_PLAN_API_KEY", raising=False)
    translated, success = summarize.translate_via_minimax_api(["Title A"], deadline=None)
    assert success is False
    assert translated == ["Title A"]


def test_translate_via_minimax_api_empty_key(monkeypatch):
    """Returns original titles when MINIMAX_CODING_PLAN_API_KEY is empty/whitespace."""
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "  ")
    translated, success = summarize.translate_via_minimax_api(["Title A"], deadline=None)
    assert success is False
    assert translated == ["Title A"]


def test_translate_via_minimax_api_http_error(monkeypatch):
    """Falls back on non-retryable HTTP errors (e.g. 400)."""
    import io
    from urllib.error import HTTPError

    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    monkeypatch.setattr(summarize, "_MINIMAX_MAX_RETRIES", 0)

    def raise_400(req, timeout=0):
        raise HTTPError("https://example.com", 400, "Bad Request", {}, io.BytesIO(b""))

    monkeypatch.setattr(summarize.urllib.request, "urlopen", raise_400)
    translated, success = summarize.translate_via_minimax_api(["Title"], deadline=None)
    assert success is False
    assert translated == ["Title"]


def test_translate_via_minimax_api_timeout(monkeypatch):
    """Falls back on timeout."""
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    monkeypatch.setattr(summarize, "_MINIMAX_MAX_RETRIES", 0)

    def raise_timeout(req, timeout=0):
        raise TimeoutError("Connection timed out")

    monkeypatch.setattr(summarize.urllib.request, "urlopen", raise_timeout)
    translated, success = summarize.translate_via_minimax_api(["Title"], deadline=None)
    assert success is False
    assert translated == ["Title"]


def test_translate_via_minimax_api_malformed_json(monkeypatch):
    """Falls back when API returns non-JSON."""
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")

    class _BadResponse:
        def read(self):
            return b"not json at all"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(summarize.urllib.request, "urlopen", lambda req, timeout=0: _BadResponse())
    translated, success = summarize.translate_via_minimax_api(["Title"], deadline=None)
    assert success is False
    assert translated == ["Title"]


def test_translate_via_minimax_api_empty_content(monkeypatch):
    """Falls back when API returns empty content array."""
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    payload = {"content": []}
    monkeypatch.setattr(
        summarize.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeUrlopenResponse(payload),
    )
    translated, success = summarize.translate_via_minimax_api(["Title"], deadline=None)
    assert success is False
    assert translated == ["Title"]


def test_translate_via_minimax_api_length_mismatch(monkeypatch):
    """Falls back when API returns wrong number of translations."""
    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    payload = {
        "content": [{"type": "text", "text": '["Only One"]'}]
    }
    monkeypatch.setattr(
        summarize.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeUrlopenResponse(payload),
    )
    translated, success = summarize.translate_via_minimax_api(
        ["Title A", "Title B", "Title C"], deadline=None,
    )
    assert success is False
    assert translated == ["Title A", "Title B", "Title C"]


def test_translate_via_minimax_api_retries_on_429(monkeypatch):
    """Retries on 429 then succeeds."""
    import io
    from urllib.error import HTTPError

    monkeypatch.setenv("MINIMAX_CODING_PLAN_API_KEY", "test-key")
    monkeypatch.setattr(summarize, "_MINIMAX_MAX_RETRIES", 1)
    monkeypatch.setattr(summarize.time, "sleep", lambda s: None)  # skip wait

    call_count = [0]
    success_payload = {
        "content": [{"type": "text", "text": '["Titel"]'}]
    }

    def mock_urlopen(req, timeout=0):
        call_count[0] += 1
        if call_count[0] == 1:
            raise HTTPError("https://example.com", 429, "Rate Limited", {}, io.BytesIO(b""))
        return _FakeUrlopenResponse(success_payload)

    monkeypatch.setattr(summarize.urllib.request, "urlopen", mock_urlopen)
    translated, success = summarize.translate_via_minimax_api(["Title"], deadline=None)
    assert success is True
    assert translated == ["Titel"]
    assert call_count[0] == 2


def test_translate_via_minimax_api_empty_list():
    """Returns empty list for empty input."""
    translated, success = summarize.translate_via_minimax_api([], deadline=None)
    assert success is True
    assert translated == []


def test_build_translation_prompt():
    """Shared prompt builder produces expected format."""
    prompt = summarize._build_translation_prompt(["Headline A", "Headline B"])
    assert "2 English headlines" in prompt
    assert "1. Headline A" in prompt
    assert "2. Headline B" in prompt
    assert "JSON array" in prompt


def test_validate_briefing_structure_success():
    labels = {
        "heading_markets": "Markets",
        "heading_sentiment": "Sentiment",
        "heading_top_headlines": "Top 5 Headlines",
        "heading_portfolio_impact": "Portfolio Impact",
        "heading_watchpoints": "Watchpoints",
    }
    summary = "\n".join([
        "## Market Briefing",
        "### Markets",
        "### Sentiment: Neutral",
        "### Top 5 Headlines",
        "### Portfolio Impact",
        "### Watchpoints",
    ])
    ok, missing = validate_briefing_structure(summary, labels)
    assert ok is True
    assert missing == []


def test_validate_briefing_structure_missing_sections():
    labels = {
        "heading_markets": "Märkte",
        "heading_sentiment": "Stimmung",
        "heading_top_headlines": "Top 5 Schlagzeilen",
        "heading_portfolio_impact": "Portfolio-Auswirkung",
        "heading_watchpoints": "Beobachtungspunkte",
    }
    summary = "\n".join([
        "## Marktbriefing",
        "### Märkte",
        "### Stimmung: Bärisch",
        "### Top 5 Schlagzeilen",
    ])
    ok, missing = validate_briefing_structure(summary, labels)
    assert ok is False
    assert "portfolio_impact" in missing
    assert "watchpoints" in missing


def test_format_symbol_display_international_ticker():
    display = format_symbol_display("NOVO-B.CO", portfolio_meta={"NOVO-B.CO": {"name": "Novo Nordisk"}})
    assert display == "Novo Nordisk (NOVO-B.CO)"


def test_build_portfolio_message_renders_attribution_without_source_dump():
    portfolio_data = {
        "meta": {"total_stocks": 2},
        "stocks": {
            "NOVO-B.CO": {
                "quote": {"price": 100.0, "change_percent": -2.5, "currency": "DKK"},
                "info": {"name": "Novo Nordisk", "category": "Healthcare"},
                "articles": [
                    {
                        "title": "Why Novo Nordisk Stock Is Moving Today",
                        "link": "https://finance.yahoo.com/news/novo-stock-moving-today",
                        "source": "Yahoo Finance",
                    }
                ],
            }
        },
    }
    labels = {
        "heading_portfolio_movers": "Portfolio-Bewegungen",
        "sources_header": "Quellen",
        "portfolio_attr_benchmark": "Einordnung",
        "portfolio_attr_residual": "Restbewegung",
        "portfolio_attr_no_catalyst": "kein bestätigter Auslöser",
        "portfolio_classification_map": {"sector_theme": "sektor-/themengetrieben"},
    }
    benchmark_config = summarize.load_benchmark_config()

    output = build_portfolio_message(
        portfolio_data,
        labels,
        "de",
        benchmark_quotes={"XLV": {"change_percent": -2.0}, "ACWI": {"change_percent": -0.3}},
        benchmark_config=benchmark_config,
    )

    assert "Novo Nordisk (NOVO-B.CO)" in output
    assert "Einordnung: sektor-/themengetrieben" in output
    assert "Restbewegung -0.5%" in output
    assert "kein bestätigter Auslöser" in output
    assert "## Quellen" not in output
    assert "finance.yahoo.com" not in output


def test_build_portfolio_message_uses_non_usd_price_symbol():
    portfolio_data = {
        "stocks": {
            "6861.T": {
                "quote": {"price": 27830.0, "change_percent": -5.2},
                "info": {"name": "Keyence", "category": "Industrials"},
                "articles": [],
            }
        }
    }

    output = build_portfolio_message(
        portfolio_data,
        {"heading_portfolio_movers": "Portfolio-Bewegungen"},
        "de",
        benchmark_quotes={"EWJ": {"change_percent": -4.7}, "ACWI": {"change_percent": -0.4}},
        benchmark_config=summarize.load_benchmark_config(),
    )

    assert "¥27830.00" in output
    assert "$27830.00" not in output


def test_fetch_portfolio_benchmark_quotes_deduplicates_fetch(monkeypatch):
    portfolio_data = {
        "stocks": {
            "NVDA": {"info": {"category": "Technology"}},
            "AMD": {"info": {"category": "Technology"}},
        }
    }
    calls = []

    def fake_fetch_market_data(symbols, **_kwargs):
        calls.append(symbols)
        return {symbol: {"change_percent": 1.0} for symbol in symbols}

    monkeypatch.setattr(summarize, "fetch_market_data", fake_fetch_market_data)

    quotes = summarize.fetch_portfolio_benchmark_quotes(portfolio_data, summarize.load_benchmark_config())

    assert calls == [["SPY", "SOXX"]]
    assert quotes["SOXX"]["change_percent"] == 1.0


def test_build_briefing_summary_uses_name_for_international_mover(monkeypatch):
    labels = {
        "heading_briefing": "Marktbriefing",
        "heading_markets": "Märkte",
        "heading_sentiment": "Stimmung",
        "heading_top_headlines": "Top 5 Schlagzeilen",
        "heading_portfolio_impact": "Portfolio-Auswirkung",
        "heading_watchpoints": "Beobachtungspunkte",
        "no_data": "Keine Daten verfügbar",
        "no_movers": "Keine deutlichen Bewegungen (±1%)",
    }
    market_data = {"markets": {}}
    movers = [{"symbol": "NOVO-B.CO", "change_pct": -2.0, "price": 95.0}]
    monkeypatch.setattr(summarize, "load_portfolio_metadata", lambda: {"NOVO-B.CO": {"name": "Novo Nordisk"}})

    summary = build_briefing_summary(market_data, None, movers, [], labels, "de")
    assert "**Novo Nordisk (NOVO-B.CO)**: -2.00%" in summary


def test_generate_briefing_auto_time_evening(capsys, monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [
                {"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"},
                {"source": "Yahoo", "title": "Headline two", "link": "https://example.com/2"},
                {"source": "CNBC", "title": "Headline three", "link": "https://example.com/3"},
            ],
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}},
                    },
                }
            },
        }

    def fake_summary(*_args, **_kwargs):
        return """## Märkte

Alles ruhig.

## Stimmung

Neutral.

## Top-Themen

- Headline one
- Headline two

## Portfolio-Auswirkungen

Beobachten.

## Beobachtungspunkte

- Watch one"""

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "summarize_with_kimi", fake_summary)
    monkeypatch.setattr(summarize, "validate_briefing_structure", lambda *_a, **_k: (True, []))
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)

    args = type(
        "Args",
        (),
        {
            "lang": "de",
            "style": "briefing",
            "time": None,
            "model": "kimi",
            "json": False,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    summarize.generate_briefing(args)
    stdout = capsys.readouterr().out
    assert "Börsen Abend-Briefing" in stdout


def test_summarize_with_kimi_uses_localized_briefing_headings(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"content": [{"type": "text", "text": "### Märkte\n\nAlles ruhig."}]}).encode('utf-8')

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["json"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setattr(summarize.urllib.request, "urlopen", fake_urlopen)

    summary = summarize.summarize_with_kimi("Raw content", language="de", style="briefing", deadline=None)
    prompt = captured["json"]["messages"][0]["content"]
    assert "### Märkte" in prompt
    assert "### Sentiment" not in prompt
    assert summary.startswith("### Märkte")


def test_summarize_with_kimi_success(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        payload = {
            "content": [
                {"type": "text", "text": "## Market Briefing\n\nKimi summary body"}
            ]
        }
        return _FakeUrlopenResponse(payload)

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_API_BASE_URL", "https://api.kimi.com/coding/")
    monkeypatch.setenv("FINANCE_NEWS_KIMI_MODEL", "k2p5")
    monkeypatch.setattr(summarize.urllib.request, "urlopen", fake_urlopen)

    summary = summarize.summarize_with_kimi("Raw content", language="en", style="briefing", deadline=None)
    assert "Kimi summary body" in summary
    assert "informational purposes only" in summary
    assert captured["url"] == "https://api.kimi.com/coding/v1/messages"
    assert captured["headers"]["X-api-key"] == "test-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"


def test_summarize_with_kimi_missing_key(monkeypatch):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    summary = summarize.summarize_with_kimi("Raw content", language="en", style="briefing", deadline=None)
    assert summary == "⚠️ Kimi briefing error: KIMI_API_KEY not set"


def test_generate_briefing_zero_deadline_disables_timeout(capsys, monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [{"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"}],
            "markets": {"us": {"name": "US Markets", "indices": {"^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}}}}},
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)
    monkeypatch.setattr(
        summarize,
        "summarize_with_kimi",
        lambda *_a, **_k: (
            "### Märkte\nAlles ruhig.\n\n"
            "### Stimmung\nNeutral.\n\n"
            "### Top 5 Schlagzeilen\n1. Headline one\n\n"
            "### Portfolio-Auswirkung\nKeine.\n\n"
            "### Beobachtungspunkte\n- Watch one"
        ),
    )

    args = type(
        "Args",
        (),
        {
            "lang": "de",
            "style": "briefing",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": 0.0,
            "fast": False,
            "llm": True,
            "debug": False,
        },
    )()

    summarize.generate_briefing(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary_mode"] == "llm"
    assert payload["summary_model_used"] == "kimi"


def test_generate_briefing_hard_fails_when_kimi_key_missing(monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [
                {"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"},
                {"source": "Yahoo", "title": "Headline two", "link": "https://example.com/2"},
                {"source": "CNBC", "title": "Headline three", "link": "https://example.com/3"},
            ],
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}},
                    },
                }
            },
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    args = type(
        "Args",
        (),
        {
            "lang": "en",
            "style": "briefing",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    import pytest
    with pytest.raises(RuntimeError, match="KIMI_API_KEY not set"):
        summarize.generate_briefing(args)


def test_generate_analysis_hard_fails_when_kimi_unavailable(monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [
                {"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"},
            ],
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}},
                    },
                }
            },
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)
    monkeypatch.setattr(summarize, "summarize_with_kimi", lambda *_a, **_k: "⚠️ Kimi briefing error: KIMI_API_KEY not set")

    args = type(
        "Args",
        (),
        {
            "lang": "en",
            "style": "analysis",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    import pytest
    with pytest.raises(RuntimeError, match="KIMI_API_KEY not set"):
        summarize.generate_briefing(args)


def test_generate_analysis_uses_kimi_even_without_llm_flag(capsys, monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [
                {"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"},
                {"source": "Yahoo", "title": "Headline two", "link": "https://example.com/2"},
            ],
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}},
                    },
                }
            },
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)

    calls = {}

    def fake_kimi(content, language, style, deadline=None):
        calls["style"] = style
        return "## Analysis\n\nKimi analysis body"

    monkeypatch.setattr(summarize, "summarize_with_kimi", fake_kimi)

    args = type(
        "Args",
        (),
        {
            "lang": "en",
            "style": "analysis",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    summarize.generate_briefing(args)
    payload = json.loads(capsys.readouterr().out)
    assert calls["style"] == "analysis"
    assert payload["summary_mode"] == "llm"
    assert payload["summary_model_used"] == "kimi"
    assert "Kimi analysis body" in payload["summary"]


# --- Tests for watchpoints feature (Issue #92) ---


class TestGetIndexChange:
    def test_extracts_sp500_change(self):
        market_data = {
            "markets": {
                "us": {
                    "indices": {
                        "^GSPC": {"data": {"change_percent": -1.5}}
                    }
                }
            }
        }
        assert get_index_change(market_data) == -1.5

    def test_returns_zero_on_missing_data(self):
        assert get_index_change({}) == 0.0
        assert get_index_change({"markets": {}}) == 0.0
        assert get_index_change({"markets": {"us": {}}}) == 0.0


class TestMatchHeadlineToSymbol:
    def test_exact_symbol_match_dollar(self):
        headlines = [{"title": "Breaking: $NVDA surges on AI demand"}]
        result = match_headline_to_symbol("NVDA", "NVIDIA Corporation", headlines)
        assert result is not None
        assert "NVDA" in result["title"]

    def test_exact_symbol_match_parens(self):
        headlines = [{"title": "Tesla (TSLA) reports record deliveries"}]
        result = match_headline_to_symbol("TSLA", "Tesla Inc", headlines)
        assert result is not None

    def test_exact_symbol_match_word_boundary(self):
        headlines = [{"title": "AAPL announces new product line"}]
        result = match_headline_to_symbol("AAPL", "Apple Inc", headlines)
        assert result is not None

    def test_company_name_match(self):
        headlines = [{"title": "Apple announces record iPhone sales"}]
        result = match_headline_to_symbol("AAPL", "Apple Inc", headlines)
        assert result is not None

    def test_no_match_returns_none(self):
        headlines = [{"title": "Fed raises interest rates"}]
        result = match_headline_to_symbol("NVDA", "NVIDIA Corporation", headlines)
        assert result is None

    def test_avoids_partial_symbol_match(self):
        # "APP" should not match "application"
        headlines = [{"title": "New application launches today"}]
        result = match_headline_to_symbol("APP", "AppLovin Corp", headlines)
        assert result is None

    def test_empty_headlines(self):
        result = match_headline_to_symbol("NVDA", "NVIDIA", [])
        assert result is None


class TestDetectSectorClusters:
    def test_detects_cluster_three_stocks_same_direction(self):
        movers = [
            {"symbol": "NVDA", "change_pct": -5.0},
            {"symbol": "AMD", "change_pct": -4.0},
            {"symbol": "INTC", "change_pct": -3.0},
        ]
        portfolio_meta = {
            "NVDA": {"category": "Tech"},
            "AMD": {"category": "Tech"},
            "INTC": {"category": "Tech"},
        }
        clusters = detect_sector_clusters(movers, portfolio_meta)
        assert len(clusters) == 1
        assert clusters[0].category == "Tech"
        assert clusters[0].direction == "down"
        assert len(clusters[0].stocks) == 3

    def test_no_cluster_if_less_than_three(self):
        movers = [
            {"symbol": "NVDA", "change_pct": -5.0},
            {"symbol": "AMD", "change_pct": -4.0},
        ]
        portfolio_meta = {
            "NVDA": {"category": "Tech"},
            "AMD": {"category": "Tech"},
        }
        clusters = detect_sector_clusters(movers, portfolio_meta)
        assert len(clusters) == 0

    def test_no_cluster_if_mixed_direction(self):
        movers = [
            {"symbol": "NVDA", "change_pct": 5.0},
            {"symbol": "AMD", "change_pct": -4.0},
            {"symbol": "INTC", "change_pct": 3.0},
        ]
        portfolio_meta = {
            "NVDA": {"category": "Tech"},
            "AMD": {"category": "Tech"},
            "INTC": {"category": "Tech"},
        }
        clusters = detect_sector_clusters(movers, portfolio_meta)
        assert len(clusters) == 0


class TestClassifyMoveType:
    def test_earnings_with_keyword(self):
        headline = {"title": "Company beats Q3 earnings expectations"}
        result = classify_move_type(headline, False, 5.0, 0.1)
        assert result == "earnings"

    def test_sector_cluster(self):
        result = classify_move_type(None, True, -3.0, -0.5)
        assert result == "sector"

    def test_market_wide(self):
        result = classify_move_type(None, False, -2.0, -2.0)
        assert result == "market_wide"

    def test_company_specific_with_headline(self):
        headline = {"title": "Company announces acquisition"}
        result = classify_move_type(headline, False, 3.0, 0.1)
        assert result == "company_specific"

    def test_company_specific_large_move_no_headline(self):
        result = classify_move_type(None, False, 8.0, 0.1)
        assert result == "company_specific"

    def test_unknown_small_move_no_context(self):
        result = classify_move_type(None, False, 1.5, 0.2)
        assert result == "unknown"


class TestFormatWatchpoints:
    def test_formats_sector_cluster(self):
        cluster = SectorCluster(
            category="Tech",
            stocks=[
                MoverContext("NVDA", -5.0, 100.0, "Tech", None, "sector", None),
                MoverContext("AMD", -4.0, 80.0, "Tech", None, "sector", None),
                MoverContext("INTC", -3.0, 30.0, "Tech", None, "sector", None),
            ],
            avg_change=-4.0,
            direction="down",
            vs_index=-3.5,
        )
        data = WatchpointsData(
            movers=[],
            sector_clusters=[cluster],
            index_change=-0.5,
            market_wide=False,
        )
        result = format_watchpoints(data, "en", {})
        assert "Legend" in result
        assert "Sector Rotation" in result
        assert "Tech" in result
        assert "-4.0%" in result
        assert "vs Index" in result

    def test_formats_individual_mover_with_headline(self):
        mover = MoverContext(
            symbol="NVDA",
            change_pct=5.0,
            price=100.0,
            category="Tech",
            matched_headline={"title": "NVIDIA reports record revenue"},
            move_type="company_specific",
            vs_index=4.5,
        )
        data = WatchpointsData(
            movers=[mover],
            sector_clusters=[],
            index_change=0.5,
            market_wide=False,
        )
        result = format_watchpoints(data, "en", {})
        assert "Single-Name Moves" in result
        assert "NVDA" in result
        assert "+5.0%" in result
        assert "record revenue" in result

    def test_truncates_long_headline_context(self):
        long_title = (
            "NVIDIA gives unusually detailed long-range guidance for AI datacenter demand "
            "into fiscal year twenty twenty-seven"
        )
        mover = MoverContext(
            symbol="NVDA",
            change_pct=6.2,
            price=100.0,
            category="Tech",
            matched_headline={"title": long_title, "source": "WSJ"},
            move_type="company_specific",
            vs_index=5.9,
        )
        data = WatchpointsData(
            movers=[mover],
            sector_clusters=[],
            index_change=0.3,
            market_wide=False,
        )
        result = format_watchpoints(data, "en", {})
        expected_truncated = f"{long_title[:60]}..."
        assert expected_truncated in result
        assert long_title not in result

    def test_formats_market_wide_move_english(self):
        data = WatchpointsData(
            movers=[],
            sector_clusters=[],
            index_change=-2.0,
            market_wide=True,
        )
        result = format_watchpoints(data, "en", {})
        assert "Market-wide move" in result
        assert "S&P 500 fell 2.0%" in result

    def test_formats_market_wide_move_german(self):
        data = WatchpointsData(
            movers=[],
            sector_clusters=[],
            index_change=2.5,
            market_wide=True,
        )
        result = format_watchpoints(data, "de", {})
        assert "Breite Marktbewegung" in result
        assert "stieg 2.5%" in result

    def test_uses_label_fallbacks(self):
        mover = MoverContext(
            symbol="XYZ",
            change_pct=1.5,
            price=50.0,
            category="Other",
            matched_headline=None,
            move_type="unknown",
            vs_index=0.2,
        )
        data = WatchpointsData(
            movers=[mover],
            sector_clusters=[],
            index_change=0.5,
            market_wide=False,
        )
        labels = {"likely_sector_contagion": " -- no news"}
        result = format_watchpoints(data, "en", labels)
        assert "XYZ" in result
        assert "no news" in result


class TestBuildWatchpointsData:
    def test_builds_complete_data_structure(self):
        movers = [
            {"symbol": "NVDA", "change_pct": -5.0, "price": 100.0},
            {"symbol": "AMD", "change_pct": -4.0, "price": 80.0},
            {"symbol": "INTC", "change_pct": -3.0, "price": 30.0},
            {"symbol": "AAPL", "change_pct": 2.0, "price": 150.0},
        ]
        headlines = [
            {"title": "NVIDIA reports weak guidance"},
            {"title": "Apple announces new product"},
        ]
        portfolio_meta = {
            "NVDA": {"category": "Tech", "name": "NVIDIA Corporation"},
            "AMD": {"category": "Tech", "name": "Advanced Micro Devices"},
            "INTC": {"category": "Tech", "name": "Intel Corporation"},
            "AAPL": {"category": "Tech", "name": "Apple Inc"},
        }
        index_change = -0.5

        result = build_watchpoints_data(movers, headlines, portfolio_meta, index_change)

        # Should detect Tech sector cluster (3 losers)
        assert len(result.sector_clusters) == 1
        assert result.sector_clusters[0].category == "Tech"
        assert result.sector_clusters[0].direction == "down"

        # All movers should be present
        assert len(result.movers) == 4

        # NVDA should have matched headline
        nvda_mover = next(m for m in result.movers if m.symbol == "NVDA")
        assert nvda_mover.matched_headline is not None
        assert "guidance" in nvda_mover.matched_headline["title"]

        # vs_index should be calculated
        assert nvda_mover.vs_index == -5.0 - (-0.5)  # -4.5

    def test_handles_empty_movers(self):
        result = build_watchpoints_data([], [], {}, 0.0)
        assert result.movers == []
        assert result.sector_clusters == []
        assert result.market_wide is False

    def test_detects_market_wide_move(self):
        result = build_watchpoints_data([], [], {}, -2.0)
        assert result.market_wide is True


def test_generate_briefing_defaults_to_kimi_path(capsys, monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [{"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"}],
            "markets": {"us": {"name": "US Markets", "indices": {"^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}}}}},
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)
    monkeypatch.setattr(summarize, "validate_briefing_structure", lambda *_a, **_k: (True, []))
    monkeypatch.setattr(summarize, "summarize_with_kimi", lambda *_a, **_k: "### Märkte\n\nAlles ruhig.\n\n### Stimmung\n\nNeutral.\n\n### Top 5 Schlagzeilen\n1. Headline one\n\n### Portfolio-Auswirkung\nBeobachten.\n\n### Beobachtungspunkte\n- Watch one")

    args = type(
        "Args",
        (),
        {
            "lang": "de",
            "style": "briefing",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    summarize.generate_briefing(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary_mode"] == "llm"
    assert payload["summary_model_used"] == "kimi"


def test_generate_analysis_hard_fails_without_non_kimi_fallback(monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [{"source": "CNBC", "title": "Headline one", "link": "https://example.com/1"}],
            "markets": {"us": {"name": "US Markets", "indices": {"^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}}}}},
        }

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "get_portfolio_movers", lambda *_a, **_k: {"movers": []})
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)
    monkeypatch.setattr(summarize, "summarize_with_kimi", lambda *_a, **_k: "⚠️ Kimi briefing error: KIMI_API_KEY not set")

    args = type(
        "Args",
        (),
        {
            "lang": "en",
            "style": "analysis",
            "time": None,
            "model": "kimi",
            "json": True,
            "research": False,
            "deadline": None,
            "fast": False,
            "llm": False,
            "debug": False,
        },
    )()

    import pytest
    with pytest.raises(RuntimeError, match="KIMI_API_KEY not set"):
        summarize.generate_briefing(args)
