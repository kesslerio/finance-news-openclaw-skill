"""Tests for research.py - deep research module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import research
from research import (
    format_headlines,
    format_market_data,
    format_portfolio_news,
    format_raw_data_report,
    generate_research_content,
    research_with_ds4,
    research_with_qwen,
)


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "markets": {
            "us": {
                "name": "US Markets",
                "indices": {
                    "SPY": {
                        "name": "S&P 500",
                        "data": {"price": 5200.50, "change_percent": 1.25}
                    },
                    "QQQ": {
                        "name": "Nasdaq 100",
                        "data": {"price": 18500.00, "change_percent": -0.50}
                    }
                }
            },
            "europe": {
                "name": "European Markets",
                "indices": {
                    "DAX": {
                        "name": "DAX",
                        "data": {"price": 18200.00, "change_percent": 0.75}
                    }
                }
            }
        },
        "headlines": [
            {"source": "Reuters", "title": "Fed holds rates steady", "link": "https://example.com/1"},
            {"source": "Bloomberg", "title": "Tech stocks rally", "link": "https://example.com/2"},
        ]
    }


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        "stocks": {
            "AAPL": {
                "quote": {"price": 185.50, "change_percent": 2.3},
                "articles": [
                    {"title": "Apple reports strong earnings", "link": "https://example.com/aapl1"},
                    {"title": "iPhone sales beat expectations", "link": "https://example.com/aapl2"},
                ]
            },
            "MSFT": {
                "quote": {"price": 420.00, "change_percent": -1.1},
                "articles": [
                    {"title": "Microsoft cloud growth slows", "link": "https://example.com/msft1"},
                ]
            }
        }
    }


class TestFormatMarketData:
    """Tests for format_market_data()."""

    def test_formats_market_indices(self, sample_market_data):
        """Format market indices with prices and changes."""
        result = format_market_data(sample_market_data)

        assert "## Market Data" in result
        assert "### US Markets" in result
        assert "S&P 500" in result
        assert "5200.5" in result  # Price (may not have trailing zero)
        assert "+1.25%" in result
        assert "📈" in result  # Positive change

    def test_shows_negative_change_emoji(self, sample_market_data):
        """Negative changes show down emoji."""
        result = format_market_data(sample_market_data)

        assert "Nasdaq 100" in result
        assert "-0.50%" in result
        assert "📉" in result  # Negative change

    def test_handles_empty_data(self):
        """Handle empty market data."""
        result = format_market_data({})
        assert "## Market Data" in result
        assert "### " not in result  # No region headers

    def test_handles_missing_index_data(self):
        """Handle indices without data."""
        data = {
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "SPY": {"name": "S&P 500"}  # No 'data' key
                    }
                }
            }
        }
        result = format_market_data(data)
        assert "## Market Data" in result
        # Should not crash, just skip the index


class TestFormatHeadlines:
    """Tests for format_headlines()."""

    def test_formats_headlines_with_links(self):
        """Format headlines with sources and links."""
        headlines = [
            {"source": "Reuters", "title": "Breaking news", "link": "https://example.com/1"},
            {"source": "Bloomberg", "title": "Market update", "link": "https://example.com/2"},
        ]
        result = format_headlines(headlines)

        assert "## Current Headlines" in result
        assert "[Reuters] Breaking news" in result
        assert "URL: https://example.com/1" in result
        assert "[Bloomberg] Market update" in result

    def test_handles_missing_source(self):
        """Handle headlines with missing source."""
        headlines = [{"title": "No source headline", "link": "https://example.com"}]
        result = format_headlines(headlines)

        assert "[Unknown] No source headline" in result

    def test_handles_missing_link(self):
        """Handle headlines without links."""
        headlines = [{"source": "Reuters", "title": "No link"}]
        result = format_headlines(headlines)

        assert "[Reuters] No link" in result
        assert "URL:" not in result

    def test_limits_to_20_headlines(self):
        """Limit output to 20 headlines max."""
        headlines = [{"source": f"Source{i}", "title": f"Title {i}"} for i in range(30)]
        result = format_headlines(headlines)

        assert "[Source19]" in result
        assert "[Source20]" not in result

    def test_handles_empty_list(self):
        """Handle empty headlines list."""
        result = format_headlines([])
        assert "## Current Headlines" in result


class TestFormatPortfolioNews:
    """Tests for format_portfolio_news()."""

    def test_formats_portfolio_stocks(self, sample_portfolio_data):
        """Format portfolio stocks with quotes and news."""
        result = format_portfolio_news(sample_portfolio_data)

        assert "## Portfolio Analysis" in result
        assert "### AAPL" in result
        assert "$185.5" in result  # Price (may not have trailing zero)
        assert "+2.30%" in result
        assert "Apple reports strong earnings" in result

    def test_shows_negative_changes(self, sample_portfolio_data):
        """Show negative change percentages."""
        result = format_portfolio_news(sample_portfolio_data)

        assert "### MSFT" in result
        assert "-1.10%" in result

    def test_limits_articles_to_5(self):
        """Limit articles per stock to 5."""
        data = {
            "stocks": {
                "AAPL": {
                    "quote": {"price": 185.0, "change_percent": 1.0},
                    "articles": [{"title": f"Article {i}"} for i in range(10)]
                }
            }
        }
        result = format_portfolio_news(data)

        assert "Article 4" in result
        assert "Article 5" not in result

    def test_handles_empty_stocks(self):
        """Handle empty stocks dict."""
        result = format_portfolio_news({"stocks": {}})
        assert "## Portfolio Analysis" in result


class TestResearchWithDS4:
    """Tests for research_with_ds4() - the DS4-high primary route."""

    def test_successful_research(self, monkeypatch):
        """Execute DS4 research successfully against the local route."""
        captured = {}

        def fake_call(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return "# Research Report\n\nMarket analysis..."

        monkeypatch.setenv("FINANCE_NEWS_DS4_BASE_URL", "http://ds4.test/v1")
        monkeypatch.setenv("FINANCE_NEWS_DS4_MODEL", "ds4-test")
        monkeypatch.setattr(research, "call_openai_chat", fake_call)

        result = research_with_ds4("Market data content")

        assert result == "# Research Report\n\nMarket analysis..."
        assert captured["kwargs"]["base_url"] == "http://ds4.test/v1"
        assert captured["kwargs"]["model"] == "ds4-test"

    def test_research_with_focus_areas(self, monkeypatch):
        """Include focus areas in prompt."""
        captured = {}

        def fake_call(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Focused analysis"

        monkeypatch.setattr(research, "call_openai_chat", fake_call)

        result = research_with_ds4("content", focus_areas=["earnings", "macro"])

        assert result == "Focused analysis"
        assert "earnings" in captured["prompt"]
        assert "macro" in captured["prompt"]

    def test_handles_route_error(self, monkeypatch):
        """Handle DS4 route error gracefully (sentinel passthrough)."""
        monkeypatch.setattr(
            research,
            "call_openai_chat",
            lambda *_a, **_k: "⚠️ DS4 research error: HTTP 503: unavailable",
        )
        result = research_with_ds4("content")

        assert "⚠️ DS4 research error" in result


class TestResearchWithQwen:
    """Tests for research_with_qwen() - the Qwen fallback route."""

    def test_uses_qwen_route(self, monkeypatch):
        """Fallback calls the local Qwen route with the serving key."""
        captured = {}

        def fake_call(prompt, **kwargs):
            captured["kwargs"] = kwargs
            return "Qwen report"

        monkeypatch.setenv("KALLIOPE_SERVING_API_KEY", "serving-key")
        monkeypatch.setenv("FINANCE_NEWS_QWEN_BASE_URL", "http://qwen.test/v1")
        monkeypatch.setenv("FINANCE_NEWS_QWEN_MODEL", "qwen-test")
        monkeypatch.setattr(research, "call_openai_chat", fake_call)

        assert research_with_qwen("content") == "Qwen report"
        assert captured["kwargs"]["base_url"] == "http://qwen.test/v1"
        assert captured["kwargs"]["model"] == "qwen-test"
        assert captured["kwargs"]["api_key"] == "serving-key"

    def test_requires_serving_key(self, monkeypatch):
        """Fail closed when the serving key is missing."""
        monkeypatch.delenv("KALLIOPE_SERVING_API_KEY", raising=False)
        result = research_with_qwen("content")

        assert result == "⚠️ Qwen research error: KALLIOPE_SERVING_API_KEY not set"


class TestFormatRawDataReport:
    """Tests for format_raw_data_report()."""

    def test_combines_market_and_portfolio(self, sample_market_data, sample_portfolio_data):
        """Combine market data, headlines, and portfolio."""
        result = format_raw_data_report(sample_market_data, sample_portfolio_data)

        assert "## Market Data" in result
        assert "## Current Headlines" in result
        assert "## Portfolio Analysis" in result

    def test_handles_no_headlines(self, sample_portfolio_data):
        """Handle market data without headlines."""
        market_data = {"markets": {"us": {"name": "US", "indices": {}}}}
        result = format_raw_data_report(market_data, sample_portfolio_data)

        assert "## Market Data" in result
        assert "## Current Headlines" not in result

    def test_handles_portfolio_error(self, sample_market_data):
        """Skip portfolio with error."""
        portfolio_data = {"error": "No portfolio configured"}
        result = format_raw_data_report(sample_market_data, portfolio_data)

        assert "## Portfolio Analysis" not in result

    def test_handles_empty_data(self):
        """Handle empty market and portfolio data."""
        result = format_raw_data_report({}, {})
        assert result == ""


class TestGenerateResearchContent:
    """Tests for generate_research_content()."""

    def test_uses_ds4_primary(self, sample_market_data, sample_portfolio_data):
        """Use the DS4-high route as the deep-research primary."""
        with patch("research.research_with_ds4", return_value="DS4 report") as mock_ds4:
            result = generate_research_content(sample_market_data, sample_portfolio_data)

            assert result["report"] == "DS4 report"
            assert result["source"] == "ds4"
            mock_ds4.assert_called_once()

    def test_falls_back_to_qwen_when_ds4_fails(self, sample_market_data, sample_portfolio_data):
        """Use the Qwen route when the DS4 route fails."""
        with patch("research.research_with_ds4", return_value="⚠️ DS4 research error: timeout"):
            with patch("research.research_with_qwen", return_value="Qwen report") as mock_qwen:
                result = generate_research_content(sample_market_data, sample_portfolio_data)

                assert result["report"] == "Qwen report"
                assert result["source"] == "qwen"
                mock_qwen.assert_called_once()

    def test_falls_back_to_raw_report(self, sample_market_data, sample_portfolio_data):
        """Fall back to raw report when both local routes fail."""
        with patch("research.research_with_ds4", return_value="⚠️ DS4 research error: timeout"):
            with patch("research.research_with_qwen", return_value="⚠️ Qwen research error: timeout"):
                result = generate_research_content(sample_market_data, sample_portfolio_data)

            assert "## Market Data" in result["report"]
            assert result["source"] == "raw"

    def test_handles_empty_report(self):
        """Return empty when no data available."""
        result = generate_research_content({}, {})

        assert result["report"] == ""
        assert result["source"] == "none"

    def test_passes_focus_areas_to_ds4(self, sample_market_data, sample_portfolio_data):
        """Pass focus areas to DS4 research."""
        focus = ["earnings", "tech"]
        with patch("research.research_with_ds4", return_value="Report") as mock_ds4:
            generate_research_content(sample_market_data, sample_portfolio_data, focus_areas=focus)

            mock_ds4.assert_called_once()
            # Focus areas passed as second positional arg
            call_args = mock_ds4.call_args
            assert call_args[0][1] == focus or call_args.kwargs.get("focus_areas") == focus
