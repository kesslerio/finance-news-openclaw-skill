"""Tests for research.py - deep research module."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from research import (
    format_headlines,
    format_market_data,
    format_portfolio_news,
    format_raw_data_report,
    gemini_available,
    generate_research_content,
    minimax_prompt_available,
    research_with_gemini,
    research_with_minimax,
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


class TestMiniMaxAvailable:
    """Tests for minimax_prompt_available()."""

    def test_returns_true_when_minimax_found(self):
        """Return True when minimax-prompt CLI is found."""
        with patch("shutil.which", return_value="/usr/local/bin/minimax-prompt"):
            assert minimax_prompt_available() is True

    def test_returns_false_when_minimax_not_found(self):
        """Return False when minimax-prompt CLI is not found."""
        with patch("shutil.which", return_value=None):
            assert minimax_prompt_available() is False

    def test_gemini_available_aliases_minimax(self):
        """Legacy availability helper remains an alias for MiniMax."""
        with patch("shutil.which", return_value="/usr/local/bin/minimax-prompt"):
            assert gemini_available() is True


class TestResearchWithMiniMax:
    """Tests for research_with_minimax()."""

    def test_successful_research(self):
        """Execute MiniMax research successfully."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "# Research Report\n\nMarket analysis..."

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = research_with_minimax("Market data content")

            assert result == "# Research Report\n\nMarket analysis..."
            mock_run.assert_called_once()

    def test_research_with_focus_areas(self):
        """Include focus areas in prompt."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Focused analysis"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = research_with_minimax("content", focus_areas=["earnings", "macro"])

            assert result == "Focused analysis"
            # Verify focus areas were in the prompt
            call_args = mock_run.call_args[0][0]
            prompt = call_args[1]
            assert "earnings" in prompt
            assert "macro" in prompt

    def test_handles_minimax_error(self):
        """Handle MiniMax error gracefully."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "API error"

        with patch("subprocess.run", return_value=mock_result):
            result = research_with_minimax("content")

            assert "⚠️ MiniMax research error" in result
            assert "API error" in result

    def test_handles_timeout(self):
        """Handle subprocess timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="minimax-prompt", timeout=120)):
            result = research_with_minimax("content")

            assert "⚠️ MiniMax research timeout" in result

    def test_handles_missing_minimax(self):
        """Handle missing minimax-prompt CLI."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = research_with_minimax("content")

            assert "⚠️ minimax-prompt not found" in result

    def test_research_with_gemini_aliases_minimax(self):
        """Legacy helper remains an alias for MiniMax callers."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Alias report"

        with patch("subprocess.run", return_value=mock_result):
            assert research_with_gemini("content") == "Alias report"


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

    def test_uses_minimax_when_available(self, sample_market_data, sample_portfolio_data):
        """Use MiniMax research when available."""
        with patch("research.minimax_prompt_available", return_value=True):
            with patch("research.research_with_minimax", return_value="MiniMax report") as mock_minimax:
                result = generate_research_content(sample_market_data, sample_portfolio_data)

                assert result["report"] == "MiniMax report"
                assert result["source"] == "minimax"
                mock_minimax.assert_called_once()

    def test_falls_back_to_raw_report(self, sample_market_data, sample_portfolio_data):
        """Fall back to raw report when MiniMax unavailable."""
        with patch("research.minimax_prompt_available", return_value=False):
            result = generate_research_content(sample_market_data, sample_portfolio_data)

            assert "## Market Data" in result["report"]
            assert result["source"] == "raw"

    def test_handles_empty_report(self):
        """Return empty when no data available."""
        result = generate_research_content({}, {})

        assert result["report"] == ""
        assert result["source"] == "none"

    def test_passes_focus_areas_to_minimax(self, sample_market_data, sample_portfolio_data):
        """Pass focus areas to MiniMax research."""
        focus = ["earnings", "tech"]
        with patch("research.minimax_prompt_available", return_value=True):
            with patch("research.research_with_minimax", return_value="Report") as mock_minimax:
                generate_research_content(sample_market_data, sample_portfolio_data, focus_areas=focus)

                mock_minimax.assert_called_once()
                # Check that focus_areas was passed (positional or keyword)
                call_args = mock_minimax.call_args
                # Focus areas passed as second positional arg
                assert call_args[0][1] == focus or call_args.kwargs.get("focus_areas") == focus
