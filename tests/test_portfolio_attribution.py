"""Tests for portfolio attribution helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from portfolio_attribution import (
    benchmark_tickers_for_portfolio,
    build_attributions,
    evidence_for_articles,
    evaluate_article_evidence,
    load_benchmark_config,
    resolve_benchmark_mapping,
)


def test_load_benchmark_config_has_global_defaults():
    config = load_benchmark_config()

    assert config["broad_markets"]["default"]["ticker"] == "ACWI"
    assert config["categories"]["Technology"]["ticker"] == "XLK"
    assert "finance.yahoo.com" in config["source_quality"]["allowed_domains"]


def test_ticker_override_wins_over_category_mapping():
    config = load_benchmark_config()

    mapping = resolve_benchmark_mapping(
        "NVDA",
        {"category": "Technology", "name": "NVIDIA"},
        config,
    )

    assert mapping.benchmark_ticker == "SOXX"
    assert mapping.reason == "ticker_override"


def test_theme_mapping_wins_when_category_is_unknown():
    config = load_benchmark_config()

    mapping = resolve_benchmark_mapping(
        "ANET",
        {"category": "Unknown", "investment_theme": "Cloud|Data Infrastructure"},
        config,
    )

    assert mapping.benchmark_ticker == "XLK"
    assert mapping.reason == "theme"
    assert mapping.uncertain is False


def test_us_class_share_ticker_uses_us_market_benchmark():
    config = load_benchmark_config()

    mapping = resolve_benchmark_mapping(
        "BRK.B",
        {"category": "Unknown", "name": "Berkshire Hathaway"},
        config,
    )

    assert mapping.market_ticker == "SPY"
    assert mapping.market_label == "S&P 500"


def test_benchmark_tickers_for_portfolio_deduplicates_market_and_sector():
    config = load_benchmark_config()
    stocks = {
        "NVDA": {"info": {"category": "Technology"}},
        "AMD": {"info": {"category": "Technology"}},
        "6861.T": {"info": {"category": "Industrials"}},
    }

    tickers = benchmark_tickers_for_portfolio(stocks, config)

    assert tickers.count("SOXX") == 1
    assert "SPY" in tickers
    assert "EWJ" in tickers


def test_sector_driven_japan_mover_without_catalyst():
    config = load_benchmark_config()
    stocks = {
        "6861.T": {
            "quote": {"price": 27830.0, "change_percent": -5.2},
            "info": {"category": "Industrials", "name": "Keyence"},
            "articles": [],
        }
    }
    benchmark_quotes = {
        "ACWI": {"change_percent": -0.4},
        "EWJ": {"change_percent": -4.7},
        "JPY=X": {"change_percent": 0.2},
    }

    result = build_attributions(stocks, benchmark_quotes, config)[0]

    assert result.classification == "sector_theme"
    assert result.residual_pct == -0.5
    assert result.evidence is None
    assert result.mapping_uncertain is False


def test_semiconductor_residual_move_preserves_credible_catalyst():
    config = load_benchmark_config()
    stocks = {
        "NVDA": {
            "quote": {"price": 120.0, "change_percent": -6.0},
            "info": {"category": "Technology", "name": "NVIDIA"},
            "articles": [
                {
                    "title": "Nvidia cuts guidance as data center demand slows",
                    "link": "https://www.reuters.com/technology/nvidia-guidance-2026-05-01/",
                    "source": "Reuters",
                }
            ],
        }
    }
    benchmark_quotes = {
        "ACWI": {"change_percent": -0.2},
        "SOXX": {"change_percent": 0.1},
    }

    result = build_attributions(stocks, benchmark_quotes, config)[0]

    assert result.classification == "idiosyncratic"
    assert result.residual_pct == -6.1
    assert result.evidence is not None
    assert result.evidence.confidence == "HIGH"
    assert "guidance" in result.evidence.title.lower()


def test_yahoo_price_action_article_is_not_evidence():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Why Nvidia Stock Is Moving Lower Today",
            "link": "https://finance.yahoo.com/news/why-nvidia-stock-moving-lower-120000000.html",
            "source": "Yahoo Finance",
        }
    ]

    assert evidence_for_articles(articles, config) is None


def test_allowed_source_why_guidance_article_is_evidence():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Why Nvidia cut guidance as data center demand slows",
            "link": "https://www.reuters.com/technology/nvidia-guidance-2026-05-01/",
            "source": "Reuters",
        }
    ]

    evidence = evidence_for_articles(articles, config)

    assert evidence is not None
    assert evidence.confidence == "HIGH"


def test_allowed_yahoo_guidance_article_is_context_not_high_confidence_evidence():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Nvidia cuts guidance as data center demand slows",
            "link": "https://finance.yahoo.com/news/nvidia-cuts-guidance-demand-slows-120000000.html",
            "source": "Yahoo Finance",
            "published_at": 1777966800.0,
        }
    ]

    evidence = evidence_for_articles(articles, config)
    audit = evaluate_article_evidence(articles, config)

    assert evidence is None
    assert audit.status == "no_high_confidence_evidence"
    assert audit.candidates[0].status == "context"
    assert audit.candidates[0].reason == "not_high_confidence_source"
    assert audit.candidates[0].confidence == "MEDIUM"


def test_yahoo_value_template_headline_is_not_evidence():
    config = load_benchmark_config()
    articles = [
        {
            "title": "TKR or HOCPY: Which Is the Better Value Stock Right Now?",
            "link": "https://finance.yahoo.com/markets/stocks/articles/tkr-hocpy-better-value-stock-154003828.html",
            "source": "Yahoo Finance",
            "published_at": 1777966800.0,
        }
    ]

    evidence = evidence_for_articles(articles, config)

    assert evidence is None


def test_evidence_selection_ranks_candidates_not_first_match():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Axon (AXON) Stock Outpacing Its Peers This Year?",
            "link": "https://finance.yahoo.com/news/axon-stock-outpacing-peers-120000000.html",
            "source": "Yahoo Finance",
            "published_at": 1777966800.0,
        },
        {
            "title": "Axon raises guidance as enterprise demand accelerates",
            "link": "https://www.reuters.com/world/us/axon-raises-guidance-2026-05-01/",
            "source": "Reuters",
            "published_at": 1777970400.0,
        },
    ]

    evidence = evidence_for_articles(articles, config)

    assert evidence is not None
    assert evidence.source == "Reuters"
    assert "raises guidance" in evidence.title.lower()


def test_evidence_audit_records_rejected_candidates():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Why Nvidia Stock Is Moving Lower Today",
            "link": "https://finance.yahoo.com/news/why-nvidia-stock-moving-lower-120000000.html",
            "source": "Yahoo Finance",
        },
        {
            "title": "Nvidia raises guidance as data center demand improves",
            "link": "https://www.reuters.com/technology/nvidia-guidance-2026-05-01/",
            "source": "Reuters",
        },
    ]

    audit = evaluate_article_evidence(articles, config)

    assert audit.status == "selected"
    assert audit.selected is not None
    assert audit.selected.source == "Reuters"
    assert [candidate.status for candidate in audit.candidates] == ["rejected", "selected"]
    assert audit.candidates[0].reason == "generic_price_action"


def test_peer_only_large_move_has_no_high_confidence_evidence_but_is_exception():
    config = load_benchmark_config()
    stocks = {
        "SPGI": {
            "quote": {"price": 403.95, "change_percent": -9.2},
            "info": {"category": "Financials", "name": "S&P Global"},
            "articles": [],
        }
    }

    result = build_attributions(stocks, {"XLF": {"change_percent": -0.2}, "SPY": {"change_percent": 0.1}}, config)[0]

    assert result.evidence is None
    assert result.has_high_confidence_evidence is False
    assert result.is_unresolved_exception is True
    assert result.evidence_audit.status == "no_source_coverage"


def test_allowed_german_catalyst_article_is_evidence():
    config = load_benchmark_config()
    articles = [
        {
            "title": "Siemens hebt Prognose an, Ausblick verbessert sich",
            "link": "https://www.handelsblatt.com/unternehmen/industrie/siemens-prognose-2026-05-01/",
            "source": "Handelsblatt",
        }
    ]

    evidence = evidence_for_articles(articles, config)

    assert evidence is not None
    assert evidence.confidence == "HIGH"


def test_unmapped_ticker_keeps_quantitative_attribution_with_uncertainty():
    config = load_benchmark_config()
    stocks = {
        "XYZ.L": {
            "quote": {"price": 42.0, "change_percent": 7.0},
            "info": {"category": "Unknown", "name": "Example PLC"},
            "articles": [],
        }
    }
    benchmark_quotes = {"ACWI": {"change_percent": 0.5}}

    result = build_attributions(stocks, benchmark_quotes, config)[0]

    assert result.classification == "unexplained"
    assert result.mapping_uncertain is True
    assert result.benchmark_ticker == "ACWI"


def test_tsx_venture_ticker_uses_cad_currency_context():
    config = load_benchmark_config()
    stocks = {
        "ABC.V": {
            "quote": {"price": 4.2, "change_percent": 2.0},
            "info": {"category": "Unknown", "name": "Example Venture"},
            "articles": [],
        }
    }
    benchmark_quotes = {"EWC": {"change_percent": 0.5}, "CAD=X": {"change_percent": 0.2}}

    result = build_attributions(stocks, benchmark_quotes, config)[0]

    assert result.currency == "CAD"
    assert result.currency_ticker == "CAD=X"
    assert result.market_ticker == "EWC"


def test_attributions_rank_by_relevance_before_classification():
    config = load_benchmark_config()
    stocks = {
        "NVDA": {
            "quote": {"price": 120.0, "change_percent": -3.2},
            "info": {"category": "Technology", "name": "NVIDIA"},
            "articles": [],
        },
        "6861.T": {
            "quote": {"price": 27830.0, "change_percent": -10.0},
            "info": {"category": "Industrials", "name": "Keyence"},
            "articles": [],
        },
    }
    benchmark_quotes = {
        "SPY": {"change_percent": 0.0},
        "SOXX": {"change_percent": 0.0},
        "EWJ": {"change_percent": -9.4},
        "JPY=X": {"change_percent": 0.1},
    }

    results = build_attributions(stocks, benchmark_quotes, config)

    assert results[0].symbol == "6861.T"
    assert results[0].classification == "sector_theme"
