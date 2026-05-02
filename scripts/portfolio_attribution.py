from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
DEFAULT_BENCHMARK_CONFIG = CONFIG_DIR / "portfolio_benchmarks.json"

CLASSIFICATION_ORDER = {
    "idiosyncratic": 0,
    "mixed": 1,
    "sector_theme": 2,
    "market": 3,
    "fx": 4,
    "unexplained": 5,
}

CATALYST_KEYWORDS = {
    "acquisition",
    "antitrust",
    "buyback",
    "capex",
    "cuts guidance",
    "demand",
    "earnings",
    "export controls",
    "forecast",
    "guidance",
    "investigation",
    "lawsuit",
    "margin",
    "merger",
    "outlook",
    "pricing",
    "profit",
    "recall",
    "regulator",
    "revenue",
    "sales",
    "supply",
    "absatz",
    "ausblick",
    "ergebnis",
    "ergebnisse",
    "gewinn",
    "marge",
    "margen",
    "prognose",
    "umsatz",
}

GENERIC_ARTICLE_PATTERNS = (
    "stock is moving",
    "stock is down",
    "stock is up",
    "stocks to watch",
    "best stocks",
    "buy now",
    "could be a millionaire",
    "price target",
)


@dataclass(frozen=True)
class BenchmarkMapping:
    benchmark_ticker: str
    benchmark_label: str
    market_ticker: str
    market_label: str
    currency_ticker: str | None
    currency_label: str | None
    reason: str
    uncertain: bool


@dataclass(frozen=True)
class Evidence:
    title: str
    source: str
    link: str
    confidence: str


@dataclass(frozen=True)
class AttributionResult:
    symbol: str
    display_symbol: str
    price: float | None
    currency: str
    change_pct: float
    benchmark_ticker: str
    benchmark_label: str
    benchmark_change_pct: float | None
    market_ticker: str
    market_change_pct: float | None
    currency_ticker: str | None
    currency_change_pct: float | None
    residual_pct: float | None
    classification: str
    evidence: Evidence | None
    mapping_uncertain: bool
    relevance_score: float


def load_benchmark_config(path: Path | None = None) -> dict:
    config_path = path or DEFAULT_BENCHMARK_CONFIG
    return json.loads(config_path.read_text(encoding="utf-8"))


def _ticker_region(symbol: str, config: dict) -> str:
    suffixes = config.get("ticker_suffix_regions", {})
    for suffix, region in suffixes.items():
        if symbol.endswith(suffix):
            return str(region)
    return "US" if "." not in symbol else "default"


def _entry_from(mapping: dict, key: str) -> tuple[str, str] | None:
    entry = mapping.get(key)
    if not isinstance(entry, dict):
        return None
    ticker = str(entry.get("ticker") or "").strip().upper()
    label = str(entry.get("label") or ticker).strip()
    if not ticker:
        return None
    return ticker, label


def _currency_proxy(symbol: str, config: dict) -> tuple[str, str] | None:
    proxies = config.get("currency_proxies", {})
    for suffix, entry in proxies.items():
        if symbol.endswith(suffix) and isinstance(entry, dict):
            ticker = str(entry.get("ticker") or "").strip().upper()
            label = str(entry.get("label") or ticker).strip()
            return (ticker, label) if ticker else None
    return None


def _theme_tokens(info: dict) -> list[str]:
    raw_values = [
        str(info.get("investment_theme") or ""),
        str(info.get("theme") or ""),
        str(info.get("notes") or ""),
    ]
    tokens: list[str] = []
    for raw_value in raw_values:
        for token in raw_value.replace(",", "|").split("|"):
            cleaned = token.strip()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
    return tokens


def _theme_entry(info: dict, config: dict) -> tuple[str, str] | None:
    themes = config.get("themes", {})
    for token in _theme_tokens(info):
        entry = _entry_from(themes, token)
        if entry:
            return entry
    return None


def resolve_benchmark_mapping(symbol: str, info: dict | None, config: dict) -> BenchmarkMapping:
    symbol_upper = symbol.upper()
    info = info or {}
    region = _ticker_region(symbol_upper, config)
    market = _entry_from(config.get("broad_markets", {}), region)
    if market is None:
        market = _entry_from(config.get("broad_markets", {}), "default") or ("ACWI", "Global equities")

    override = _entry_from(config.get("ticker_overrides", {}), symbol_upper)
    if override:
        benchmark = override
        reason = "ticker_override"
        uncertain = False
    else:
        benchmark = _theme_entry(info, config)
        if benchmark:
            reason = "theme"
            uncertain = False
        else:
            category = str(info.get("sector_specific") or info.get("category") or "").strip()
            benchmark = _entry_from(config.get("categories", {}), category)
            reason = "category" if benchmark else "market_fallback"
            uncertain = benchmark is None

    if benchmark is None:
        benchmark = market

    currency = _currency_proxy(symbol_upper, config)
    return BenchmarkMapping(
        benchmark_ticker=benchmark[0],
        benchmark_label=benchmark[1],
        market_ticker=market[0],
        market_label=market[1],
        currency_ticker=currency[0] if currency else None,
        currency_label=currency[1] if currency else None,
        reason=reason,
        uncertain=uncertain,
    )


def benchmark_tickers_for_portfolio(stocks: dict, config: dict) -> list[str]:
    tickers: list[str] = []
    for symbol, data in stocks.items():
        if not isinstance(data, dict):
            continue
        mapping = resolve_benchmark_mapping(str(symbol), data.get("info") or {}, config)
        for ticker in (mapping.market_ticker, mapping.benchmark_ticker, mapping.currency_ticker):
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers


def _change_pct(raw: dict | None) -> float | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("change_percent")
    return round(float(value), 1) if isinstance(value, (int, float)) else None


def _domain(link: str) -> str:
    host = urlparse(link).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _domain_matches(domain: str, candidates: list[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def _has_catalyst(title: str) -> bool:
    normalized = title.lower()
    return any(keyword in normalized for keyword in CATALYST_KEYWORDS)


def _is_generic_article(title: str) -> bool:
    normalized = title.lower()
    if normalized.startswith("why ") and "stock" in normalized:
        price_action = ("is moving", "is down", "is up", "moving lower", "moving higher")
        if any(pattern in normalized for pattern in price_action):
            return True
    return any(pattern in normalized for pattern in GENERIC_ARTICLE_PATTERNS)


def evidence_for_articles(articles: list[dict], config: dict) -> Evidence | None:
    quality = config.get("source_quality", {})
    blocked_domains = [str(item).lower() for item in quality.get("blocked_domains", [])]
    allowed_domains = [str(item).lower() for item in quality.get("allowed_domains", [])]
    allowed_sources = [str(item).lower() for item in quality.get("allowed_sources", [])]

    for article in articles:
        title = str(article.get("title") or "").strip()
        if not title or _is_generic_article(title) or not _has_catalyst(title):
            continue
        link = str(article.get("link") or "").strip()
        domain = _domain(link)
        if domain and _domain_matches(domain, blocked_domains):
            continue
        source = str(article.get("source") or domain or "Source").strip()
        source_l = source.lower()
        allowed = source_l in allowed_sources or (domain and _domain_matches(domain, allowed_domains))
        if allowed:
            return Evidence(title=title, source=source, link=link, confidence="HIGH")
    return None


def _classification(
    change_pct: float,
    benchmark_change: float | None,
    market_change: float | None,
    currency_change: float | None,
    residual: float | None,
    mapping_uncertain: bool,
) -> str:
    if currency_change is not None and abs(currency_change) >= 1.0 and abs(change_pct - currency_change) <= 1.2:
        return "fx"
    if benchmark_change is not None and residual is not None:
        if abs(benchmark_change) >= 1.0 and abs(residual) <= 1.0:
            return "sector_theme"
        if abs(residual) >= 3.0:
            return "unexplained" if mapping_uncertain else "idiosyncratic"
        if market_change is not None and abs(market_change) >= 1.0 and abs(change_pct - market_change) <= 1.0:
            return "market"
        if abs(benchmark_change) >= 1.0 and abs(residual) <= 2.0:
            return "mixed"
    if market_change is not None and abs(market_change) >= 1.0 and abs(change_pct - market_change) <= 1.0:
        return "market"
    return "unexplained"


def _display_symbol(symbol: str, info: dict) -> str:
    name = str(info.get("name") or "").strip()
    return f"{name} ({symbol})" if name and "." in symbol else symbol


def _currency_for_symbol(symbol: str, quote: dict) -> str:
    raw_currency = str(quote.get("currency") or "").strip().upper()
    if raw_currency:
        return raw_currency
    if symbol.endswith(".T"):
        return "JPY"
    if symbol.endswith((".TO", ".V")):
        return "CAD"
    if symbol.endswith(".SW"):
        return "CHF"
    if symbol.endswith((".DE", ".PA", ".AS")):
        return "EUR"
    if symbol.endswith(".TW"):
        return "TWD"
    return "USD"


def _target_weight(info: dict) -> float:
    try:
        return float(info.get("target_weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_attributions(stocks: dict, benchmark_quotes: dict, config: dict) -> list[AttributionResult]:
    results: list[AttributionResult] = []
    for symbol, data in stocks.items():
        if not isinstance(data, dict):
            continue
        quote = data.get("quote") or {}
        change = quote.get("change_percent")
        if not isinstance(change, (int, float)):
            continue
        info = data.get("info") or {}
        mapping = resolve_benchmark_mapping(str(symbol), info, config)
        benchmark_change = _change_pct(benchmark_quotes.get(mapping.benchmark_ticker))
        market_change = _change_pct(benchmark_quotes.get(mapping.market_ticker))
        currency_change = _change_pct(benchmark_quotes.get(mapping.currency_ticker))
        residual = round(float(change) - benchmark_change, 1) if benchmark_change is not None else None
        classification = _classification(float(change), benchmark_change, market_change, currency_change, residual, mapping.uncertain)
        evidence = evidence_for_articles(data.get("articles") or [], config)
        score = abs(float(change)) + (_target_weight(info) * 100)
        results.append(
            AttributionResult(
                symbol=str(symbol),
                display_symbol=_display_symbol(str(symbol), info),
                price=quote.get("price") if isinstance(quote.get("price"), (int, float)) else None,
                currency=_currency_for_symbol(str(symbol), quote),
                change_pct=round(float(change), 1),
                benchmark_ticker=mapping.benchmark_ticker,
                benchmark_label=mapping.benchmark_label,
                benchmark_change_pct=benchmark_change,
                market_ticker=mapping.market_ticker,
                market_change_pct=market_change,
                currency_ticker=mapping.currency_ticker,
                currency_change_pct=currency_change,
                residual_pct=residual,
                classification=classification,
                evidence=evidence,
                mapping_uncertain=mapping.uncertain,
                relevance_score=score,
            )
        )

    results.sort(key=lambda item: (-item.relevance_score, CLASSIFICATION_ORDER.get(item.classification, 9)))
    return results
