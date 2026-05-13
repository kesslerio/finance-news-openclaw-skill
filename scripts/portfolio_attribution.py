from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
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
    "analyst",
    "antitrust",
    "buyback",
    "capex",
    "capital markets day",
    "cuts guidance",
    "demand",
    "downgrade",
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
    "q1",
    "q2",
    "q3",
    "q4",
    "rating",
    "recall",
    "regulator",
    "revenue",
    "sales",
    "supply",
    "target cut",
    "trading update",
    "upgrade",
    "absatz",
    "analyst",
    "ausblick",
    "herabstufung",
    "ergebnis",
    "ergebnisse",
    "gewinn",
    "hochgestuft",
    "marge",
    "margen",
    "prognose",
    "q1",
    "q2",
    "q3",
    "q4",
    "quartal",
    "trading update",
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
)

EXCEPTION_MOVE_THRESHOLD_PCT = 9.0


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
    title_de: str | None
    source: str
    link: str
    confidence: str


@dataclass(frozen=True)
class EvidenceCandidateAudit:
    title: str
    source: str
    link: str
    status: str
    reason: str
    score: float | None = None
    confidence: str | None = None


@dataclass(frozen=True)
class EvidenceAudit:
    status: str
    checked_count: int
    selected: Evidence | None
    candidates: tuple[EvidenceCandidateAudit, ...]


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
    evidence_audit: EvidenceAudit
    mapping_uncertain: bool
    relevance_score: float

    @property
    def has_high_confidence_evidence(self) -> bool:
        return self.evidence is not None and self.evidence.confidence == "HIGH"

    @property
    def is_unresolved_exception(self) -> bool:
        return not self.has_high_confidence_evidence and abs(self.change_pct) >= EXCEPTION_MOVE_THRESHOLD_PCT


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


def _confidence_for_source(source: str, domain: str, quality: dict) -> str:
    high_sources = [str(item).lower() for item in quality.get("high_confidence_sources", [])]
    high_domains = [str(item).lower() for item in quality.get("high_confidence_domains", [])]
    if source.lower() in high_sources or (domain and _domain_matches(domain, high_domains)):
        return "HIGH"
    return "MEDIUM"


def _article_timestamp(article: dict) -> float | None:
    raw = article.get("published_at")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _compile_patterns(raw_patterns: list[object]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for item in raw_patterns:
        token = str(item).strip()
        if not token:
            continue
        try:
            compiled.append(re.compile(token, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _template_hit_count(title: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(1 for pattern in patterns if pattern.search(title))


def _recency_bonus(
    published_at: float | None,
    *,
    max_bonus: float,
    half_life_hours: float,
) -> float:
    if published_at is None:
        return 0.0
    age_hours = max(0.0, (time.time() - published_at) / 3600.0)
    if half_life_hours <= 0:
        return 0.0
    decay = math.exp(-math.log(2.0) * (age_hours / half_life_hours))
    return max_bonus * decay


def evaluate_article_evidence(articles: list[dict], config: dict) -> EvidenceAudit:
    quality = config.get("source_quality", {})
    blocked_domains = [str(item).lower() for item in quality.get("blocked_domains", [])]
    allowed_domains = [str(item).lower() for item in quality.get("allowed_domains", [])]
    allowed_sources = [str(item).lower() for item in quality.get("allowed_sources", [])]
    scoring = quality.get("quality_scoring", {})
    min_score = float(scoring.get("minimum_score", 1.0))
    allowed_source_bonus = float(scoring.get("allowed_source_bonus", 0.5))
    catalyst_bonus = float(scoring.get("catalyst_bonus", 1.0))
    recency_max_bonus = float(scoring.get("recency_max_bonus", 0.4))
    recency_half_life_hours = float(scoring.get("recency_half_life_hours", 24.0))
    low_value_template_penalty = float(scoring.get("low_value_template_penalty", 1.5))
    low_value_patterns = _compile_patterns(scoring.get("low_value_template_patterns", []))
    candidates: list[tuple[float, float, int, Evidence]] = []
    audits: list[EvidenceCandidateAudit] = []

    for index, article in enumerate(articles):
        title = str(article.get("title") or "").strip()
        link = str(article.get("link") or "").strip()
        domain = _domain(link)
        source = str(article.get("source") or domain or "Source").strip()
        if not title:
            audits.append(EvidenceCandidateAudit(title="", source=source, link=link, status="rejected", reason="missing_title"))
            continue
        if _is_generic_article(title):
            audits.append(EvidenceCandidateAudit(title=title, source=source, link=link, status="rejected", reason="generic_price_action"))
            continue
        has_catalyst = _has_catalyst(title)
        if not has_catalyst:
            audits.append(EvidenceCandidateAudit(title=title, source=source, link=link, status="rejected", reason="no_direct_catalyst"))
            continue
        if domain and _domain_matches(domain, blocked_domains):
            audits.append(EvidenceCandidateAudit(title=title, source=source, link=link, status="rejected", reason="blocked_domain"))
            continue
        source_l = source.lower()
        allowed = source_l in allowed_sources or (domain and _domain_matches(domain, allowed_domains))
        if not allowed:
            audits.append(EvidenceCandidateAudit(title=title, source=source, link=link, status="rejected", reason="source_not_allowed"))
            continue
        published_at = _article_timestamp(article)
        score = 0.0
        score += allowed_source_bonus
        if has_catalyst:
            score += catalyst_bonus
        score += _recency_bonus(
            published_at,
            max_bonus=recency_max_bonus,
            half_life_hours=recency_half_life_hours,
        )
        score -= _template_hit_count(title, low_value_patterns) * low_value_template_penalty
        if score < min_score:
            audits.append(
                EvidenceCandidateAudit(
                    title=title,
                    source=source,
                    link=link,
                    status="rejected",
                    reason="below_quality_threshold",
                    score=round(score, 3),
                )
            )
            continue
        confidence = _confidence_for_source(source, domain, quality)
        if confidence != "HIGH":
            audits.append(
                EvidenceCandidateAudit(
                    title=title,
                    source=source,
                    link=link,
                    status="context",
                    reason="not_high_confidence_source",
                    score=round(score, 3),
                    confidence=confidence,
                )
            )
            continue
        evidence = Evidence(
            title=title,
            title_de=(str(article.get("title_de") or "").strip() or None),
            source=source,
            link=link,
            confidence=confidence,
        )
        audits.append(
            EvidenceCandidateAudit(
                title=title,
                source=source,
                link=link,
                status="eligible",
                reason="direct_catalyst",
                score=round(score, 3),
                confidence=confidence,
            )
        )
        candidates.append(
            (
                score,
                published_at or 0.0,
                index,
                evidence,
            )
        )

    if not candidates:
        status = "no_source_coverage" if not articles else "no_high_confidence_evidence"
        return EvidenceAudit(status=status, checked_count=len(articles), selected=None, candidates=tuple(audits))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = candidates[0][3]
    selected_audits = [
        EvidenceCandidateAudit(
            title=audit.title,
            source=audit.source,
            link=audit.link,
            status="selected" if audit.title == selected.title and audit.link == selected.link else audit.status,
            reason=audit.reason,
            score=audit.score,
            confidence=audit.confidence,
        )
        for audit in audits
    ]
    return EvidenceAudit(
        status="selected",
        checked_count=len(articles),
        selected=selected,
        candidates=tuple(selected_audits),
    )


def evidence_for_articles(articles: list[dict], config: dict) -> Evidence | None:
    return evaluate_article_evidence(articles, config).selected


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
        evidence_audit = evaluate_article_evidence(data.get("articles") or [], config)
        evidence = evidence_audit.selected
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
                evidence_audit=evidence_audit,
                mapping_uncertain=mapping.uncertain,
                relevance_score=score,
            )
        )

    results.sort(key=lambda item: (-item.relevance_score, CLASSIFICATION_ORDER.get(item.classification, 9)))
    return results


def evidence_audit_payload(results: list[AttributionResult]) -> list[dict]:
    payload = []
    for result in results:
        payload.append(
            {
                "symbol": result.symbol,
                "display_symbol": result.display_symbol,
                "change_pct": result.change_pct,
                "classification": result.classification,
                "visible_normal": result.has_high_confidence_evidence,
                "visible_exception": result.is_unresolved_exception,
                "evidence_status": result.evidence_audit.status,
                "selected_evidence": _evidence_payload(result.evidence_audit.selected),
                "candidates": [
                    {
                        "title": candidate.title,
                        "source": candidate.source,
                        "link": candidate.link,
                        "status": candidate.status,
                        "reason": candidate.reason,
                        "score": candidate.score,
                        "confidence": candidate.confidence,
                    }
                    for candidate in result.evidence_audit.candidates
                ],
                "attribution": {
                    "benchmark_ticker": result.benchmark_ticker,
                    "benchmark_label": result.benchmark_label,
                    "benchmark_change_pct": result.benchmark_change_pct,
                    "market_ticker": result.market_ticker,
                    "market_change_pct": result.market_change_pct,
                    "currency_ticker": result.currency_ticker,
                    "currency_change_pct": result.currency_change_pct,
                    "residual_pct": result.residual_pct,
                    "mapping_uncertain": result.mapping_uncertain,
                },
            }
        )
    return payload


def _evidence_payload(evidence: Evidence | None) -> dict | None:
    if evidence is None:
        return None
    return {
        "title": evidence.title,
        "title_de": evidence.title_de,
        "source": evidence.source,
        "link": evidence.link,
        "confidence": evidence.confidence,
    }
