#!/usr/bin/env python3
"""
News Fetcher - Aggregate news from multiple sources.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from utils import clamp_timeout, compute_deadline, time_left
from datetime import datetime, timedelta
from pathlib import Path
import ssl
import urllib.request
import urllib.error

import feedparser

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
CACHE_DIR = SCRIPT_DIR.parent / "cache"

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

CA_FILE = (
    os.environ.get("SSL_CERT_FILE")
    or ("/etc/ssl/certs/ca-bundle.crt" if os.path.exists("/etc/ssl/certs/ca-bundle.crt") else None)
    or ("/etc/ssl/certs/ca-certificates.crt" if os.path.exists("/etc/ssl/certs/ca-certificates.crt") else None)
)
SSL_CONTEXT = ssl.create_default_context(cafile=CA_FILE) if CA_FILE else ssl.create_default_context()


class PortfolioError(Exception):
    """Portfolio configuration or fetch error."""


def ensure_portfolio_config():
    """Copy portfolio.csv.example to portfolio.csv if real file doesn't exist."""
    example_file = CONFIG_DIR / "portfolio.csv.example"
    real_file = CONFIG_DIR / "portfolio.csv"

    if real_file.exists():
        return

    if example_file.exists():
        try:
            shutil.copy(example_file, real_file)
            print(f"📋 Created portfolio.csv from example", file=sys.stderr)
        except PermissionError:
            print(f"⚠️ Cannot create portfolio.csv (read-only environment)", file=sys.stderr)
    else:
        print(f"⚠️ No portfolio.csv or portfolio.csv.example found", file=sys.stderr)


# Initialize user config (copy example if needed)
ensure_portfolio_config()


def get_openbb_binary() -> str:
    """
    Find openbb-quote binary.
    
    Checks (in order):
    1. OPENBB_QUOTE_BIN environment variable
    2. PATH via shutil.which()
    
    Returns:
        Path to openbb-quote binary
        
    Raises:
        RuntimeError: If openbb-quote is not found
    """
    # Check env var override
    env_path = os.environ.get('OPENBB_QUOTE_BIN')
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        else:
            print(f"⚠️ OPENBB_QUOTE_BIN={env_path} is not a valid executable", file=sys.stderr)
    
    # Check PATH
    binary = shutil.which('openbb-quote')
    if binary:
        return binary
    
    # Not found - show helpful error
    raise RuntimeError(
        "openbb-quote not found!\n\n"
        "Installation options:\n"
        "1. Install via pip: pip install openbb\n"
        "2. Use existing install: export OPENBB_QUOTE_BIN=/path/to/openbb-quote\n"
        "3. Add to PATH: export PATH=$PATH:$HOME/.local/bin\n\n"
        "See: https://github.com/kesslerio/finance-news-clawdbot-skill#dependencies"
    )


# Cache the binary path on module load
try:
    OPENBB_BINARY = get_openbb_binary()
except RuntimeError as e:
    print(f"❌ {e}", file=sys.stderr)
    OPENBB_BINARY = None


def load_sources():
    """Load source configuration."""
    with open(CONFIG_DIR / "sources.json", 'r') as f:
        return json.load(f)


def _get_best_feed_url(feeds: dict) -> str | None:
    """Get the best feed URL from a feeds configuration dict.
    
    Uses explicit priority list and validates URLs to avoid selecting
    non-URL values like 'name' or other config keys.
    
    Args:
        feeds: Dict with feed keys like 'top', 'markets', 'tech'
        
    Returns:
        Best URL string or None if no valid URL found
    """
    # Priority order for feed types (most relevant first)
    PRIORITY_KEYS = ['top', 'markets', 'headlines', 'breaking']
    
    for key in PRIORITY_KEYS:
        if key in feeds:
            value = feeds[key]
            # Validate it's a string and starts with http
            if isinstance(value, str) and value.startswith('http'):
                return value
    
    # Fallback: search all values for valid URLs (skip non-string/non-URL)
    for key, value in feeds.items():
        if key == 'name':
            continue  # Skip 'name' field
        if isinstance(value, str) and value.startswith('http'):
            return value
    
    return None


def fetch_rss(url: str, limit: int = 10, timeout: int = 15) -> list[dict]:
    """Fetch and parse RSS/Atom feed using feedparser."""
    try:
        # Fetch feed content first (handles SSL/certs properly)
        req = urllib.request.Request(url, headers={'User-Agent': 'Clawdbot/1.0'})
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
            content = response.read()
        
        # Parse with feedparser (handles RSS and Atom formats)
        parsed = feedparser.parse(content)
        
        items = []
        for entry in parsed.entries[:limit]:
            # Skip entries without title or link
            title = entry.get('title', '').strip()
            if not title:
                continue
            
            # Link handling: Atom uses 'link' dict, RSS uses string
            link = entry.get('link', '')
            if isinstance(link, dict):
                link = link.get('href', '').strip()
            if not link:
                continue
            
            # Date handling: different formats across feeds
            published = entry.get('published', '') or entry.get('updated', '')
            
            # Description handling: summary vs description
            description = entry.get('summary', '') or entry.get('description', '')
            
            items.append({
                'title': title,
                'link': link,
                'date': published.strip() if published else '',
                'description': (description or '')[:200].strip()
            })
        
        return items
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}", file=sys.stderr)
        return []


def fetch_market_data(
    symbols: list[str],
    timeout: int = 30,
    deadline: float | None = None,
    allow_price_fallback: bool = False,
) -> dict:
    """Fetch market data using openbb-quote."""
    results = {}
    
    # Check if openbb-quote is available
    if OPENBB_BINARY is None:
        print("❌ openbb-quote not available - skipping market data fetch", file=sys.stderr)
        return results
    
    for symbol in symbols:
        try:
            try:
                effective_timeout = clamp_timeout(timeout, deadline)
            except TimeoutError:
                return results
            result = subprocess.run(
                [OPENBB_BINARY, symbol],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=effective_timeout,
                check=False
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Normalize provider output
                if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                    data = data["results"][0] if data["results"] else {}
                elif isinstance(data, list):
                    data = data[0] if data else {}

                # Fix missing prices for indices only when explicitly allowed.
                if allow_price_fallback and isinstance(data, dict) and data.get("price") is None:
                    if data.get("open") is not None:
                        data["price"] = data["open"]
                    elif data.get("prev_close") is not None:
                        data["price"] = data["prev_close"]

                # Calculate change_percent
                if (
                    isinstance(data, dict)
                    and data.get("change_percent") is None
                    and data.get("price")
                    and data.get("prev_close")
                ):
                    price = data["price"]
                    prev_close = data["prev_close"]
                    if prev_close != 0:
                        data["change_percent"] = ((price - prev_close) / prev_close) * 100

                results[symbol] = data
        except subprocess.TimeoutExpired:
            print(f"⚠️ Timeout fetching {symbol}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON from openbb-quote for {symbol}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}", file=sys.stderr)
    
    return results


def fetch_ticker_news(symbol: str, limit: int = 5) -> list[dict]:
    """Fetch news for a specific ticker via Yahoo Finance RSS."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    return fetch_rss(url, limit)


def get_cached_news(cache_key: str) -> dict | None:
    """Get cached news if fresh (< 15 minutes)."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(minutes=15):
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    return None


def save_cache(cache_key: str, data: dict):
    """Save news to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def fetch_all_news(args):
    """Fetch news from all configured sources."""
    sources = load_sources()
    cache_key = f"all_news_{datetime.now().strftime('%Y%m%d_%H')}"
    
    # Check cache first
    if not args.force:
        cached = get_cached_news(cache_key)
        if cached:
            print(json.dumps(cached, indent=2))
            return
    
    news = {
        'fetched_at': datetime.now().isoformat(),
        'sources': {}
    }
    
    # Fetch RSS feeds
    for source_id, feeds in sources['rss_feeds'].items():
        # Skip disabled sources
        if not feeds.get('enabled', True):
            continue
            
        news['sources'][source_id] = {
            'name': feeds.get('name', source_id),
            'articles': []
        }
        
        for feed_name, feed_url in feeds.items():
            if feed_name in ('name', 'enabled', 'note'):
                continue
            
            articles = fetch_rss(feed_url, args.limit)
            for article in articles:
                article['feed'] = feed_name
            news['sources'][source_id]['articles'].extend(articles)
    
    # Save to cache
    save_cache(cache_key, news)
    
    if args.json:
        print(json.dumps(news, indent=2))
    else:
        for source_id, source_data in news['sources'].items():
            print(f"\n### {source_data['name']}\n")
            for article in source_data['articles'][:args.limit]:
                print(f"• {article['title']}")
                if args.verbose and article.get('description'):
                    print(f"  {article['description'][:100]}...")


def get_market_news(
    limit: int = 5,
    regions: list[str] | None = None,
    max_indices_per_region: int | None = None,
    deadline: float | None = None,
    rss_timeout: int = 15,
    subprocess_timeout: int = 30,
) -> dict:
    """Get market overview (indices + top headlines) as data."""
    sources = load_sources()
    
    result = {
        'fetched_at': datetime.now().isoformat(),
        'markets': {},
        'headlines': []
    }
    
    # Fetch market indices
    for region, config in sources['markets'].items():
        if time_left(deadline) is not None and time_left(deadline) <= 0:
            break
        if regions is not None and region not in regions:
            continue

        result['markets'][region] = {
            'name': config['name'],
            'indices': {}
        }
        
        symbols = config['indices']
        if max_indices_per_region is not None:
            symbols = symbols[:max_indices_per_region]

        for symbol in symbols:
            if time_left(deadline) is not None and time_left(deadline) <= 0:
                break
            data = fetch_market_data(
                [symbol],
                timeout=subprocess_timeout,
                deadline=deadline,
                allow_price_fallback=True,
            )
            if symbol in data:
                result['markets'][region]['indices'][symbol] = {
                    'name': config['index_names'].get(symbol, symbol),
                    'data': data[symbol]
                }
    
    # Fetch top headlines from CNBC and Yahoo
    for source in ['cnbc', 'yahoo']:
        if time_left(deadline) is not None and time_left(deadline) <= 0:
            break
        if source in sources['rss_feeds']:
            feeds = sources['rss_feeds'][source]
            feed_url = _get_best_feed_url(feeds)
            if feed_url:
                try:
                    effective_timeout = clamp_timeout(rss_timeout, deadline)
                except TimeoutError:
                    break
                articles = fetch_rss(feed_url, limit, timeout=effective_timeout)
                for article in articles:
                    article['source'] = feeds.get('name', source)
                result['headlines'].extend(articles)
    
    return result


def fetch_market_news(args):
    """Fetch market overview (indices + top headlines)."""
    deadline = compute_deadline(args.deadline)
    result = get_market_news(args.limit, deadline=deadline)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n📊 Market Overview\n")
        for region, data in result['markets'].items():
            print(f"**{data['name']}**")
            for symbol, idx in data['indices'].items():
                if 'data' in idx and idx['data']:
                    price = idx['data'].get('price', 'N/A')
                    change_pct = idx['data'].get('change_percent', 0)
                    emoji = '📈' if change_pct >= 0 else '📉'
                    print(f"  {emoji} {idx['name']}: {price} ({change_pct:+.2f}%)")
            print()
        
        print("\n🔥 Top Headlines\n")
        for article in result['headlines'][:args.limit]:
            print(f"• [{article['source']}] {article['title']}")


def get_portfolio_news(
    limit: int = 5,
    max_stocks: int = 5,
    deadline: float | None = None,
    subprocess_timeout: int = 30,
) -> dict:
    """Get news for portfolio stocks as data."""
    if not (CONFIG_DIR / "portfolio.csv").exists():
        raise PortfolioError("Portfolio config missing: config/portfolio.csv")
    # Get symbols from portfolio
    try:
        effective_timeout = clamp_timeout(10, deadline)
        result = subprocess.run(
            ['python3', str(SCRIPT_DIR / 'portfolio.py'), 'symbols'],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=effective_timeout,
            check=False
        )
        
        if result.returncode != 0:
            print(f"❌ Failed to load portfolio: {result.stderr}", file=sys.stderr)
            raise PortfolioError(f"Portfolio load failed: {result.stderr}")
        
        symbols = [s.strip() for s in result.stdout.strip().split(',') if s.strip()]
        if not symbols:
            raise PortfolioError("No portfolio symbols found")
    
    except subprocess.TimeoutExpired:
        print("❌ Portfolio fetch timeout", file=sys.stderr)
        raise PortfolioError("Portfolio fetch timeout")
    except TimeoutError:
        raise PortfolioError("Deadline exceeded while fetching portfolio symbols")
    except PortfolioError:
        raise
    except Exception as e:
        print(f"❌ Portfolio error: {e}", file=sys.stderr)
        raise PortfolioError(str(e))
    
    # Limit stocks for performance

    if max_stocks and len(symbols) > max_stocks:
        symbols = symbols[:max_stocks]
    
    news = {
        'fetched_at': datetime.now().isoformat(),
        'stocks': {}
    }
    
    for symbol in symbols:
        if time_left(deadline) is not None and time_left(deadline) <= 0:
            print("⚠️ Deadline exceeded; returning partial portfolio news", file=sys.stderr)
            break
        if not symbol:
            continue
        
        articles = fetch_ticker_news(symbol, limit)
        quotes = fetch_market_data(
            [symbol],
            timeout=subprocess_timeout,
            deadline=deadline,
        )
        
        news['stocks'][symbol] = {
            'quote': quotes.get(symbol, {}),
            'articles': articles
        }

    return news


def fetch_portfolio_news(args):
    """Fetch news for portfolio stocks."""
    try:
        deadline = compute_deadline(args.deadline)
        news = get_portfolio_news(
            args.limit,
            args.max_stocks,
            deadline=deadline
        )
    except PortfolioError as exc:
        if not args.json:
            print(f"\n❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(news, indent=2))
    else:
        print(f"\n📊 Portfolio News ({len(news['stocks'])} stocks)\n")
        for symbol, data in news['stocks'].items():
            quote = data.get('quote', {})
            price = quote.get('price')
            prev_close = quote.get('prev_close', 0)
            open_price = quote.get('open', 0)

            # Calculate daily change
            # If markets are closed (price is null), calculate from last session (prev_close vs day-before close)
            # Since we don't have day-before close, use open -> prev_close as proxy for last session move
            change_pct = 0
            display_price = price or prev_close

            if price and prev_close and prev_close != 0:
                # Markets open: current price vs prev close
                change_pct = ((price - prev_close) / prev_close) * 100
            elif not price and open_price and prev_close and prev_close != 0:
                # Markets closed: last session change (prev_close vs open)
                change_pct = ((prev_close - open_price) / open_price) * 100

            emoji = '📈' if change_pct >= 0 else '📉'
            price_str = f"${display_price:.2f}" if isinstance(display_price, (int, float)) else str(display_price)

            print(f"\n**{symbol}** {emoji} {price_str} ({change_pct:+.2f}%)")
            for article in data['articles'][:3]:
                print(f"  • {article['title'][:80]}...")
def get_portfolio_symbols() -> list[str]:
    """Get list of portfolio symbols."""
    try:
        result = subprocess.run(
            ['python3', str(SCRIPT_DIR / 'portfolio.py'), 'symbols'],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False
        )
        if result.returncode == 0:
            return [s.strip() for s in result.stdout.strip().split(',') if s.strip()]
    except Exception:
        pass
    return []


def deduplicate_news(articles: list[dict]) -> list[dict]:
    """Remove duplicate news by URL, fallback to title+date."""
    seen = set()
    unique = []
    for article in articles:
        url = article.get('link', '')
        if not url:
            key = f"{article.get('title', '')}|{article.get('date', '')}"
        else:
            key = url
        if key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def get_portfolio_only_news(limit_per_ticker: int = 5) -> dict:
    """
    Get portfolio news with top 5 gainers and 5 losers, plus news per ticker.
    
    Args:
        limit_per_ticker: Max news items per ticker (default: 5)
    
    Returns:
        dict with 'gainers', 'losers' (each: list of tickers with price + news)
    """
    symbols = get_portfolio_symbols()
    if not symbols:
        return {'error': 'No portfolio symbols found', 'gainers': [], 'losers': []}
    
    # Fetch prices for all symbols
    quotes = fetch_market_data(symbols)
    
    # Build list of (symbol, change_pct)
    tickers_with_prices = []
    for symbol in symbols:
        quote = quotes.get(symbol, {})
        price = quote.get('price')
        prev_close = quote.get('prev_close', 0)
        open_price = quote.get('open', 0)
        
        if price and prev_close and prev_close != 0:
            change_pct = ((price - prev_close) / prev_close) * 100
        elif price and open_price and open_price != 0:
            change_pct = ((price - open_price) / open_price) * 100
        else:
            change_pct = 0
        
        tickers_with_prices.append({
            'symbol': symbol,
            'price': price,
            'change_pct': change_pct,
            'quote': quote
        })
    
    # Sort by change_pct
    sorted_tickers = sorted(tickers_with_prices, key=lambda x: x['change_pct'], reverse=True)
    
    # Get top 5 gainers and 5 losers
    gainers = sorted_tickers[:5]
    losers = sorted_tickers[-5:][::-1]  # Reverse to show biggest loser first
    
    # Fetch news for each ticker
    for ticker_list in [gainers, losers]:
        for ticker in ticker_list:
            symbol = ticker['symbol']
            # Try RSS first
            articles = fetch_ticker_news(symbol, limit_per_ticker)
            if not articles:
                # Fallback to web search if no RSS
                articles = web_search_news(symbol, limit_per_ticker)
            ticker['news'] = deduplicate_news(articles)
    
    return {
        'fetched_at': datetime.now().isoformat(),
        'gainers': gainers,
        'losers': losers
    }


def web_search_news(symbol: str, limit: int = 5) -> list[dict]:
    """Fallback: search for news via web search."""
    articles = []
    try:
        result = subprocess.run(
            ['web-search', f'{symbol} stock news today', '--count', str(limit)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        if result.returncode == 0:
            import json as json_mod
            data = json_mod.loads(result.stdout)
            for item in data.get('results', [])[:limit]:
                articles.append({
                    'title': item.get('title', ''),
                    'link': item.get('url', ''),
                    'source': item.get('site', 'Web'),
                    'date': '',
                    'description': ''
                })
    except Exception as e:
        print(f"⚠️ Web search failed for {symbol}: {e}", file=sys.stderr)
    return articles


def fetch_portfolio_only(args):
    """Fetch portfolio-only news (top 5 gainers + top 5 losers with news)."""
    result = get_portfolio_only_news(limit_per_ticker=args.limit)
    
    if "error" in result:
        print(f"\n❌ Error: {result.get('error', 'Unknown')}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    # Text output
    def format_ticker(ticker: dict):
        symbol = ticker['symbol']
        price = ticker.get('price')
        change = ticker['change_pct']
        emoji = '📈' if change >= 0 else '📉'
        price_str = f"${price:.2f}" if price else 'N/A'
        lines = [f"**{symbol}** {emoji} {price_str} ({change:+.2f}%)"]
        if ticker.get('news'):
            for article in ticker['news'][:args.limit]:
                source = article.get('source', 'Unknown')
                title = article.get('title', '')[:70]
                lines.append(f"  • [{source}] {title}...")
        else:
            lines.append("  • No recent news")
        return '\n'.join(lines)
    
    print("\n🚀 **Top Gainers**\n")
    for ticker in result['gainers']:
        print(format_ticker(ticker))
        print()
    
    print("\n📉 **Top Losers**\n")
    for ticker in result['losers']:
        print(format_ticker(ticker))
        print()


def main():
    parser = argparse.ArgumentParser(description='News Fetcher')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # All news
    all_parser = subparsers.add_parser('all', help='Fetch all news sources')
    all_parser.add_argument('--json', action='store_true', help='Output as JSON')
    all_parser.add_argument('--limit', type=int, default=5, help='Max articles per source')
    all_parser.add_argument('--force', action='store_true', help='Bypass cache')
    all_parser.add_argument('--verbose', '-v', action='store_true', help='Show descriptions')
    all_parser.set_defaults(func=fetch_all_news)
    
    # Market news
    market_parser = subparsers.add_parser('market', help='Market overview + headlines')
    market_parser.add_argument('--json', action='store_true', help='Output as JSON')
    market_parser.add_argument('--limit', type=int, default=5, help='Max articles per source')
    market_parser.add_argument('--deadline', type=int, default=None, help='Overall deadline in seconds')
    market_parser.set_defaults(func=fetch_market_news)
    
    # Portfolio news
    portfolio_parser = subparsers.add_parser('portfolio', help='News for portfolio stocks')
    portfolio_parser.add_argument('--json', action='store_true', help='Output as JSON')
    portfolio_parser.add_argument('--limit', type=int, default=5, help='Max articles per source')
    portfolio_parser.add_argument('--max-stocks', type=int, default=5, help='Max stocks to fetch (default: 5)')
    portfolio_parser.add_argument('--deadline', type=int, default=None, help='Overall deadline in seconds')
    portfolio_parser.set_defaults(func=fetch_portfolio_news)
    
    # Portfolio-only news (top 5 gainers + top 5 losers)
    portfolio_only_parser = subparsers.add_parser('portfolio-only', help='Top 5 gainers + top 5 losers with news')
    portfolio_only_parser.add_argument('--json', action='store_true', help='Output as JSON')
    portfolio_only_parser.add_argument('--limit', type=int, default=5, help='Max news items per ticker')
    portfolio_only_parser.set_defaults(func=fetch_portfolio_only)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
