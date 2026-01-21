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
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
CACHE_DIR = SCRIPT_DIR.parent / "cache"

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)


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
        "3. Add to PATH: export PATH=$PATH:/home/art/.local/bin\n\n"
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


def fetch_rss(url: str, limit: int = 10) -> list[dict]:
    """Fetch and parse RSS feed."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Clawdbot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
        
        root = ET.fromstring(content)
        items = []
        
        # Handle both RSS 2.0 and Atom formats
        for item in root.findall('.//item')[:limit]:
            title = item.find('title')
            link = item.find('link')
            pub_date = item.find('pubDate')
            description = item.find('description')
            
            items.append({
                'title': title.text if title is not None else '',
                'link': link.text if link is not None else '',
                'date': pub_date.text if pub_date is not None else '',
                'description': (description.text or '')[:200] if description is not None else ''
            })
        
        return items
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}", file=sys.stderr)
        return []


def fetch_market_data(symbols: list[str]) -> dict:
    """Fetch market data using openbb-quote."""
    results = {}
    
    # Check if openbb-quote is available
    if OPENBB_BINARY is None:
        print("❌ openbb-quote not available - skipping market data fetch", file=sys.stderr)
        return results
    
    for symbol in symbols:
        try:
            result = subprocess.run(
                [OPENBB_BINARY, symbol],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=False
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
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
            data = fetch_market_data([symbol])
            if symbol in data:
                result['markets'][region]['indices'][symbol] = {
                    'name': config['index_names'].get(symbol, symbol),
                    'data': data[symbol]
                }
    
    # Fetch top headlines from CNBC and Yahoo
    for source in ['cnbc', 'yahoo']:
        if source in sources['rss_feeds']:
            feeds = sources['rss_feeds'][source]
            feed_url = feeds.get('top') or feeds.get('markets') or list(feeds.values())[1]
            articles = fetch_rss(feed_url, limit)
            for article in articles:
                article['source'] = feeds.get('name', source)
            result['headlines'].extend(articles)
    
    return result


def fetch_market_news(args):
    """Fetch market overview (indices + top headlines)."""
    result = get_market_news(args.limit)
    
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


def get_portfolio_news(limit: int = 5, max_stocks: int = 5) -> dict:
    """Get news for portfolio stocks as data."""
    # Get symbols from portfolio
    try:
        result = subprocess.run(
            ['python3', str(SCRIPT_DIR / 'portfolio.py'), 'symbols'],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False
        )
        
        if result.returncode != 0:
            print(f"❌ Failed to load portfolio: {result.stderr}", file=sys.stderr)
            return {
                'fetched_at': datetime.now().isoformat(),
                'stocks': {},
                'error': f"Portfolio load failed: {result.stderr}"
            }
        
        symbols = result.stdout.strip().split(',')
    
    except subprocess.TimeoutExpired:
        print("❌ Portfolio fetch timeout", file=sys.stderr)
        return {
            'fetched_at': datetime.now().isoformat(),
            'stocks': {},
            'error': 'Portfolio fetch timeout'
        }
    except Exception as e:
        print(f"❌ Portfolio error: {e}", file=sys.stderr)
        return {
            'fetched_at': datetime.now().isoformat(),
            'stocks': {},
            'error': str(e)
        }
    
    # Limit to max 5 stocks for briefings (performance)
    if max_stocks and len(symbols) > max_stocks:
        symbols = symbols[:max_stocks]
    
    news = {
        'fetched_at': datetime.now().isoformat(),
        'stocks': {}
    }
    
    for symbol in symbols:
        if not symbol:
            continue
        
        articles = fetch_ticker_news(symbol, limit)
        quotes = fetch_market_data([symbol])
        
        news['stocks'][symbol] = {
            'quote': quotes.get(symbol, {}),
            'articles': articles
        }

    return news


def fetch_portfolio_news(args):
    """Fetch news for portfolio stocks."""
    news = get_portfolio_news(args.limit, args.max_stocks)
    
    # Check for errors (P2 fix: preserve non-zero exit on failure)
    if 'error' in news:
        if not args.json:
            print(f"\n❌ Error: {news['error']}", file=sys.stderr)
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
    market_parser.set_defaults(func=fetch_market_news)
    
    # Portfolio news
    portfolio_parser = subparsers.add_parser('portfolio', help='News for portfolio stocks')
    portfolio_parser.add_argument('--json', action='store_true', help='Output as JSON')
    portfolio_parser.add_argument('--limit', type=int, default=5, help='Max articles per source')
    portfolio_parser.add_argument('--max-stocks', type=int, default=5, help='Max stocks to fetch (default: 5)')
    portfolio_parser.set_defaults(func=fetch_portfolio_news)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
