#!/usr/bin/env python3
"""
Research Module - Deep research using Gemini CLI.
Crawls articles, finds correlations, researches companies.
Outputs research_report.md for later analysis.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fetch_news import get_market_news, get_portfolio_news

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
OUTPUT_DIR = SCRIPT_DIR.parent / "research"


def format_market_data(market_data: dict) -> str:
    """Format market data for research prompt."""
    lines = ["## Marktdaten\n"]
    
    for region, data in market_data.get('markets', {}).items():
        lines.append(f"### {data['name']}")
        for symbol, idx in data.get('indices', {}).items():
            if 'data' in idx and idx['data']:
                price = idx['data'].get('price', 'N/A')
                change_pct = idx['data'].get('change_percent', 0)
                emoji = '📈' if change_pct >= 0 else '📉'
                lines.append(f"- {idx['name']}: {price} ({change_pct:+.2f}%) {emoji}")
        lines.append("")
    
    return '\n'.join(lines)


def format_headlines(headlines: list) -> str:
    """Format headlines for research prompt."""
    lines = ["## Aktuelle Schlagzeilen\n"]
    
    for article in headlines[:20]:
        source = article.get('source', 'Unknown')
        title = article.get('title', '')
        link = article.get('link', '')
        lines.append(f"- [{source}] {title}")
        if link:
            lines.append(f"  URL: {link}")
    
    return '\n'.join(lines)


def format_portfolio_news(portfolio_data: dict) -> str:
    """Format portfolio news for research prompt."""
    lines = ["## Portfolio Analyse\n"]
    
    for symbol, data in portfolio_data.get('stocks', {}).items():
        quote = data.get('quote', {})
        price = quote.get('price', 'N/A')
        change_pct = quote.get('change_percent', 0)
        
        lines.append(f"### {symbol} (${price}, {change_pct:+.2f}%)")
        
        for article in data.get('articles', [])[:5]:
            title = article.get('title', '')
            link = article.get('link', '')
            lines.append(f"- {title}")
            if link:
                lines.append(f"  URL: {link}")
        lines.append("")
    
    return '\n'.join(lines)


def research_with_gemini(content: str, focus_areas: list = None) -> str:
    """Perform deep research using Gemini CLI.
    
    Args:
        content: Combined market/headlines/portfolio content
        focus_areas: Optional list of focus areas (e.g., ['earnings', 'macro', 'sectors'])
    
    Returns:
        Research report text
    """
    focus_prompt = ""
    if focus_areas:
        focus_prompt = f"""
Fokusbereiche für die Recherche:
{', '.join(f'- {area}' for area in focus_areas)}

Gehe bei jedem Punkt tief ins Detail.
"""
    
    prompt = f"""Du bist ein erfahrener Investment-Research-Analyst.

Deine Aufgabe ist es, tiefgehende Recherche zu aktuellen Marktentwicklungen zu liefern.

{focus_prompt}
Bitte analysiere folgende Marktdaten:

{content}

## Analyse-Anforderungen:

1. **Makrotrends**: Was treibt den Markt heute? Welche Wirtschaftsdaten/Entscheidungen sind relevant?

2. **Sektor-Analyse**: Welche Sektoren performen am besten/schlechtesten? Warum?

3. **Unternehmens-Nachrichten**: Relevante Earnings, Übernahmen, Produkt-Launches?

4. **Risiken**: Welche Abwärtsrisiken sollten beachtet werden?

5. **Chancen**: Welche positiven Entwicklungen bieten Opportunitäten?

6. **Korrelationen**: Gibt es Zusammenhänge zwischen verschiedenen Nachrichten/Asset-Klassen?

7. **Handels-Ideen**: Konkrete Setups basierend auf der Analyse (keine Finanzberatung!)

8. **Quellen**: Original-Links für weitere Recherche

Sei analytisch, objektiv und meinungsstark wo es angebracht ist.
Liefere einen substanziellen Bericht (500-800 Wörter).
"""

    try:
        result = subprocess.run(
            ['gemini', prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"⚠️ Gemini research error: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "⚠️ Gemini research timeout"
    except FileNotFoundError:
        return "⚠️ Gemini CLI not found. Install: brew install gemini-cli"


def generate_research_report(args):
    """Generate full research report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    config_path = CONFIG_DIR / "sources.json"
    if not config_path.exists():
        print("⚠️ No config found. Run 'finance-news wizard' first.", file=sys.stderr)
        sys.exit(1)
    
    # Fetch fresh data
    print("📡 Fetching market data...", file=sys.stderr)
    
    # Get market overview
    market_data = get_market_news(
        args.limit if hasattr(args, 'limit') else 5,
        regions=args.regions.split(',') if hasattr(args, 'regions') else ["us", "europe"],
        max_indices_per_region=2
    )
    
    # Get portfolio news
    portfolio_data = get_portfolio_news(
        args.limit if hasattr(args, 'limit') else 5,
        args.max_stocks if hasattr(args, 'max_stocks') else 10
    )
    
    # Build content
    content_parts = []
    
    if market_data:
        content_parts.append(format_market_data(market_data))
        if market_data.get('headlines'):
            content_parts.append(format_headlines(market_data['headlines']))
    
    if portfolio_data and 'error' not in portfolio_data:
        content_parts.append(format_portfolio_news(portfolio_data))
    
    content = '\n\n'.join(content_parts)
    
    if not content.strip():
        print("⚠️ No data available for research", file=sys.stderr)
        return
    
    # Run Gemini research
    print("🔬 Running deep research with Gemini...", file=sys.stderr)
    
    focus_areas = None
    if hasattr(args, 'focus') and args.focus:
        focus_areas = args.focus.split(',')
    
    research_report = research_with_gemini(content, focus_areas)
    
    # Add metadata header
    timestamp = datetime.now().isoformat()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    full_report = f"""# Marktforschungsbericht
**Generiert:** {date_str}
**Quelle:** Finance News Skill

---

{research_report}

---

*Dieser Bericht wurde automatisch generiert. Keine Finanzberatung.*
"""
    
    # Save to file
    output_file = OUTPUT_DIR / f"research_{datetime.now().strftime('%Y-%m-%d')}.md"
    with open(output_file, 'w') as f:
        f.write(full_report)
    
    print(f"✅ Research report saved to: {output_file}", file=sys.stderr)
    
    # Also output to stdout
    if args.json:
        print(json.dumps({
            'report': research_report,
            'saved_to': str(output_file),
            'timestamp': timestamp
        }))
    else:
        print("\n" + "="*60)
        print("RESEARCH REPORT")
        print("="*60)
        print(research_report)


def main():
    parser = argparse.ArgumentParser(description='Deep Market Research')
    parser.add_argument('--limit', type=int, default=5, help='Max headlines per source')
    parser.add_argument('--regions', default='us,europe', help='Comma-separated regions')
    parser.add_argument('--max-stocks', type=int, default=10, help='Max portfolio stocks')
    parser.add_argument('--focus', help='Focus areas (comma-separated)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    generate_research_report(args)


if __name__ == '__main__':
    main()
