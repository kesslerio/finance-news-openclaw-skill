#!/usr/bin/env python3
"""
News Summarizer - Generate AI summaries of market news in configurable language.
Uses Gemini CLI for summarization and translation.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"


def load_config():
    """Load source configuration."""
    with open(CONFIG_DIR / "sources.json", 'r') as f:
        return json.load(f)


def summarize_with_gemini(content: str, language: str = "de", style: str = "briefing") -> str:
    """Generate AI summary using Gemini CLI."""
    
    lang_prompts = {
        "de": "Antworte auf Deutsch.",
        "en": "Respond in English."
    }
    
    style_prompts = {
        "briefing": """Du bist ein Finanzanalyst, der einen prägnanten Markt-Briefing erstellt.
Fasse die wichtigsten Punkte zusammen:
- Marktstimmung (bullisch/bärisch/neutral)
- Top 3 wichtigste Nachrichten
- Auswirkungen auf das Portfolio
- Kurze Handlungsempfehlung

Halte es unter 200 Wörtern. Verwende Emojis sparsam für Lesbarkeit.""",
        
        "analysis": """Du bist ein erfahrener Finanzanalyst.
Analysiere die Nachrichten und gib:
- Detaillierte Marktanalyse
- Sektortrends
- Risiken und Chancen
- Konkrete Empfehlungen

Sei professionell aber verständlich.""",
        
        "headlines": """Fasse die wichtigsten Schlagzeilen in 5 Bulletpoints zusammen.
Jeder Punkt sollte maximal 15 Wörter haben."""
    }
    
    prompt = f"""{style_prompts.get(style, style_prompts['briefing'])}

{lang_prompts.get(language, lang_prompts['de'])}

Hier sind die aktuellen Marktnachrichten:

{content}
"""
    
    try:
        result = subprocess.run(
            ['gemini', prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"⚠️ Gemini error: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "⚠️ Gemini timeout"
    except FileNotFoundError:
        return "⚠️ Gemini CLI not found. Install: brew install gemini-cli"


def format_market_data(market_data: dict) -> str:
    """Format market data for the prompt."""
    lines = ["## Marktdaten\n"]
    
    for region, data in market_data.get('markets', {}).items():
        lines.append(f"### {data['name']}")
        for symbol, idx in data.get('indices', {}).items():
            if 'data' in idx and idx['data']:
                price = idx['data'].get('price', 'N/A')
                change_pct = idx['data'].get('change_percent', 0)
                lines.append(f"- {idx['name']}: {price} ({change_pct:+.2f}%)")
        lines.append("")
    
    return '\n'.join(lines)


def format_headlines(headlines: list) -> str:
    """Format headlines for the prompt."""
    lines = ["## Schlagzeilen\n"]
    
    for article in headlines[:15]:
        source = article.get('source', 'Unknown')
        title = article.get('title', '')
        lines.append(f"- [{source}] {title}")
    
    return '\n'.join(lines)


def format_portfolio_news(portfolio_data: dict) -> str:
    """Format portfolio news for the prompt."""
    lines = ["## Portfolio Nachrichten\n"]
    
    for symbol, data in portfolio_data.get('stocks', {}).items():
        quote = data.get('quote', {})
        price = quote.get('price', 'N/A')
        change_pct = quote.get('change_percent', 0)
        
        lines.append(f"### {symbol} (${price}, {change_pct:+.2f}%)")
        
        for article in data.get('articles', [])[:3]:
            lines.append(f"- {article.get('title', '')}")
        lines.append("")
    
    return '\n'.join(lines)


def generate_briefing(args):
    """Generate full market briefing."""
    config = load_config()
    language = args.lang or config['language']['default']
    
    # Fetch fresh data
    print("📡 Fetching market data...", file=sys.stderr)
    
    # Get market overview
    market_result = subprocess.run(
        ['python3', str(SCRIPT_DIR / 'fetch_news.py'), 'market', '--json', '--limit', '3'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    market_data = {}
    if market_result.returncode == 0:
        try:
            market_data = json.loads(market_result.stdout)
        except json.JSONDecodeError:
            pass
    
    # Get portfolio news (limit to 5 stocks max for performance)
    portfolio_result = subprocess.run(
        ['python3', str(SCRIPT_DIR / 'fetch_news.py'), 'portfolio', '--json', '--limit', '2', '--max-stocks', '5'],
        capture_output=True,
        text=True,
        timeout=90
    )
    
    portfolio_data = {}
    if portfolio_result.returncode == 0:
        try:
            portfolio_data = json.loads(portfolio_result.stdout)
        except json.JSONDecodeError:
            pass
    
    # Build content for summarization
    content_parts = []
    
    if market_data:
        content_parts.append(format_market_data(market_data))
        if market_data.get('headlines'):
            content_parts.append(format_headlines(market_data['headlines']))
    
    if portfolio_data:
        content_parts.append(format_portfolio_news(portfolio_data))
    
    content = '\n\n'.join(content_parts)
    
    if not content.strip():
        print("⚠️ No data available for briefing", file=sys.stderr)
        return
    
    print("🤖 Generating AI summary...", file=sys.stderr)
    
    # Generate summary
    summary = summarize_with_gemini(content, language, args.style)
    
    # Format output
    time_str = datetime.now().strftime("%H:%M")
    date_str = datetime.now().strftime("%A, %d. %B %Y")
    
    if args.time == "morning":
        emoji = "🌅"
        title = "Morgen-Briefing"
    else:
        emoji = "🌆"
        title = "Abend-Briefing"
    
    output = f"""{emoji} **Börsen-{title}**
{date_str} | {time_str} Uhr

{summary}
"""
    
    if args.json:
        print(json.dumps({
            'title': f"Börsen-{title}",
            'date': date_str,
            'time': time_str,
            'language': language,
            'summary': summary,
            'raw_data': {
                'market': market_data,
                'portfolio': portfolio_data
            }
        }, indent=2, ensure_ascii=False))
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(description='News Summarizer')
    parser.add_argument('--lang', choices=['de', 'en'], help='Output language')
    parser.add_argument('--style', choices=['briefing', 'analysis', 'headlines'], 
                        default='briefing', help='Summary style')
    parser.add_argument('--time', choices=['morning', 'evening'], 
                        default='morning', help='Briefing type')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    generate_briefing(args)


if __name__ == '__main__':
    main()
