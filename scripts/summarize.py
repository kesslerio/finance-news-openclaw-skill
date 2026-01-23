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

from fetch_news import PortfolioError, get_market_news, get_portfolio_news
from research import generate_research_content

SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
DEFAULT_PORTFOLIO_SAMPLE_SIZE = 3
MAX_HEADLINES_IN_PROMPT = 10

LANG_PROMPTS = {
    "de": "Output must be in German only.",
    "en": "Output must be in English only."
}

STYLE_PROMPTS = {
    "briefing": """You are a financial analyst. Create a concise market briefing.

IMPORTANT:
- Use only the provided market data and headlines.
- No speculation, no invented numbers, no external facts.
- If information is missing, say clearly: "Keine Daten verfügbar".
- Follow the language constraint exactly.

Structure:
1) Market sentiment (bullish/bearish/neutral) with a short rationale from the data
2) Top 3 headlines as a numbered list with source tags in brackets
3) Portfolio impact (only if portfolio data exists)
4) Short action recommendation

Max 200 words. Use emojis sparingly.""",

    "analysis": """You are an experienced financial analyst.
Analyze the news and provide:
- Detailed market analysis
- Sector trends
- Risks and opportunities
- Concrete recommendations

Be professional but clear.""",

    "headlines": """Summarize the most important headlines in 5 bullet points.
Each bullet must be at most 15 words."""
}


def load_config():
    """Load source configuration."""
    with open(CONFIG_DIR / "sources.json", 'r') as f:
        return json.load(f)


def extract_agent_reply(raw: str) -> str:
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if isinstance(data, dict):
        for key in ("reply", "message", "text", "output", "result"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
        if "messages" in data:
            messages = data.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    text = last.get("text") or last.get("message")
                    if isinstance(text, str):
                        return text.strip()

    return raw.strip()


def summarize_with_claude(content: str, language: str = "de", style: str = "briefing") -> str:
    """Generate AI summary using Claude via Clawdbot agent."""
    prompt = f"""{STYLE_PROMPTS.get(style, STYLE_PROMPTS['briefing'])}

{LANG_PROMPTS.get(language, LANG_PROMPTS['de'])}

Use only the following information for the briefing:

{content}
"""

    try:
        result = subprocess.run(
            [
                'clawdbot', 'agent',
                '--session-id', 'finance-news-briefing',
                '--message', prompt,
                '--json',
                '--timeout', '120'
            ],
            capture_output=True,
            text=True,
            timeout=150
        )
    except subprocess.TimeoutExpired:
        return "⚠️ Claude briefing error: timeout"
    except FileNotFoundError:
        return "⚠️ Claude briefing error: clawdbot CLI not found"
    except OSError as exc:
        return f"⚠️ Claude briefing error: {exc}"

    if result.returncode == 0:
        return extract_agent_reply(result.stdout)

    stderr = result.stderr.strip() or "unknown error"
    return f"⚠️ Claude briefing error: {stderr}"


def summarize_with_minimax(content: str, language: str = "de", style: str = "briefing") -> str:
    """Generate AI summary using MiniMax model via clawdbot agent."""
    prompt = f"""{STYLE_PROMPTS.get(style, STYLE_PROMPTS['briefing'])}

{LANG_PROMPTS.get(language, LANG_PROMPTS['de'])}

Use only the following information for the briefing:

{content}
"""

    try:
        result = subprocess.run(
            [
                'clawdbot', 'agent',
                '--session-id', 'finance-news-briefing',
                '--message', prompt,
                '--model', 'minimax',
                '--json',
                '--timeout', '120'
            ],
            capture_output=True,
            text=True,
            timeout=150
        )
    except subprocess.TimeoutExpired:
        return "⚠️ MiniMax briefing error: timeout"
    except FileNotFoundError:
        return "⚠️ MiniMax briefing error: clawdbot CLI not found"
    except OSError as exc:
        return f"⚠️ MiniMax briefing error: {exc}"

    if result.returncode == 0:
        return extract_agent_reply(result.stdout)

    stderr = result.stderr.strip() or "unknown error"
    return f"⚠️ MiniMax briefing error: {stderr}"


def summarize_with_gemini(content: str, language: str = "de", style: str = "briefing") -> str:
    """Generate AI summary using Gemini CLI."""
    
    prompt = f"""{STYLE_PROMPTS.get(style, STYLE_PROMPTS['briefing'])}

{LANG_PROMPTS.get(language, LANG_PROMPTS['de'])}

Here are the current market items:

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

    for article in headlines[:MAX_HEADLINES_IN_PROMPT]:
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


def classify_sentiment(market_data: dict) -> str:
    changes = []
    for region in market_data.get("markets", {}).values():
        for idx in region.get("indices", {}).values():
            data = idx.get("data") or {}
            change = data.get("change_percent")
            if isinstance(change, (int, float)):
                changes.append(change)
                continue

            price = data.get("price")
            prev_close = data.get("prev_close")
            if isinstance(price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close != 0:
                changes.append(((price - prev_close) / prev_close) * 100)

    if not changes:
        return "Keine Daten verfügbar"

    avg = sum(changes) / len(changes)
    if avg >= 0.5:
        return "Bullisch"
    if avg <= -0.5:
        return "Bärisch"
    return "Neutral"


def build_briefing_summary(market_data: dict, portfolio_data: dict | None) -> str:
    sentiment = classify_sentiment(market_data)
    headlines = market_data.get("headlines", [])[:3]

    lines = ["## Marktbriefing", "", f"### Sentiment: {sentiment}"]

    lines.append("")
    lines.append("### Top 3 Schlagzeilen")
    if headlines:
        for idx, article in enumerate(headlines, start=1):
            source = article.get("source", "Unknown")
            title = article.get("title", "").strip()
            lines.append(f"{idx}. {title} [{source}]")
    else:
        lines.append("Keine Daten verfügbar")

    lines.append("")
    lines.append("### Portfolio-Auswirkungen")
    if portfolio_data:
        for symbol, data in portfolio_data.get("stocks", {}).items():
            quote = data.get("quote") or {}
            change = quote.get("change_percent")
            if isinstance(change, (int, float)):
                lines.append(f"- **{symbol}**: {change:+.2f}%")
            else:
                lines.append(f"- **{symbol}**: Keine Kursdaten")
    else:
        lines.append("Keine Daten verfügbar")

    lines.append("")
    lines.append("### Empfehlung")
    if sentiment == "Bullisch":
        lines.append("Chancen selektiv nutzen, aber Risikomanagement beibehalten.")
    elif sentiment == "Bärisch":
        lines.append("Risiken reduzieren und Liquidität priorisieren.")
    elif sentiment == "Neutral":
        lines.append("Abwarten und Fokus auf Qualitätstitel.")
    else:
        lines.append("Keine klare Empfehlung ohne belastbare Daten.")

    return "\n".join(lines)


def generate_briefing(args):
    """Generate full market briefing."""
    config = load_config()
    language = args.lang or config['language']['default']
    
    # Fetch fresh data
    print("📡 Fetching market data...", file=sys.stderr)
    
    # Get market overview
    market_data = get_market_news(
        3,
        regions=["us", "europe"],
        max_indices_per_region=1
    )
    
    # Get portfolio news (limit stocks for performance)
    try:
        portfolio_data = get_portfolio_news(2, DEFAULT_PORTFOLIO_SAMPLE_SIZE)
    except PortfolioError as exc:
        print(f"⚠️ Skipping portfolio: {exc}", file=sys.stderr)
        portfolio_data = None
    
    # Build raw content for summarization
    content_parts = []

    if market_data:
        content_parts.append(format_market_data(market_data))
        if market_data.get('headlines'):
            content_parts.append(format_headlines(market_data['headlines']))

    # Only include portfolio if fetch succeeded (no error key)
    if portfolio_data:
        content_parts.append(format_portfolio_news(portfolio_data))

    raw_content = '\n\n'.join(content_parts)

    if not raw_content.strip():
        print("⚠️ No data available for briefing", file=sys.stderr)
        return

    if not market_data.get('headlines'):
        print("⚠️ No headlines available; skipping summary generation", file=sys.stderr)
        return

    research_report = ''
    source = 'none'
    if args.research:
        research_result = generate_research_content(market_data, portfolio_data)
        research_report = research_result['report']
        source = research_result['source']

    if research_report.strip():
        content = f"""# Research Report ({source})

{research_report}

# Raw Market Data

{raw_content}
"""
    else:
        content = raw_content

    model = getattr(args, 'model', 'claude')

    if args.style == "briefing" and not args.llm:
        summary = build_briefing_summary(market_data, portfolio_data)
    else:
        print(f"🤖 Generating AI summary with {model}...", file=sys.stderr)

        # Generate summary based on selected model
        if model == 'minimax':
            summary = summarize_with_minimax(content, language, args.style)
            if summary.startswith("⚠️ MiniMax briefing error"):
                print(summary, file=sys.stderr)
                print("⚠️ MiniMax failed; falling back to Claude...", file=sys.stderr)
                summary = summarize_with_claude(content, language, args.style)
                if summary.startswith("⚠️ Claude briefing error"):
                    print(summary, file=sys.stderr)
                    print("⚠️ Claude also failed; falling back to Gemini...", file=sys.stderr)
                    summary = summarize_with_gemini(content, language, args.style)
        elif model == 'gemini':
            summary = summarize_with_gemini(content, language, args.style)
        else:  # claude (default)
            summary = summarize_with_claude(content, language, args.style)
            if summary.startswith("⚠️ Claude briefing error"):
                print(summary, file=sys.stderr)
                print("⚠️ Claude failed; falling back to Gemini summarizer", file=sys.stderr)
                summary = summarize_with_gemini(content, language, args.style)
    
    # Format output
    time_str = datetime.now().strftime("%H:%M")
    date_str = datetime.now().strftime("%A, %d. %B %Y")
    
    if args.time == "morning":
        emoji = "🌅"
        title = "Morgen-Briefing"
    elif args.time == "evening":
        emoji = "🌆"
        title = "Abend-Briefing"
    else:
        hour = datetime.now().hour
        emoji = "🌅" if hour < 12 else "🌆"
        title = "Morgen-Briefing" if hour < 12 else "Abend-Briefing"
    
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
                        default=None, help='Briefing type (default: auto)')
    parser.add_argument('--model', choices=['claude', 'minimax', 'gemini'],
                        default='claude', help='AI model for summarization')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--research', action='store_true', help='Include deep research section (slower)')
    parser.add_argument('--llm', action='store_true', help='Use LLM for briefing (default: deterministic)')

    args = parser.parse_args()
    generate_briefing(args)


if __name__ == '__main__':
    main()
