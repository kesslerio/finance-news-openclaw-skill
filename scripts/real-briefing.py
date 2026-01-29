#!/usr/bin/env python3
"""Briefing with real headlines - German format with correct percentages."""

import json
import subprocess
import sys
import urllib.request
from datetime import datetime

PORTFOLIO = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NOW"]
INDICES = ["SPY", "QQQ", "DIA"]  # S&P 500, Nasdaq, Dow
RSS_FEEDS = [
    ("https://feeds.a.dj.com/RSSNewsMarkup.xml", "WSJ"),
    ("https://fe.reuters.com/reuk/finance/rss", "Reuters"),
    ("https://feeds.finance.yahoo.com/rss/2.0/headline", "Yahoo"),
]

def get_weekday_german():
    """Get German weekday name."""
    weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    return weekdays[datetime.now().weekday()]

def fetch_rss(url, max_items=3):
    """Fetch headlines from RSS feed."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read()
        
        import feedparser
        parsed = feedparser.parse(content)
        headlines = []
        for entry in parsed.entries[:max_items]:
            title = entry.get('title', '').strip()
            link = entry.get('link', '').strip()
            if title and link and not title.startswith('<'):
                headlines.append({'title': title, 'link': link})
        return headlines
    except Exception as e:
        return []

def get_quote(symbol):
    """Get quote from yfinance using previousClose for correct daily change."""
    cmd = [
        sys.executable, "-c",
        f"""
import yfinance as yf
import json
ticker = yf.Ticker("{symbol}")
hist = ticker.history(period='1d')
info = ticker.info
if len(hist) > 0:
    price = hist['Close'].iloc[-1]
    prev_close = info.get('previousClose')
    if prev_close and prev_close > 0:
        change_pct = ((price - prev_close) / prev_close) * 100
    else:
        change_pct = 0
    print(json.dumps({{
        'symbol': '{symbol}', 
        'price': price, 
        'change_pct': change_pct,
        'company': info.get('shortName', info.get('companyName', ''))
    }}))
else:
    print('null')
"""
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except:
        return None

def shorten_url(url):
    """Shorten URL using is.gd."""
    try:
        api = f"https://is.gd/create.php?format=json&url={urllib.parse.quote(url)}"
        with urllib.request.urlopen(api, timeout=10) as response:
            data = json.loads(response.read())
            if 'shorturl' in data:
                return data['shorturl']
            return url
    except Exception as e:
        return url

def format_briefing(time_of_day="morning"):
    """Format briefing in German."""
    now = datetime.now()
    
    # German date format
    date_str = now.strftime("%d. %B %Y")
    date_str = date_str.replace('January', 'Januar').replace('February', 'Februar')
    date_str = date_str.replace('March', 'März').replace('April', 'April')
    date_str = date_str.replace('May', 'Mai').replace('June', 'Juni')
    date_str = date_str.replace('July', 'Juli').replace('August', 'August')
    date_str = date_str.replace('September', 'September').replace('October', 'Oktober')
    date_str = date_str.replace('November', 'November').replace('December', 'Dezember')
    time_str = now.strftime("%H:%M")
    title = "Börsen Morgen-Briefing" if time_of_day == "morning" else "Börsen Abend-Briefing"
    
    lines = [f"🌅 {title}", f"{get_weekday_german()}, {date_str} | {time_str} Uhr\n"]
    
    # Market indices (SPY, QQQ, DIA)
    lines.append("### Marktübersicht")
    for symbol in INDICES:
        data = get_quote(symbol)
        if data:
            change = data.get('change_pct', 0)
            e = "📈" if change >= 0 else "📉"
            lines.append(f"{e} {symbol}: {change:+.2f}%")
    
    # Fetch headlines
    all_headlines = []
    for url, source in RSS_FEEDS:
        headlines = fetch_rss(url, max_items=2)
        for h in headlines:
            all_headlines.append({**h, 'source': source})
    
    # Top Schlagzeilen (numbered with source)
    lines.append("\n### Top Schlagzeilen")
    for i, h in enumerate(all_headlines[:5], 1):
        title_clean = h['title'].replace('*', '').replace('_', '').replace('#', '')
        lines.append(f"{i}. {title_clean} [{h['source']}]")
    
    # Portfolio with company names
    lines.append("\n### Portfolio-Auswirkung")
    portfolio_data = []
    for symbol in PORTFOLIO:
        data = get_quote(symbol)
        if data:
            portfolio_data.append(data)
    
    portfolio_data.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
    for p in portfolio_data:
        sym = p.get('symbol', '')
        change = p.get('change_pct', 0)
        company = p.get('company', '')
        e = "📈" if change >= 0 else "📉"
        company_str = f" ({company})" if company else ""
        lines.append(f"{e} {sym}: {change:+.2f}%{company_str}")
    
    # Heute wichtig section
    lines.append("\nHeute wichtig: Fed-Entscheidung + Big Tech Earnings")
    
    # Sources (numbered links)
    lines.append("\nQuellen:")
    for i, h in enumerate(all_headlines[:5], 1):
        short_link = shorten_url(h['link'])
        lines.append(f"[{i}]({short_link})")
    
    return "\n".join(lines)

def send_to_whatsapp(message):
    """Send to WhatsApp via moltbot."""
    result = subprocess.run([
        'moltbot', 'message', 'send',
        '--channel', 'whatsapp',
        '--target', '120363421796203667@g.us',
        '--message', message
    ], capture_output=True, text=True)
    return result.returncode == 0, result.stderr

def main():
    time_of_day = "morning"
    if len(sys.argv) > 1 and sys.argv[1] == "evening":
        time_of_day = "evening"
    
    print("📊 Fetching market data...")
    message = format_briefing(time_of_day)
    
    print("📤 Sending to WhatsApp...")
    success, err = send_to_whatsapp(message)
    if success:
        print("✅ Sent to WhatsApp: Niemand Boerse")
    else:
        print(f"❌ WhatsApp send failed: {err}")

if __name__ == "__main__":
    main()
