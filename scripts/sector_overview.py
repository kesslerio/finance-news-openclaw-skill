#!/usr/bin/env python3
"""
Weekly Sector Overview - Report sector performance using SPDR ETFs.

Schedule: Saturday morning (cron: 0 9 * * 6)
Delivery: WhatsApp target configured with FINANCE_NEWS_TARGET

Usage:
    finance-news sector-overview              # Generate and print report
    finance-news sector-overview --send       # Also send to WhatsApp
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

# Sector SPDR ETFs
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLV": "Healthcare",
    "XLP": "Consumer Staples",
}

def _download_history(symbols: list[str]):
    """Fetch daily closes for one month so a full trading week is available."""
    import yfinance as yf

    return yf.download(
        " ".join(symbols),
        period="1mo",
        interval="1d",
        progress=False,
        threads=True,
        ignore_tz=True,
        timeout=30,
    )


def _normalize_weekly_market_data(history, symbols: list[str]) -> dict[str, float]:
    if history is None or history.empty:
        return {}

    results = {}
    for symbol in symbols:
        frame = None
        if getattr(history.columns, "nlevels", 1) > 1:
            for level in (1, 0):
                try:
                    frame = history.xs(symbol, level=level, axis=1, drop_level=True)
                    break
                except (AttributeError, KeyError):
                    continue
        elif len(symbols) == 1:
            frame = history

        if frame is None or "Close" not in frame.columns:
            continue
        closes = frame["Close"].dropna()
        if len(closes) < 2:
            continue

        # Six closes span five trading sessions.
        previous_close = closes.iloc[-6] if len(closes) >= 6 else closes.iloc[0]
        latest_close = closes.iloc[-1]
        try:
            previous_close = float(previous_close)
            latest_close = float(latest_close)
        except (TypeError, ValueError):
            continue
        if previous_close <= 0:
            continue

        results[symbol] = ((latest_close - previous_close) / previous_close) * 100
    return results


def get_weekly_market_data() -> dict[str, float]:
    """Fetch five-session performance for sector ETFs and the S&P 500."""
    symbols = [*SECTOR_ETFS, "^GSPC"]
    try:
        history = _download_history(symbols)
    except Exception as exc:
        print(f"⚠️ Weekly market data unavailable: {exc}", file=sys.stderr)
        return {}
    return _normalize_weekly_market_data(history, symbols)


def format_sector_overview() -> str:
    """Generate sector overview message."""
    market_data = get_weekly_market_data()
    sector_data = {symbol: market_data[symbol] for symbol in SECTOR_ETFS if symbol in market_data}
    if not sector_data:
        return "⚠️ No sector data available"

    sp500_change = market_data.get("^GSPC", 0.0)

    # Build list of (symbol, change_pct, name)
    sectors = []
    for symbol, name in SECTOR_ETFS.items():
        if symbol in sector_data:
            sectors.append((symbol, sector_data[symbol], name))

    # Sort by change (best to worst)
    sectors.sort(key=lambda x: x[1], reverse=True)

    # Find cutoff for "top" and "bottom"
    top_3 = sectors[:3]
    bottom_start = max(3, len(sectors) - 3)
    bottom_3 = sectors[bottom_start:]

    middle = sectors[3:bottom_start]

    report_date = datetime.now()
    week_start = (report_date - timedelta(days=7)).strftime("%d.%m")
    week_end = report_date.strftime("%d.%m")

    lines = [
        f"📊 **Sektor-Übersicht** (Woche {week_start}-{week_end})",
        "",
        f"S&P 500: {sp500_change:+.2f}%",
        "",
        "🟢 **Top Performer:**",
    ]

    for symbol, change, name in top_3:
        emoji = "📈" if change >= 0 else "📉"
        lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    if bottom_3:
        lines.append("")
        lines.append("🔴 **Schwächste Performer:**")

        for symbol, change, name in reversed(bottom_3):
            emoji = "📈" if change >= 0 else "📉"
            lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    if middle:
        lines.append("")
        lines.append("⚪ **Mittelfeld:**")
        for symbol, change, name in middle:
            emoji = "📈" if change >= 0 else "📉"
            lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    return "\n".join(lines)


def send_to_whatsapp(message: str, group_name: str | None = None) -> bool:
    """Send message to WhatsApp."""
    group_name = group_name or os.environ.get("FINANCE_NEWS_TARGET", "")
    if not group_name:
        print("❌ No target specified. Set FINANCE_NEWS_TARGET or use --group", file=sys.stderr)
        return False

    try:
        result = subprocess.run(
            ['openclaw', 'message', 'send',
             '--channel', 'whatsapp',
             '--target', group_name,
             '--message', message],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"✅ Sent to WhatsApp: {group_name}")
            return True
        else:
            print(f"⚠️ Send failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ WhatsApp error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly Sector Overview")
    parser.add_argument('--send', action='store_true', help='Send to WhatsApp')
    parser.add_argument(
        '--group',
        default=os.environ.get('FINANCE_NEWS_TARGET', ''),
        help='WhatsApp group name or JID (default: FINANCE_NEWS_TARGET env var)',
    )
    args = parser.parse_args()

    message = format_sector_overview()

    if args.send:
        return 0 if send_to_whatsapp(message, args.group) else 1
    else:
        print(message)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
