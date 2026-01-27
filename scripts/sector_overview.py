#!/usr/bin/env python3
"""
Weekly Sector Overview - Report sector performance using SPDR ETFs.

Schedule: Saturday morning (cron: 0 9 * * 6)
Delivery: WhatsApp group "Niemand Börse"

Usage:
    sector_overview.py              # Generate and print report
    sector_overview.py --send       # Also send to WhatsApp
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

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

# Get openbb binary
OPENBB_BINARY = None
try:
    env_path = os.environ.get('OPENBB_QUOTE_BIN')
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        OPENBB_BINARY = env_path
    else:
        OPENBB_BINARY = shutil.which('openbb-quote')
except Exception:
    pass


def get_sector_data() -> dict:
    """Fetch current price for all sector ETFs."""
    results = {}
    symbols = list(SECTOR_ETFS.keys())

    if not OPENBB_BINARY:
        print("⚠️ openbb-quote not found", file=sys.stderr)
        return results

    # Fetch all at once if supported, otherwise one by one
    for symbol in symbols:
        try:
            result = subprocess.run(
                [OPENBB_BINARY, symbol],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        # Normalize data structure
                        if "results" in data and isinstance(data["results"], list):
                            data = data["results"][0] if data["results"] else {}
                        if "change_percent" not in data and "price" in data and "prev_close" in data:
                            if data["prev_close"]:
                                data["change_percent"] = ((data["price"] - data["prev_close"]) / data["prev_close"]) * 100
                        results[symbol] = data
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    return results


def get_sp500_change() -> float:
    """Get S&P 500 weekly change for context."""
    if not OPENBB_BINARY:
        return 0.0

    try:
        result = subprocess.run(
            [OPENBB_BINARY, "^GSPC"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                if "results" in data and isinstance(data["results"], list):
                    data = data["results"][0] if data["results"] else {}
                return data.get("change_percent", 0.0)
    except Exception:
        pass
    return 0.0


def format_sector_overview() -> str:
    """Generate sector overview message."""
    data = get_sector_data()
    if not data:
        return "⚠️ No sector data available"

    sp500_change = get_sp500_change()

    # Build list of (symbol, change_pct, name)
    sectors = []
    for symbol, name in SECTOR_ETFS.items():
        if symbol in data:
            change = data[symbol].get("change_percent", 0.0)
            price = data[symbol].get("price", 0.0)
            sectors.append((symbol, change, name, price))

    # Sort by change (best to worst)
    sectors.sort(key=lambda x: x[1], reverse=True)

    # Find cutoff for "top" and "bottom"
    top_3 = sectors[:3]
    bottom_3 = sectors[-3:]

    # Middle sectors (indices 3-7, if available)
    middle = []
    if len(sectors) > 6:
        middle = sectors[3:len(sectors)-3]

    week_start = (datetime.now() - timedelta(days=7)).strftime("%d.%m")
    week_end = datetime.now().strftime("%d.%m")

    lines = [
        f"📊 **Sektor-Übersicht** (Woche {week_start}-{week_end})",
        "",
        f"S&P 500: {sp500_change:+.2f}%",
        "",
        "🟢 **Top Performer:**",
    ]

    for symbol, change, name, price in top_3:
        emoji = "📈" if change >= 0 else "📉"
        lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    lines.append("")
    lines.append("🔴 **Schwächste Performer:**")

    for symbol, change, name, price in reversed(bottom_3):
        emoji = "📈" if change >= 0 else "📉"
        lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    if middle:
        lines.append("")
        lines.append("⚪ **Mittelfeld:**")
        for symbol, change, name, price in middle:
            emoji = "📈" if change >= 0 else "📉"
            lines.append(f"{emoji} {name} ({symbol}): {change:+.2f}%")

    return "\n".join(lines)


def send_to_whatsapp(message: str, group_name: str = "Niemand Börse"):
    """Send message to WhatsApp."""
    try:
        result = subprocess.run(
            ['clawdbot', 'message', 'send',
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


def main():
    parser = argparse.ArgumentParser(description="Weekly Sector Overview")
    parser.add_argument('--send', action='store_true', help='Send to WhatsApp')
    parser.add_argument('--group', default='Niemand Börse', help='WhatsApp group name')
    args = parser.parse_args()

    message = format_sector_overview()

    if args.send:
        send_to_whatsapp(message, args.group)
    else:
        print(message)


if __name__ == '__main__':
    main()
