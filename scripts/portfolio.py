#!/usr/bin/env python3
"""
Portfolio Manager - CRUD operations for stock watchlist.
"""

import argparse
import csv
import sys
from pathlib import Path

PORTFOLIO_FILE = Path(__file__).parent.parent / "config" / "portfolio.csv"


def load_portfolio() -> list[dict]:
    """Load portfolio from CSV."""
    if not PORTFOLIO_FILE.exists():
        return []
    
    with open(PORTFOLIO_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_portfolio(portfolio: list[dict]):
    """Save portfolio to CSV."""
    if not portfolio:
        PORTFOLIO_FILE.write_text("symbol,name,category,notes,type\n")
        return
    
    with open(PORTFOLIO_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['symbol', 'name', 'category', 'notes', 'type'])
        writer.writeheader()
        writer.writerows(portfolio)


def list_portfolio(args):
    """List all stocks in portfolio."""
    portfolio = load_portfolio()
    
    if not portfolio:
        print("📂 Portfolio is empty. Use 'portfolio add <SYMBOL>' to add stocks.")
        return
    
    print(f"\n📊 Portfolio ({len(portfolio)} stocks)\n")
    
    # Group by Type then Category
    by_type = {}
    for stock in portfolio:
        t = stock.get('type', 'Watchlist') or 'Watchlist'
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(stock)
        
    for t, type_stocks in by_type.items():
        print(f"# {t}")
        categories = {}
        for stock in type_stocks:
            cat = stock.get('category', 'Other') or 'Other'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(stock)
        
        for cat, stocks in categories.items():
            print(f"### {cat}")
            for s in stocks:
                notes = f" — {s['notes']}" if s.get('notes') else ""
                print(f"  • {s['symbol']}: {s['name']}{notes}")
            print()


def add_stock(args):
    """Add a stock to portfolio."""
    portfolio = load_portfolio()
    
    # Check if already exists
    if any(s['symbol'].upper() == args.symbol.upper() for s in portfolio):
        print(f"⚠️ {args.symbol.upper()} already in portfolio")
        return
    
    new_stock = {
        'symbol': args.symbol.upper(),
        'name': args.name or args.symbol.upper(),
        'category': args.category or '',
        'notes': args.notes or '',
        'type': args.type
    }
    
    portfolio.append(new_stock)
    save_portfolio(portfolio)
    print(f"✅ Added {args.symbol.upper()} to portfolio ({args.type})")


def remove_stock(args):
    """Remove a stock from portfolio."""
    portfolio = load_portfolio()
    
    original_len = len(portfolio)
    portfolio = [s for s in portfolio if s['symbol'].upper() != args.symbol.upper()]
    
    if len(portfolio) == original_len:
        print(f"⚠️ {args.symbol.upper()} not found in portfolio")
        return
    
    save_portfolio(portfolio)
    print(f"✅ Removed {args.symbol.upper()} from portfolio")


def import_csv(args):
    """Import portfolio from external CSV."""
    import_path = Path(args.file)
    
    if not import_path.exists():
        print(f"❌ File not found: {args.file}")
        sys.exit(1)
    
    with open(import_path, 'r') as f:
        reader = csv.DictReader(f)
        imported = list(reader)
    
    # Normalize fields
    normalized = []
    for row in imported:
        normalized.append({
            'symbol': row.get('symbol', row.get('Symbol', row.get('ticker', ''))).upper(),
            'name': row.get('name', row.get('Name', row.get('company', ''))),
            'category': row.get('category', row.get('Category', row.get('sector', ''))),
            'notes': row.get('notes', row.get('Notes', '')),
            'type': row.get('type', 'Watchlist')
        })
    
    save_portfolio(normalized)
    print(f"✅ Imported {len(normalized)} stocks from {args.file}")


def create_interactive(args):
    """Interactive portfolio creation."""
    print("\n📊 Portfolio Creator\n")
    print("Enter stocks one per line (format: SYMBOL or SYMBOL,Name,Category)")
    print("Type 'done' when finished.\n")
    
    portfolio = []
    
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if line.lower() == 'done':
            break
        
        if not line:
            continue
        
        parts = line.split(',')
        symbol = parts[0].strip().upper()
        name = parts[1].strip() if len(parts) > 1 else symbol
        category = parts[2].strip() if len(parts) > 2 else ''
        
        portfolio.append({
            'symbol': symbol,
            'name': name,
            'category': category,
            'notes': '',
            'type': 'Watchlist'
        })
        print(f"  Added: {symbol}")
    
    if portfolio:
        save_portfolio(portfolio)
        print(f"\n✅ Created portfolio with {len(portfolio)} stocks")
    else:
        print("\n⚠️ No stocks added")


def get_symbols(args=None):
    """Get list of symbols (for other scripts to use)."""
    portfolio = load_portfolio()
    symbols = [s['symbol'] for s in portfolio]
    
    if args and args.json:
        import json
        print(json.dumps(symbols))
    else:
        print(','.join(symbols))


def main():
    parser = argparse.ArgumentParser(description='Portfolio Manager')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List portfolio')
    list_parser.set_defaults(func=list_portfolio)
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add stock')
    add_parser.add_argument('symbol', help='Stock symbol')
    add_parser.add_argument('--name', help='Company name')
    add_parser.add_argument('--category', help='Category (e.g., Tech, Finance)')
    add_parser.add_argument('--notes', help='Notes')
    add_parser.add_argument('--type', choices=['Holding', 'Watchlist'], default='Watchlist', help='Portfolio type')
    add_parser.set_defaults(func=add_stock)
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove stock')
    remove_parser.add_argument('symbol', help='Stock symbol')
    remove_parser.set_defaults(func=remove_stock)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import from CSV')
    import_parser.add_argument('file', help='CSV file path')
    import_parser.set_defaults(func=import_csv)
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Interactive creation')
    create_parser.set_defaults(func=create_interactive)
    
    # Symbols command (for other scripts)
    symbols_parser = subparsers.add_parser('symbols', help='Get symbols list')
    symbols_parser.add_argument('--json', action='store_true', help='Output as JSON')
    symbols_parser.set_defaults(func=get_symbols)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
