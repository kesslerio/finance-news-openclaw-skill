#!/usr/bin/env python3
"""
Briefing Generator - Main entry point for market briefings.
Generates and optionally sends to WhatsApp group.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def send_to_whatsapp(message: str, group_name: str = "Niemand Boerse"):
    """Send message to WhatsApp group via Clawdbot message tool."""
    # Use clawdbot message tool
    try:
        result = subprocess.run(
            [
                'clawdbot', 'message', 'send',
                '--channel', 'whatsapp',
                '--target', group_name,
                '--message', message
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ Sent to WhatsApp group: {group_name}", file=sys.stderr)
            return True
        else:
            print(f"⚠️ WhatsApp send failed: {result.stderr}", file=sys.stderr)
            return False
    
    except Exception as e:
        print(f"❌ WhatsApp error: {e}", file=sys.stderr)
        return False


def generate_and_send(args):
    """Generate briefing and optionally send to WhatsApp."""
    
    # Determine briefing type based on current time or args
    if args.time:
        briefing_time = args.time
    else:
        hour = datetime.now().hour
        briefing_time = 'morning' if hour < 12 else 'evening'
    
    # Generate the briefing
    cmd = [
        'python3', SCRIPT_DIR / 'summarize.py',
        '--time', briefing_time,
        '--style', args.style,
        '--lang', args.lang
    ]
    
    # Pass --json flag if requested
    if args.json:
        cmd.append('--json')
    
    print(f"📊 Generating {briefing_time} briefing...", file=sys.stderr)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300
    )
    
    if result.returncode != 0:
        print(f"❌ Briefing generation failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    
    briefing = result.stdout.strip()
    
    # Print to stdout
    print(briefing)
    
    # Send to WhatsApp if requested
    if args.send and args.group:
        if args.json:
            # Parse JSON and send summary only
            try:
                data = json.loads(briefing)
                message = data.get('summary', '')
                if message:
                    send_to_whatsapp(message, args.group)
                else:
                    print(f"⚠️ No summary field in JSON output", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"⚠️ Cannot parse JSON for WhatsApp send", file=sys.stderr)
        else:
            send_to_whatsapp(briefing, args.group)
    
    return briefing


def main():
    parser = argparse.ArgumentParser(description='Briefing Generator')
    parser.add_argument('--time', choices=['morning', 'evening'], 
                        help='Briefing type (auto-detected if not specified)')
    parser.add_argument('--style', choices=['briefing', 'analysis', 'headlines'],
                        default='briefing', help='Summary style')
    parser.add_argument('--lang', choices=['en', 'de'], default='en',
                        help='Output language')
    parser.add_argument('--send', action='store_true',
                        help='Send to WhatsApp group')
    parser.add_argument('--group', default='Niemand Boerse',
                        help='WhatsApp group name')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    
    args = parser.parse_args()
    generate_and_send(args)


if __name__ == '__main__':
    main()
