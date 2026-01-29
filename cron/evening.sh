#!/usr/bin/env bash
# Evening Briefing Cron Job
# Schedule: 1:00 PM PT (US Market Close at 4:00 PM ET)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date)] Starting evening briefing..."

# Run directly on host with Python 3.14 packages from pybox
export PYTHONPATH="/home/art/.local/lib/python3.14/site-packages:$PYTHONPATH"
python3 "$SCRIPT_DIR/scripts/real-briefing.py" evening

echo "[$(date)] Evening briefing complete."
