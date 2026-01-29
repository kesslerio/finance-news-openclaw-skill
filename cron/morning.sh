#!/usr/bin/env bash
# Morning Briefing Cron Job
# Schedule: 6:30 AM PT (US Market Open at 9:30 AM ET)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date)] Starting morning briefing..."

# Run directly on host with Python 3.14 packages from pybox
export PYTHONPATH="/home/art/.local/lib/python3.14/site-packages:$PYTHONPATH"
python3 "$SCRIPT_DIR/scripts/real-briefing.py" morning

echo "[$(date)] Morning briefing complete."
