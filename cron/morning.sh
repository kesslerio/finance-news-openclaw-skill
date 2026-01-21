#!/usr/bin/env bash
# Morning Briefing Cron Job
# Schedule: 6:30 AM PT (US Market Open at 9:30 AM ET)
# Target: WhatsApp group (configure --group flag)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date)] Starting morning briefing..."

python3 "$SCRIPT_DIR/scripts/briefing.py" \
    --time morning \
    --style briefing \
    --lang en \
    --send \
    --group "Market Briefing"

echo "[$(date)] Morning briefing complete."
