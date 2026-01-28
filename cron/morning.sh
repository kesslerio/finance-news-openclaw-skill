#!/usr/bin/env bash
# Morning Briefing Cron Job (Docker-based)
# Schedule: 6:30 AM PT (US Market Open at 9:30 AM ET)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date)] Starting morning briefing..."

# Use Docker to avoid NixOS libstdc++ issues
# Mount config/ so runtime changes (portfolio, sources) are picked up
cd "$SCRIPT_DIR"
docker run --rm \
  -v "$SCRIPT_DIR/config:/app/config:ro" \
  finance-news-briefing python3 scripts/briefing.py \
    --time morning \
    --style briefing \
    --lang de

echo "[$(date)] Morning briefing complete."
