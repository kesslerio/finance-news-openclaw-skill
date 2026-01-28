#!/usr/bin/env bash
# Evening Briefing Cron Job (Docker-based)
# Schedule: 1:00 PM PT (US Market Close at 4:00 PM ET)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date)] Starting evening briefing..."

# Use Docker to avoid NixOS libstdc++ issues
# Mount config/ so runtime changes (portfolio, sources) are picked up
cd "$SCRIPT_DIR"
docker run --rm \
  -v "$SCRIPT_DIR/config:/app/config:ro" \
  finance-news-briefing python3 scripts/briefing.py \
    --time evening \
    --style briefing \
    --lang de

echo "[$(date)] Evening briefing complete."
