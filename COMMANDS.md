# Finance News — Full Command Reference

## Briefing Generation

```bash
# Morning briefing (English is default)
finance-news briefing --morning

# Evening briefing with WhatsApp delivery
finance-news briefing --evening --send --group "Market Briefing"

# German language option
finance-news briefing --morning --lang de

# Analysis style (more detailed)
finance-news briefing --style analysis
```

## Market Data

```bash
# Market overview (indices + top headlines)
finance-news market

# JSON output for processing
finance-news market --json
```

## Portfolio Management

```bash
finance-news portfolio-list                                              # List portfolio
finance-news portfolio-add NVDA --name "NVIDIA Corporation" --category Tech
finance-news portfolio-remove TSLA                                       # Remove stock
finance-news portfolio-import ~/my_stocks.csv                            # Import from CSV
finance-news portfolio-create                                            # Interactive setup
```

### Portfolio CSV Format

Location: `~/clawd/skills/finance-news/config/portfolio.csv`

```csv
symbol,name,category,notes
AAPL,Apple Inc.,Tech,Core holding
NVDA,NVIDIA Corporation,Tech,AI play
MSFT,Microsoft Corporation,Tech,
```

## Ticker News

```bash
finance-news news AAPL
finance-news news TSLA
```

## Configuration

**Sources:** `~/clawd/skills/finance-news/config/config.json` (legacy fallback: `config/sources.json`) — RSS feeds, market indices by region, and language settings.

## Cron Jobs

### Setup via OpenClaw

```bash
# Add morning briefing cron job
openclaw cron add --schedule "30 6 * * 1-5" \
  --timezone "America/Los_Angeles" \
  --command "bash ~/clawd/skills/finance-news/cron/morning.sh"

# Add evening briefing cron job
openclaw cron add --schedule "0 13 * * 1-5" \
  --timezone "America/Los_Angeles" \
  --command "bash ~/clawd/skills/finance-news/cron/evening.sh"
```

Verify cron jobs are registered and test a dry-run:

```bash
openclaw cron list
bash ~/clawd/skills/finance-news/cron/morning.sh  # Manual test run
```

### Manual Cron (crontab)

```cron
# Morning briefing (6:30 AM PT, weekdays)
30 6 * * 1-5 bash ~/clawd/skills/finance-news/cron/morning.sh

# Evening briefing (1:00 PM PT, weekdays)
0 13 * * 1-5 bash ~/clawd/skills/finance-news/cron/evening.sh
```
